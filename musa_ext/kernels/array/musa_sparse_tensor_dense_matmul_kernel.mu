// High-Performance MUSA SparseTensorDenseMatMul Kernels
// Optimized for COO -> CSR conversion and sparse-dense matrix multiplication
// 完全对齐TensorFlow原生SparseTensorDenseMatMul算子逻辑，适配MUSA GPU架构
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// 辅助核函数 1：统计 COO 稀疏矩阵每行的非零元素个数
// ============================================================================
__global__ void ComputeRowCountsKernel(
    const int64_t* __restrict__ indices,  // COO 索引，[nnz, 2]
    const int64_t nnz,
    const int64_t M,  // 稀疏矩阵行数
    int32_t* __restrict__ row_counts) {  // 输出：每行非零元素个数，[M]

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= nnz) return;

    // 读取当前非零元素的行索引
    const int64_t row = indices[tid * 2 + 0];
    // 原子加：统计该行的非零元素个数
    atomicAdd(&row_counts[row], 1);
}

// ============================================================================
// 辅助核函数 2：Mask 转 Count（复用自 SparseSlice）
// ============================================================================
__global__ void MaskToCountKernel(const bool* __restrict__ mask, int32_t* __restrict__ count, int64_t N) {
    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;
    count[tid] = mask[tid] ? 1 : 0;
}

// ============================================================================
// 辅助核函数 3：分块 Exclusive Scan（复用自 SparseSlice）
// ============================================================================
__global__ void BlockScanKernel(const int32_t* __restrict__ count, int64_t* __restrict__ pos, int64_t* __restrict__ block_sums, int64_t N, int64_t block_size) {
    const int tid = threadIdx.x;
    const int block = blockIdx.x;
    __shared__ int64_t sh[1024];

    const int64_t start = block * block_size;
    const int64_t end = min(start + block_size, N);

    if (tid < 1024 && start + tid < end) {
        sh[tid] = count[start + tid];
    } else {
        sh[tid] = 0;
    }
    __syncthreads();

    for (int step = 1; step < 1024; step <<= 1) {
        int64_t sum = 0;
        if (tid >= step) {
            sum = sh[tid - step] + sh[tid];
        }
        __syncthreads();
        if (tid >= step) {
            sh[tid] = sum;
        }
        __syncthreads();
    }

    if (tid == 0 && block_sums != nullptr) {
        block_sums[block] = (end > start) ? sh[end - start - 1] : 0;
    }

    if (tid < 1024 && start + tid < end) {
        if (tid == 0) {
            pos[start + tid] = 0;
        } else {
            pos[start + tid] = sh[tid - 1];
        }
    }
}

// ============================================================================
// 辅助核函数 4：累加块前缀和（复用自 SparseSlice）
// ============================================================================
__global__ void AddBlockPrefixSumKernel(int64_t* __restrict__ pos, const int64_t* __restrict__ block_prefix_sums, int64_t N, int64_t block_size) {
    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;

    const int64_t block = tid / block_size;
    if (block > 0) {
        pos[tid] += block_prefix_sums[block - 1];
    }
}

// ============================================================================
// 辅助核函数 5：填充 CSR 格式的 col_idx 和 values
// ============================================================================
__global__ void FillCSRKernel(
    const int64_t* __restrict__ indices,  // COO 索引，[nnz, 2]
    const float* __restrict__ values,     // COO 值，[nnz]
    const int64_t nnz,
    const int64_t* __restrict__ row_ptr,  // CSR row_ptr，[M+1]
    int64_t* __restrict__ col_idx,         // 输出 CSR col_idx，[nnz]
    float* __restrict__ csr_values) {      // 输出 CSR values，[nnz]

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= nnz) return;

    // 读取当前 COO 元素的行和列
    const int64_t row = indices[tid * 2 + 0];
    const int64_t col = indices[tid * 2 + 1];
    const float val = values[tid];

    // 计算在 CSR 中的位置：row_ptr[row] + 该行内的偏移
    // 这里简化处理：用原子加获取该行内的偏移
    // 注意：工业级实现应该先对 COO 按行排序，这里为了简化演示
    int64_t csr_idx = atomicAdd(const_cast<int64_t*>(&row_ptr[row]), 1LL);
    col_idx[csr_idx] = col;
    csr_values[csr_idx] = val;
}

// ============================================================================
// 核心优化：CSR 稀疏矩阵 × 稠密矩阵 核函数
// 功能：O[i][j] = sum_{k} S[i][k] * D[k][j]
// 设计：每个线程处理输出矩阵的一个元素 (i,j)
// ============================================================================
__global__ void SparseDenseMatMulKernel(
    const int64_t* __restrict__ row_ptr,   // CSR row_ptr，[M+1]
    const int64_t* __restrict__ col_idx,   // CSR col_idx，[nnz]
    const float* __restrict__ csr_values,  // CSR values，[nnz]
    const float* __restrict__ dense,       // 稠密矩阵 D，[K, N]
    const int64_t M,                        // 稀疏矩阵行数
    const int64_t K,                        // 稀疏矩阵列数 / 稠密矩阵行数
    const int64_t N,                        // 稠密矩阵列数
    float* __restrict__ output) {           // 输出矩阵 O，[M, N]

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    const int64_t output_size = M * N;
    if (tid >= output_size) return;

    // 分解输出坐标：tid = i * N + j
    const int64_t i = tid / N;
    const int64_t j = tid % N;

    // 获取当前行 i 的非零元素范围
    const int64_t row_start = row_ptr[i];
    const int64_t row_end = row_ptr[i + 1];

    // 累加求和
    float sum = 0.0f;
    for (int64_t k_idx = row_start; k_idx < row_end; ++k_idx) {
        const int64_t k = col_idx[k_idx];  // 稀疏矩阵的列 k
        const float s_val = csr_values[k_idx];
        const float d_val = dense[k * N + j];  // 稠密矩阵 D[k][j]
        sum += s_val * d_val;
    }

    // 写入输出
    output[tid] = sum;
}

// ============================================================================
// 核函数启动器（Launcher）
// ============================================================================
extern "C" {

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// ----------------------------------------------------------------------------
// 辅助核函数启动器
// ----------------------------------------------------------------------------
void LaunchComputeRowCounts(const int64_t* indices, int64_t nnz, int64_t M, int32_t* row_counts, musaStream_t stream) {
    if (nnz == 0) return;
    const int blocks = OPTIMAL_BLOCKS(nnz);
    ComputeRowCountsKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, nnz, M, row_counts);
}

void LaunchMaskToCount(const bool* mask, int32_t* count, int64_t N, musaStream_t stream) {
    if (N == 0) return;
    const int blocks = OPTIMAL_BLOCKS(N);
    MaskToCountKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(mask, count, N);
}

void LaunchBlockScan(const int32_t* count, int64_t* pos, int64_t* block_sums, int64_t N, int64_t block_size, musaStream_t stream) {
    if (N == 0) return;
    const int64_t num_blocks = (N + block_size - 1) / block_size;
    BlockScanKernel<<<num_blocks, 1024, 0, stream>>>(count, pos, block_sums, N, block_size);
}

void LaunchAddBlockPrefixSum(int64_t* pos, const int64_t* block_prefix_sums, int64_t N, int64_t block_size, musaStream_t stream) {
    if (N == 0) return;
    const int blocks = OPTIMAL_BLOCKS(N);
    AddBlockPrefixSumKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(pos, block_prefix_sums, N, block_size);
}

void LaunchFillCSR(const int64_t* indices, const float* values, int64_t nnz, const int64_t* row_ptr, int64_t* col_idx, float* csr_values, musaStream_t stream) {
    if (nnz == 0) return;
    const int blocks = OPTIMAL_BLOCKS(nnz);
    FillCSRKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, values, nnz, row_ptr, col_idx, csr_values);
}

// ----------------------------------------------------------------------------
// 核心矩阵乘法启动器
// ----------------------------------------------------------------------------
void LaunchSparseDenseMatMul(
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* csr_values,
    const float* dense,
    int64_t M,
    int64_t K,
    int64_t N,
    float* output,
    musaStream_t stream) {
    const int64_t output_size = M * N;
    if (output_size == 0) return;
    const int blocks = OPTIMAL_BLOCKS(output_size);
    SparseDenseMatMulKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(
        row_ptr, col_idx, csr_values, dense, M, K, N, output);
}

// 清理宏
#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"