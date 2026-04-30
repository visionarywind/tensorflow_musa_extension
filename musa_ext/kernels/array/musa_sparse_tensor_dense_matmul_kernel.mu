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

__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + static_cast<unsigned long long>(val));
    } while (assumed != old);

    return static_cast<int64_t>(old);
}

__device__ long long atomicAdd(long long* address, long long val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + val);
    } while (assumed != old);

    return (long long)old;
}

__global__ void ComputeRowCountsKernel(
    const int64_t* __restrict__ indices,
    const int64_t nnz,
    const int64_t M,
    int32_t* __restrict__ row_counts) {

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= nnz) return;

    const int64_t row = indices[tid * 2 + 0];
    atomicAdd(&row_counts[row], 1);
}

__global__ void FillCSRKernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ values,
    const int64_t nnz,
    int64_t* __restrict__ row_ptr,
    int64_t* __restrict__ col_idx,
    float* __restrict__ csr_values) {

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= nnz) return;

    const int64_t row = indices[tid * 2 + 0];
    const int64_t col = indices[tid * 2 + 1];
    const float val = values[tid];

    int64_t csr_idx = atomicAdd(&row_ptr[row], 1LL);
    col_idx[csr_idx] = col;
    csr_values[csr_idx] = val;
}

__global__ void SparseDenseMatMulKernel(
    const int64_t* __restrict__ row_ptr,   // CSR row_ptr，[M+1]
    const int64_t* __restrict__ col_idx,   // CSR col_idx，[nnz]
    const float* __restrict__ csr_values,  // CSR values，[nnz]
    const float* __restrict__ dense,
    const int64_t M,
    const int64_t K,
    const int64_t N,
    float* __restrict__ output) {
    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    const int64_t output_size = M * N;
    if (tid >= output_size) return;

    const int64_t i = tid / N;
    const int64_t j = tid % N;

    const int64_t row_start = row_ptr[i];
    const int64_t row_end = row_ptr[i + 1];

    float sum = 0.0f;
    for (int64_t k_idx = row_start; k_idx < row_end; ++k_idx) {
        const int64_t k = col_idx[k_idx];
        const float s_val = csr_values[k_idx];
        const float d_val = dense[k * N + j];
        sum += s_val * d_val;
    }

    output[tid] = sum;
}

extern "C" {

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

void LaunchComputeRowCounts(const int64_t* indices, int64_t nnz, int64_t M, int32_t* row_counts, musaStream_t stream) {
    if (nnz == 0) return;
    const int blocks = OPTIMAL_BLOCKS(nnz);
    ComputeRowCountsKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, nnz, M, row_counts);
}

void LaunchFillCSR(const int64_t* indices, const float* values, int64_t nnz, int64_t* row_ptr, int64_t* col_idx, float* csr_values, musaStream_t stream) {
    if (nnz == 0) return;
    const int blocks = OPTIMAL_BLOCKS(nnz);
    FillCSRKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, values, nnz, row_ptr, col_idx, csr_values);
}

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

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"