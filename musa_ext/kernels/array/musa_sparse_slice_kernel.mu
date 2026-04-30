// High-Performance MUSA SparseSlice Kernels
// Optimized for sparse tensor indexing and range checking
// 完全对齐TensorFlow原生SparseSlice算子逻辑，适配MUSA GPU架构
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// ============================================================================
// 核心优化：SparseSlice 掩码生成核函数
// ============================================================================
template <int ndims>
__global__ void GenerateSparseSliceMaskKernel(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ start,
    const int64_t* __restrict__ size,
    const int64_t N,
    bool* __restrict__ mask) {

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;

    bool in_range = true;
    #pragma unroll
    for (int d = 0; d < ndims; ++d) {
        const int64_t idx = indices[tid * ndims + d];
        const int64_t s = start[d];
        const int64_t e = s + size[d];
        if (idx < s || idx >= e) {
            in_range = false;
            break;
        }
    }

    mask[tid] = in_range;
}

// ============================================================================
// 核心优化：SparseSlice 元素收集核函数
// ============================================================================
template <typename T, int ndims>
__global__ void GatherSparseSliceElementsKernel(
    const int64_t* __restrict__ indices,
    const T* __restrict__ values,
    const int64_t* __restrict__ start,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ pos,
    const int64_t N,
    int64_t* __restrict__ out_indices,
    T* __restrict__ out_values) {

    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;

    if (mask[tid]) {
        const int64_t out_idx = pos[tid];
        #pragma unroll
        for (int d = 0; d < ndims; ++d) {
            out_indices[out_idx * ndims + d] = indices[tid * ndims + d] - start[d];
        }
        out_values[out_idx] = values[tid];
    }
}

// ============================================================================
// 新增：纯 MUSA 原生辅助核函数（替代 Thrust）
// ============================================================================

// 1. Mask 转 Count 核函数
__global__ void MaskToCountKernel(const bool* __restrict__ mask, int32_t* __restrict__ count, int64_t N) {
    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;
    count[tid] = mask[tid] ? 1 : 0;
}

// 2. 分块 Exclusive Scan 核函数（第一阶段：块内扫描）
__global__ void BlockScanKernel(const int32_t* __restrict__ count, int64_t* __restrict__ pos, int64_t* __restrict__ block_sums, int64_t N, int64_t block_size) {
    const int tid = threadIdx.x;
    const int block = blockIdx.x;
    __shared__ int64_t sh[1024];

    const int64_t start = block * block_size;
    const int64_t end = min(start + block_size, N);

    // Step 1: 读取块内数据到共享内存
    if (tid < 1024 && start + tid < end) {
        sh[tid] = count[start + tid];
    } else {
        sh[tid] = 0;
    }
    __syncthreads();

    // Step 2: Kogge-Stone Adder 风格的块内扫描
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

    // Step 3: 保存块总和
    if (tid == 0 && block_sums != nullptr) {
        block_sums[block] = (end > start) ? sh[end - start - 1] : 0;
    }

    // Step 4: 写入块内扫描结果（Exclusive）
    if (tid < 1024 && start + tid < end) {
        if (tid == 0) {
            pos[start + tid] = 0;
        } else {
            pos[start + tid] = sh[tid - 1];
        }
    }
}

// 3. 累加块前缀和核函数（第二阶段：块间累加）
__global__ void AddBlockPrefixSumKernel(int64_t* __restrict__ pos, const int64_t* __restrict__ block_prefix_sums, int64_t N, int64_t block_size) {
    const int64_t tid = blockIdx.x * 256 + threadIdx.x;
    if (tid >= N) return;

    const int64_t block = tid / block_size;
    if (block > 0) {
        pos[tid] += block_prefix_sums[block - 1];
    }
}

// ============================================================================
// 核函数启动器（Launcher）
// ============================================================================
extern "C" {

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// ----------------------------------------------------------------------------
// 掩码生成启动器（1D-5D）
// ----------------------------------------------------------------------------
#define DEFINE_GENERATE_MASK_LAUNCHER(ndims, Name) \
  void Name( \
      const int64_t* indices, \
      const int64_t* start, \
      const int64_t* size, \
      int64_t N, \
      bool* mask, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GenerateSparseSliceMaskKernel<ndims><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, start, size, N, mask); \
  }

DEFINE_GENERATE_MASK_LAUNCHER(1, LaunchGenerateSparseSliceMask1D)
DEFINE_GENERATE_MASK_LAUNCHER(2, LaunchGenerateSparseSliceMask2D)
DEFINE_GENERATE_MASK_LAUNCHER(3, LaunchGenerateSparseSliceMask3D)
DEFINE_GENERATE_MASK_LAUNCHER(4, LaunchGenerateSparseSliceMask4D)
DEFINE_GENERATE_MASK_LAUNCHER(5, LaunchGenerateSparseSliceMask5D)

// ----------------------------------------------------------------------------
// 元素收集启动器（所有类型+1D-5D）
// ----------------------------------------------------------------------------
#define DEFINE_GATHER_ELEMENTS_LAUNCHER(T, ndims, Name) \
  void Name( \
      const int64_t* indices, \
      const T* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      T* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<T, ndims><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, values, start, mask, pos, N, out_indices, out_values); \
  }

#define DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(T, Prefix) \
  DEFINE_GATHER_ELEMENTS_LAUNCHER(T, 1, Prefix##1D) \
  DEFINE_GATHER_ELEMENTS_LAUNCHER(T, 2, Prefix##2D) \
  DEFINE_GATHER_ELEMENTS_LAUNCHER(T, 3, Prefix##3D) \
  DEFINE_GATHER_ELEMENTS_LAUNCHER(T, 4, Prefix##4D) \
  DEFINE_GATHER_ELEMENTS_LAUNCHER(T, 5, Prefix##5D)

// 基础类型
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(float, LaunchGatherSparseSliceElementsFloat)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(double, LaunchGatherSparseSliceElementsDouble)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(int32_t, LaunchGatherSparseSliceElementsInt32)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(int64_t, LaunchGatherSparseSliceElementsInt64)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(uint8_t, LaunchGatherSparseSliceElementsUInt8)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(uint16_t, LaunchGatherSparseSliceElementsUInt16)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(int8_t, LaunchGatherSparseSliceElementsInt8)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(int16_t, LaunchGatherSparseSliceElementsInt16)
DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS(bool, LaunchGatherSparseSliceElementsBool)

// FP16 (Eigen::half)
#define DEFINE_GATHER_ELEMENTS_HALF_FOR_ALL_NDIMS(Prefix) \
  void Prefix##1D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<half, 1><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const half*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<half*>(out_values)); \
  } \
  void Prefix##2D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<half, 2><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const half*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<half*>(out_values)); \
  } \
  void Prefix##3D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<half, 3><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const half*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<half*>(out_values)); \
  } \
  void Prefix##4D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<half, 4><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const half*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<half*>(out_values)); \
  } \
  void Prefix##5D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<half, 5><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const half*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<half*>(out_values)); \
  }

DEFINE_GATHER_ELEMENTS_HALF_FOR_ALL_NDIMS(LaunchGatherSparseSliceElementsHalf)

// BF16
#define DEFINE_GATHER_ELEMENTS_BF16_FOR_ALL_NDIMS(Prefix) \
  void Prefix##1D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<__mt_bfloat16, 1><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<__mt_bfloat16*>(out_values)); \
  } \
  void Prefix##2D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<__mt_bfloat16, 2><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<__mt_bfloat16*>(out_values)); \
  } \
  void Prefix##3D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<__mt_bfloat16, 3><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<__mt_bfloat16*>(out_values)); \
  } \
  void Prefix##4D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<__mt_bfloat16, 4><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<__mt_bfloat16*>(out_values)); \
  } \
  void Prefix##5D( \
      const int64_t* indices, \
      const void* values, \
      const int64_t* start, \
      const bool* mask, \
      const int64_t* pos, \
      int64_t N, \
      int64_t* out_indices, \
      void* out_values, \
      musaStream_t stream) { \
    if (N == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(N); \
    GatherSparseSliceElementsKernel<__mt_bfloat16, 5><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, \
        out_indices, reinterpret_cast<__mt_bfloat16*>(out_values)); \
  }

DEFINE_GATHER_ELEMENTS_BF16_FOR_ALL_NDIMS(LaunchGatherSparseSliceElementsBFloat16)

// ----------------------------------------------------------------------------
// 辅助核函数启动器（替代 Thrust）
// ----------------------------------------------------------------------------
void LaunchMaskToCount(const bool* mask, int32_t* count, int64_t N, musaStream_t stream) {
    if (N == 0) return;
    const int blocks = (N + 256 - 1) / 256;
    MaskToCountKernel<<<blocks, 256, 0, stream>>>(mask, count, N);
}

void LaunchBlockScan(const int32_t* count, int64_t* pos, int64_t* block_sums, int64_t N, int64_t block_size, musaStream_t stream) {
    if (N == 0) return;
    const int64_t num_blocks = (N + block_size - 1) / block_size;
    BlockScanKernel<<<num_blocks, 1024, 0, stream>>>(count, pos, block_sums, N, block_size);
}

void LaunchAddBlockPrefixSum(int64_t* pos, const int64_t* block_prefix_sums, int64_t N, int64_t block_size, musaStream_t stream) {
    if (N == 0) return;
    const int blocks = (N + 256 - 1) / 256;
    AddBlockPrefixSumKernel<<<blocks, 256, 0, stream>>>(pos, block_prefix_sums, N, block_size);
}

// 清理宏
#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS
#undef DEFINE_GENERATE_MASK_LAUNCHER
#undef DEFINE_GATHER_ELEMENTS_LAUNCHER
#undef DEFINE_GATHER_ELEMENTS_FOR_ALL_NDIMS
#undef DEFINE_GATHER_ELEMENTS_HALF_FOR_ALL_NDIMS
#undef DEFINE_GATHER_ELEMENTS_BF16_FOR_ALL_NDIMS

}  // extern "C"