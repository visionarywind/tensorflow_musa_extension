// musa_sparse_slice_kernel.mu

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// Device Helpers
// -----------------------------------------------------------------------------
__device__ int64_t atomicAdd(int64_t* address, int64_t val) {
    unsigned long long* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + static_cast<unsigned long long>(val));
    } while (assumed != old);
    return static_cast<int64_t>(old);
}

// -----------------------------------------------------------------------------
// Kernel 1: Generate Mask AND Count (Fused)
// -----------------------------------------------------------------------------
template <int ndims>
__global__ void GenerateMaskAndCountKernel(
    const int64_t* __restrict__ indices,
    const int64_t* __restrict__ start,
    const int64_t* __restrict__ size,
    const int64_t N,
    bool* __restrict__ mask,
    int32_t* __restrict__ count) {

    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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

    bool m = in_range;
    mask[tid] = m;
    count[tid] = m ? 1 : 0;
}

// -----------------------------------------------------------------------------
// Kernel 2: Parallel Exclusive Scan - Local Step
// Input: int32_t counts, Output: int64_t pos (exclusive scan result)
// -----------------------------------------------------------------------------
__global__ void ScanLocalKernel(
    const int32_t* __restrict__ input,
    int64_t* __restrict__ output, // pos
    int64_t* __restrict__ block_sums,
    int64_t N,
    int64_t blockSize) {
    
    extern __shared__ int32_t s_data[]; // Shared memory for int32 input
    
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockSize + tid;
    
    // Load input into shared memory
    if (globalIdx < N) {
        s_data[tid] = input[globalIdx];
    } else {
        s_data[tid] = 0;
    }
    __syncthreads();

    // Inclusive Scan (Hillis-Steele) on shared memory
    for (int offset = 1; offset < blockSize; offset <<= 1) {
        int32_t t = 0;
        if (tid >= offset) {
            t = s_data[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            s_data[tid] += t;
        }
        __syncthreads();
    }

    // Save block sum (inclusive sum of this block)
    if (tid == blockSize - 1) {
        block_sums[blockIdx.x] = static_cast<int64_t>(s_data[tid]);
    }

    // Convert to Exclusive Scan result for this block
    int32_t exclusive_val = (tid == 0) ? 0 : s_data[tid - 1];
    
    if (globalIdx < N) {
        output[globalIdx] = static_cast<int64_t>(exclusive_val);
    }
}

// -----------------------------------------------------------------------------
// Kernel 3: Scan Block Sums (Small array, single block)
// Input/Output: int64_t
// -----------------------------------------------------------------------------
__global__ void ScanBlockSumsKernel(
    const int64_t* __restrict__ input,
    int64_t* __restrict__ output,
    int n) {
    
    if (blockIdx.x > 0) return;
    
    extern __shared__ int64_t s_data_i64[]; // Shared memory for int64
    int tid = threadIdx.x;
    
    if (tid < n) s_data_i64[tid] = input[tid];
    else s_data_i64[tid] = 0;
    __syncthreads();
    
    // Inclusive Scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t t = 0;
        if (tid >= offset) t = s_data_i64[tid - offset];
        __syncthreads();
        if (tid >= offset) s_data_i64[tid] += t;
        __syncthreads();
    }
    
    // Convert to Exclusive
    int64_t exclusive_val = (tid == 0) ? 0 : s_data_i64[tid - 1];
    
    if (tid < n) {
        output[tid] = exclusive_val;
    }
}

// -----------------------------------------------------------------------------
// Kernel 4: Add Block Offsets
// -----------------------------------------------------------------------------
__global__ void AddOffsetsKernel(
    int64_t* __restrict__ data,
    const int64_t* __restrict__ block_offsets,
    int64_t N,
    int64_t blockSize) {
    
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    
    int64_t blockId = blockIdx.x;
    if (blockId > 0) {
        data[tid] += block_offsets[blockId];
    }
}

// -----------------------------------------------------------------------------
// Kernel 5: Gather Elements
// -----------------------------------------------------------------------------
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

    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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

// -----------------------------------------------------------------------------
// Kernel 6: Sum Block Sums to get Total Count
// -----------------------------------------------------------------------------
__global__ void SumBlockSumsKernel(
    const int64_t* __restrict__ block_sums,
    int64_t* __restrict__ total_count,
    int num_blocks) {
    
    if (blockIdx.x > 0) return;
    
    extern __shared__ int64_t s_reduce[];
    int tid = threadIdx.x;
    
    if (tid < num_blocks) s_reduce[tid] = block_sums[tid];
    else s_reduce[tid] = 0;
    __syncthreads();
    
    // Reduction Sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_reduce[tid] += s_reduce[tid + offset];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *total_count = s_reduce[0];
    }
}

// -----------------------------------------------------------------------------
// C Interface Implementations
// -----------------------------------------------------------------------------
extern "C" {

#define OPTIMAL_THREADS 256
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// --- Fused Mask and Count Launchers ---
template <int ndims>
void LaunchGenerateMaskAndCountImpl(
    const int64_t* indices,
    const int64_t* start,
    const int64_t* size,
    int64_t N,
    bool* mask,
    int32_t* count,
    musaStream_t stream) {
    
    if (N == 0) return;
    const int blocks = OPTIMAL_BLOCKS(N);
    GenerateMaskAndCountKernel<ndims><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
        indices, start, size, N, mask, count);
}

void LaunchGenerateSparseSliceMaskAndCount1D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream) {
    LaunchGenerateMaskAndCountImpl<1>(indices, start, size, N, mask, count, stream);
}
void LaunchGenerateSparseSliceMaskAndCount2D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream) {
    LaunchGenerateMaskAndCountImpl<2>(indices, start, size, N, mask, count, stream);
}
void LaunchGenerateSparseSliceMaskAndCount3D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream) {
    LaunchGenerateMaskAndCountImpl<3>(indices, start, size, N, mask, count, stream);
}
void LaunchGenerateSparseSliceMaskAndCount4D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream) {
    LaunchGenerateMaskAndCountImpl<4>(indices, start, size, N, mask, count, stream);
}
void LaunchGenerateSparseSliceMaskAndCount5D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream) {
    LaunchGenerateMaskAndCountImpl<5>(indices, start, size, N, mask, count, stream);
}

// --- Scan Pipeline ---
void LaunchFullScanPipeline(
    const int32_t* d_counts,
    int64_t* d_pos,
    int64_t* d_block_sums,
    int64_t* d_block_prefix_sums,
    int64_t N,
    int64_t block_size,
    int64_t num_blocks,
    musaStream_t stream) {
    
    if (N == 0) return;

    // Step 1: Local Exclusive Scan
    ScanLocalKernel<<<num_blocks, block_size, block_size * sizeof(int32_t), stream>>>(
        d_counts, d_pos, d_block_sums, N, block_size);

    if (num_blocks > 1) {
        // Step 2: Scan Block Sums
        ScanBlockSumsKernel<<<1, block_size, block_size * sizeof(int64_t), stream>>>(
            d_block_sums, d_block_prefix_sums, num_blocks);
        
        // Step 3: Add Offsets
        AddOffsetsKernel<<<num_blocks, OPTIMAL_THREADS, 0, stream>>>(
            d_pos, d_block_prefix_sums, N, block_size);
    }
}

void LaunchGetTotalCount(
    const int64_t* d_block_sums,
    int64_t* d_total_count,
    int64_t num_blocks,
    musaStream_t stream) {
    
    if (num_blocks == 0) {
        musaMemsetAsync(d_total_count, 0, sizeof(int64_t), stream);
        return;
    }
    
    int threads = OPTIMAL_THREADS;
    if (num_blocks < threads) threads = num_blocks;
    
    SumBlockSumsKernel<<<1, threads, threads * sizeof(int64_t), stream>>>(
        d_block_sums, d_total_count, num_blocks);
}

// --- Gather Launchers ---
template <typename T, int ndims>
void LaunchGatherImpl(
    const int64_t* indices,
    const T* values,
    const int64_t* start,
    const bool* mask,
    const int64_t* pos,
    int64_t N,
    int64_t* out_indices,
    T* out_values,
    musaStream_t stream) {
    
    if (N == 0) return;
    const int blocks = OPTIMAL_BLOCKS(N);
    GatherSparseSliceElementsKernel<T, ndims><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
        indices, values, start, mask, pos, N, out_indices, out_values);
}

// Macro to define C wrappers for Gather
#define DEFINE_GATHER_LAUNCHER(T, ndims, Name) \
  void Name(const int64_t* indices, const T* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, T* out_values, musaStream_t stream) { \
      LaunchGatherImpl<T, ndims>(indices, values, start, mask, pos, N, out_indices, out_values, stream); \
  }

#define DEFINE_GATHER_FOR_ALL_NDIMS(T, Prefix) \
  DEFINE_GATHER_LAUNCHER(T, 1, Prefix##1D) \
  DEFINE_GATHER_LAUNCHER(T, 2, Prefix##2D) \
  DEFINE_GATHER_LAUNCHER(T, 3, Prefix##3D) \
  DEFINE_GATHER_LAUNCHER(T, 4, Prefix##4D) \
  DEFINE_GATHER_LAUNCHER(T, 5, Prefix##5D)

DEFINE_GATHER_FOR_ALL_NDIMS(float, LaunchGatherSparseSliceElementsFloat)
DEFINE_GATHER_FOR_ALL_NDIMS(double, LaunchGatherSparseSliceElementsDouble)
DEFINE_GATHER_FOR_ALL_NDIMS(int32_t, LaunchGatherSparseSliceElementsInt32)
DEFINE_GATHER_FOR_ALL_NDIMS(int64_t, LaunchGatherSparseSliceElementsInt64)
DEFINE_GATHER_FOR_ALL_NDIMS(uint8_t, LaunchGatherSparseSliceElementsUInt8)
DEFINE_GATHER_FOR_ALL_NDIMS(uint16_t, LaunchGatherSparseSliceElementsUInt16)
DEFINE_GATHER_FOR_ALL_NDIMS(int8_t, LaunchGatherSparseSliceElementsInt8)
DEFINE_GATHER_FOR_ALL_NDIMS(int16_t, LaunchGatherSparseSliceElementsInt16)
DEFINE_GATHER_FOR_ALL_NDIMS(bool, LaunchGatherSparseSliceElementsBool)

// Half Wrappers
#define DEFINE_HALF_GATHER_LAUNCHER(ndims, Name) \
  void Name(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream) { \
      LaunchGatherImpl<half, ndims>(indices, reinterpret_cast<const half*>(values), start, mask, pos, N, out_indices, reinterpret_cast<half*>(out_values), stream); \
  }

DEFINE_HALF_GATHER_LAUNCHER(1, LaunchGatherSparseSliceElementsHalf1D)
DEFINE_HALF_GATHER_LAUNCHER(2, LaunchGatherSparseSliceElementsHalf2D)
DEFINE_HALF_GATHER_LAUNCHER(3, LaunchGatherSparseSliceElementsHalf3D)
DEFINE_HALF_GATHER_LAUNCHER(4, LaunchGatherSparseSliceElementsHalf4D)
DEFINE_HALF_GATHER_LAUNCHER(5, LaunchGatherSparseSliceElementsHalf5D)

// BFloat16 Wrappers
#define DEFINE_BF16_GATHER_LAUNCHER(ndims, Name) \
  void Name(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream) { \
      LaunchGatherImpl<__mt_bfloat16, ndims>(indices, reinterpret_cast<const __mt_bfloat16*>(values), start, mask, pos, N, out_indices, reinterpret_cast<__mt_bfloat16*>(out_values), stream); \
  }

DEFINE_BF16_GATHER_LAUNCHER(1, LaunchGatherSparseSliceElementsBFloat161D)
DEFINE_BF16_GATHER_LAUNCHER(2, LaunchGatherSparseSliceElementsBFloat162D)
DEFINE_BF16_GATHER_LAUNCHER(3, LaunchGatherSparseSliceElementsBFloat163D)
DEFINE_BF16_GATHER_LAUNCHER(4, LaunchGatherSparseSliceElementsBFloat164D)
DEFINE_BF16_GATHER_LAUNCHER(5, LaunchGatherSparseSliceElementsBFloat165D)

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS
#undef DEFINE_GATHER_LAUNCHER
#undef DEFINE_GATHER_FOR_ALL_NDIMS
#undef DEFINE_HALF_GATHER_LAUNCHER
#undef DEFINE_BF16_GATHER_LAUNCHER

}  // extern "C"