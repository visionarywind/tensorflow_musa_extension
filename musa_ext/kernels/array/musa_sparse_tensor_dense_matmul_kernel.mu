// musa_sparse_tensor_dense_matmul_kernel.mu

#include <musa_runtime.h>
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
// Kernel 1: Compute Row Counts from COO indices
// -----------------------------------------------------------------------------
__global__ void ComputeRowCountsKernel(
    const int64_t* __restrict__ indices,
    const int64_t nnz,
    int32_t* __restrict__ row_counts) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    // COO indices format: [row0, col0, row1, col1, ...]
    const int64_t row = indices[tid * 2]; 
    atomicAdd(&row_counts[row], 1);
}

// -----------------------------------------------------------------------------
// Kernel 2: Parallel Exclusive Scan (Blelloch Algorithm)
// Converts row_counts (int32) to row_ptr (int64)
// Step 1: Local Exclusive Scan within each block
// -----------------------------------------------------------------------------
__global__ void ScanLocalKernel(
    const int32_t* __restrict__ input,
    int64_t* __restrict__ output,
    int32_t* __restrict__ block_sums,
    int64_t n) {
    
    extern __shared__ int32_t temp[];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int globalIdx = blockIdx.x * blockSize + tid;
    
    // Load input into shared memory
    if (globalIdx < n) {
        temp[tid] = input[globalIdx];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Inclusive Scan (Up-Sweep / Reduce phase can be merged, but here we do standard Hillis-Steele for simplicity in shared mem)
    // Actually, for Exclusive Scan, it's often easier to do Inclusive then shift.
    // Let's do Inclusive Scan first.
    
    for (int offset = 1; offset < blockSize; offset <<= 1) {
        int32_t t = 0;
        if (tid >= offset) {
            t = temp[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    // Save block sum (last element of inclusive scan)
    if (tid == blockSize - 1) {
        block_sums[blockIdx.x] = temp[tid];
    }

    // Convert to Exclusive: shift right by 1, set first to 0
    int32_t exclusive_val = (tid == 0) ? 0 : temp[tid - 1];
    
    if (globalIdx < n) {
        output[globalIdx] = static_cast<int64_t>(exclusive_val);
    }
}

// -----------------------------------------------------------------------------
// Kernel 3: Scan Block Sums (Small array, single block)
// Computes exclusive scan of the block_sums array
// -----------------------------------------------------------------------------
__global__ void ScanBlockSumsKernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int n) {
    
    if (blockIdx.x > 0) return; // Only one block needed
    
    extern __shared__ int32_t temp[];
    int tid = threadIdx.x;
    
    if (tid < n) temp[tid] = input[tid];
    else temp[tid] = 0;
    __syncthreads();
    
    // Inclusive Scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int32_t t = 0;
        if (tid >= offset) t = temp[tid - offset];
        __syncthreads();
        if (tid >= offset) temp[tid] += t;
        __syncthreads();
    }
    
    // Convert to Exclusive
    int32_t exclusive_val = (tid == 0) ? 0 : temp[tid - 1];
    
    if (tid < n) {
        output[tid] = exclusive_val;
    }
    
    // We also need the total sum for the last block offset if we were doing inclusive, 
    // but for exclusive scan of blocks, the last element is not used as an offset for subsequent data 
    // unless we are doing inclusive. 
    // Wait, for adding offsets:
    // Block 0 elements add 0.
    // Block 1 elements add BlockSum[0].
    // Block i elements add Sum(BlockSum[0..i-1]).
    // So we need the Exclusive Scan of block_sums.
}

// -----------------------------------------------------------------------------
// Kernel 4: Add Block Offsets to Local Results
// -----------------------------------------------------------------------------
__global__ void AddOffsetsKernel(
    int64_t* __restrict__ data,
    const int32_t* __restrict__ block_offsets, // Exclusive scan of block sums
    int64_t n,
    int blockSize) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    int blockId = blockIdx.x;
    if (blockId > 0) {
        data[tid] += static_cast<int64_t>(block_offsets[blockId]);
    }
}

// -----------------------------------------------------------------------------
// Kernel 5: Set Last Element of Row Ptr
// row_ptr[M] = nnz
// -----------------------------------------------------------------------------
__global__ void SetLastRowPtrKernel(int64_t* row_ptr, int64_t M, int64_t nnz) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        row_ptr[M] = nnz;
    }
}

// -----------------------------------------------------------------------------
// Kernel 6: Fill CSR using Atomic Counters
// -----------------------------------------------------------------------------
__global__ void FillCSRKernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ values,
    const int64_t nnz,
    int64_t* __restrict__ row_ptr_atomic, // Writable copy of row_ptr
    int64_t* __restrict__ col_idx,
    float* __restrict__ csr_values) {

    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    const int64_t row = indices[tid * 2];
    const int64_t col = indices[tid * 2 + 1];
    const float val = values[tid];

    // Atomically get the write position and increment the counter
    int64_t insert_pos = atomicAdd(&row_ptr_atomic[row], 1LL);
    
    col_idx[insert_pos] = col;
    csr_values[insert_pos] = val;
}

// -----------------------------------------------------------------------------
// Kernel 7: Sparse Dense MatMul
// -----------------------------------------------------------------------------
__global__ void SparseDenseMatMulKernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ csr_values,
    const float* __restrict__ dense,
    const int64_t M,
    const int64_t K,
    const int64_t N,
    float* __restrict__ output) {
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
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
    ComputeRowCountsKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, nnz, row_counts);
}

// Helper to perform the full Exclusive Scan pipeline
void LaunchExclusiveScanPipeline(
    const int32_t* d_input, 
    int64_t* d_output, 
    int32_t* d_block_sums, 
    int64_t n, 
    musaStream_t stream) {
    
    if (n == 0) return;

    int blockSize = OPTIMAL_THREADS;
    int numBlocks = (static_cast<int>(n) + blockSize - 1) / blockSize;

    // Step 1: Local Exclusive Scan
    ScanLocalKernel<<<numBlocks, blockSize, blockSize * sizeof(int32_t), stream>>>(
        d_input, d_output, d_block_sums, n);

    // Step 2: Scan Block Sums (if more than 1 block)
    if (numBlocks > 1) {
        ScanBlockSumsKernel<<<1, blockSize, blockSize * sizeof(int32_t), stream>>>(
            d_block_sums, d_block_sums, numBlocks);
        
        // Step 3: Add Offsets
        AddOffsetsKernel<<<numBlocks, blockSize, 0, stream>>>(
            d_output, d_block_sums, n, blockSize);
    }
}

void LaunchFillCSR(const int64_t* indices, const float* values, int64_t nnz, 
                   int64_t* row_ptr_atomic, int64_t* col_idx, float* csr_values, musaStream_t stream) {
    if (nnz == 0) return;
    const int blocks = OPTIMAL_BLOCKS(nnz);
    FillCSRKernel<<<blocks, OPTIMAL_THREADS, 0, stream>>>(indices, values, nnz, row_ptr_atomic, col_idx, csr_values);
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

void LaunchSetLastRowPtr(int64_t* row_ptr, int64_t M, int64_t nnz, musaStream_t stream) {
    SetLastRowPtrKernel<<<1, 1, 0, stream>>>(row_ptr, M, nnz);
}

#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C"