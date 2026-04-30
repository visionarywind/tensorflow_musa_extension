// musa_sparse_tensor_dense_matmul_op.cc

#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

using namespace tensorflow;

extern "C" {
void LaunchComputeRowCounts(const int64_t* indices, int64_t nnz, int64_t M,
                            int32_t* row_counts, musaStream_t stream);

// New Scan Pipeline Function
void LaunchExclusiveScanPipeline(
    const int32_t* d_input, 
    int64_t* d_output, 
    int32_t* d_block_sums, 
    int64_t n, 
    musaStream_t stream);

void LaunchFillCSR(const int64_t* indices, const float* values, int64_t nnz,
                   int64_t* row_ptr_atomic, int64_t* col_idx, float* csr_values,
                   musaStream_t stream);
void LaunchSparseDenseMatMul(const int64_t* row_ptr, const int64_t* col_idx,
                             const float* csr_values, const float* dense,
                             int64_t M, int64_t K, int64_t N, float* output,
                             musaStream_t stream);
                             
void LaunchSetLastRowPtr(int64_t* row_ptr, int64_t M, int64_t nnz, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T>
class MusaSparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit MusaSparseTensorDenseMatMulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint_b", &adjoint_b_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& dense_shape = context->input(2);
    const Tensor& dense = context->input(3);

    const int64_t nnz = indices.dim_size(0);
    const int64_t ndims = indices.dim_size(1);
    const int64_t M = dense_shape.flat<int64_t>()(0);
    const int64_t K = dense_shape.flat<int64_t>()(1);
    const int64_t dense_rows = dense.dim_size(0);
    const int64_t N = dense.dim_size(1);

    OP_REQUIRES(
        context, ndims == 2,
        errors::InvalidArgument("Sparse tensor must be 2D, got ", ndims, "D"));
    OP_REQUIRES(context, values.dim_size(0) == nnz,
                errors::InvalidArgument("Values size must match indices size: ",
                                        values.dim_size(0), " vs ", nnz));
    OP_REQUIRES(context, dense_rows == K,
                errors::InvalidArgument(
                    "Dense matrix rows must match sparse matrix cols: ",
                    dense_rows, " vs ", K));

    if (nnz == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({M, N}), &output));
      output->flat<T>().setZero();
      return;
    }

    musaStream_t raw_stream = GetMusaStreamByCtx(context);

    // --- Memory Allocation ---
    
    // 1. Row Counts (Int32, size M)
    Tensor d_row_counts_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({M}),
                                                   &d_row_counts_tensor));
    
    // 2. Row Ptr (Int64, size M+1) - Final result for MatMul
    Tensor d_row_ptr_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_tensor));
    
    // 3. Row Ptr Atomic (Int64, size M+1) - Temporary for FillCSR
    Tensor d_row_ptr_atomic_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_atomic_tensor));

    // 4. Col Idx (Int64, size nnz)
    Tensor d_col_idx_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({nnz}),
                                                   &d_col_idx_tensor));

    // 5. CSR Values (Float, size nnz)
    Tensor d_csr_values_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({nnz}),
                                                   &d_csr_values_tensor));

    // 6. Block Sums for Scan (Int32). 
    // Number of blocks = ceil(M / 256)
    int num_scan_blocks = (static_cast<int>(M) + 255) / 256;
    Tensor d_block_sums_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({num_scan_blocks}),
                                                   &d_block_sums_tensor));

    // --- Pointers ---
    int32_t* d_row_counts = d_row_counts_tensor.flat<int32_t>().data();
    int64_t* d_row_ptr = d_row_ptr_tensor.flat<int64_t>().data();
    int64_t* d_row_ptr_atomic = d_row_ptr_atomic_tensor.flat<int64_t>().data();
    int64_t* d_col_idx = d_col_idx_tensor.flat<int64_t>().data();
    float* d_csr_values = d_csr_values_tensor.flat<float>().data();
    int32_t* d_block_sums = d_block_sums_tensor.flat<int32_t>().data();

    // --- Execution Pipeline (Fully Async) ---

    // 1. Initialize row_counts to 0
    musaMemsetAsync(d_row_counts, 0, M * sizeof(int32_t), raw_stream);

    // 2. Compute Row Counts (COO -> Counts)
    LaunchComputeRowCounts(indices.flat<int64_t>().data(), nnz, M, d_row_counts, raw_stream);

    // 3. Exclusive Scan (Counts -> Row Ptr)
    // Input: d_row_counts (size M)
    // Output: d_row_ptr (size M) -> contains ptr[0] to ptr[M-1]
    // Temp: d_block_sums
    LaunchExclusiveScanPipeline(d_row_counts, d_row_ptr, d_block_sums, M, raw_stream);

    // 4. Set the last element of row_ptr: row_ptr[M] = nnz
    LaunchSetLastRowPtr(d_row_ptr, M, nnz, raw_stream);

    // 5. Copy d_row_ptr to d_row_ptr_atomic for FillCSR
    // FillCSR will atomically increment these values, so we need a clean copy for MatMul later.
    musaMemcpyAsync(d_row_ptr_atomic, d_row_ptr, (M + 1) * sizeof(int64_t),
                    musaMemcpyDeviceToDevice, raw_stream);

    // 6. Fill CSR Data (Col Indices and Values)
    LaunchFillCSR(indices.flat<int64_t>().data(), values.flat<float>().data(),
                  nnz, d_row_ptr_atomic, d_col_idx, d_csr_values, raw_stream);

    // 7. Allocate Output Tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({M, N}), &output));

    // 8. Sparse Dense MatMul
    // Uses the original, unmodified d_row_ptr
    LaunchSparseDenseMatMul(d_row_ptr, d_col_idx, d_csr_values,
                            dense.flat<float>().data(), M, K, N,
                            output->flat<float>().data(), raw_stream);

    // 9. Error Check
    musaError_t err = musaGetLastError();
    OP_REQUIRES(
        context, err == musaSuccess,
        errors::Internal("MUSA SparseTensorDenseMatMul kernel launch failed: ",
                         musaGetErrorString(err)));
  }

 private:
  bool adjoint_a_;
  bool adjoint_b_;
};

#define REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL(T)                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseTensorDenseMatMul").Device("MUSA").TypeConstraint<T>("T"), \
      MusaSparseTensorDenseMatMulOp<T>);

REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL(float);

#undef REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL

}  // namespace musa
}  // namespace tensorflow