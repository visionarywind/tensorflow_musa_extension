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
void LaunchFillCSR(const int64_t* indices, const float* values, int64_t nnz,
                   int64_t* row_ptr, int64_t* col_idx, float* csr_values,
                   musaStream_t stream);
void LaunchSparseDenseMatMul(const int64_t* row_ptr, const int64_t* col_idx,
                             const float* csr_values, const float* dense,
                             int64_t M, int64_t K, int64_t N, float* output,
                             musaStream_t stream);
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

    Tensor d_row_counts_tensor;
    Tensor d_row_ptr_tensor;
    Tensor d_col_idx_tensor;
    Tensor d_csr_values_tensor;
    Tensor d_row_ptr_copy_tensor;

    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({M}),
                                                   &d_row_counts_tensor));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_tensor));

    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({nnz}),
                                                   &d_col_idx_tensor));

    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({nnz}),
                                                   &d_csr_values_tensor));

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_copy_tensor));

    int32_t* d_row_counts = d_row_counts_tensor.flat<int32_t>().data();
    int64_t* d_row_ptr = d_row_ptr_tensor.flat<int64_t>().data();
    int64_t* d_col_idx = d_col_idx_tensor.flat<int64_t>().data();
    float* d_csr_values = d_csr_values_tensor.flat<float>().data();
    int64_t* d_row_ptr_copy = d_row_ptr_copy_tensor.flat<int64_t>().data();

    musaMemsetAsync(d_row_counts, 0, M * sizeof(int32_t), raw_stream);

    LaunchComputeRowCounts(indices.flat<int64_t>().data(), nnz, M, d_row_counts,
                           raw_stream);

    std::vector<int32_t> h_row_counts(M);
    std::vector<int64_t> h_row_ptr(M + 1);

    musaStreamSynchronize(raw_stream);

    musaMemcpy(h_row_counts.data(), d_row_counts, M * sizeof(int32_t),
               musaMemcpyDeviceToHost);

    h_row_ptr[0] = 0;
    for (int64_t i = 0; i < M; ++i) {
      h_row_ptr[i + 1] = h_row_ptr[i] + h_row_counts[i];
    }

    musaMemcpyAsync(d_row_ptr, h_row_ptr.data(), (M + 1) * sizeof(int64_t),
                    musaMemcpyHostToDevice, raw_stream);

    musaMemcpyAsync(d_row_ptr_copy, d_row_ptr, (M + 1) * sizeof(int64_t),
                    musaMemcpyDeviceToDevice, raw_stream);

    LaunchFillCSR(indices.flat<int64_t>().data(), values.flat<float>().data(),
                  nnz,
                  d_row_ptr_copy, 
                  d_col_idx, d_csr_values, raw_stream);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({M, N}), &output));

    LaunchSparseDenseMatMul(d_row_ptr, d_col_idx, d_csr_values,
                            dense.flat<float>().data(), M, K, N,
                            output->flat<float>().data(), raw_stream);

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