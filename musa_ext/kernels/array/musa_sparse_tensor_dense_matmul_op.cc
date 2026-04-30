#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

using namespace tensorflow;

// ----------------------------------------------------------------------------
// 核函数启动器声明
// ----------------------------------------------------------------------------
extern "C" {
void LaunchComputeRowCounts(const int64_t* indices, int64_t nnz, int64_t M,
                            int32_t* row_counts, musaStream_t stream);
void LaunchMaskToCount(const bool* mask, int32_t* count, int64_t N,
                       musaStream_t stream);
void LaunchBlockScan(const int32_t* count, int64_t* pos, int64_t* block_sums,
                     int64_t N, int64_t block_size, musaStream_t stream);
void LaunchAddBlockPrefixSum(int64_t* pos, const int64_t* block_prefix_sums,
                             int64_t N, int64_t block_size,
                             musaStream_t stream);
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

// ----------------------------------------------------------------------------
// MusaSparseTensorDenseMatMulOp 主类
// ----------------------------------------------------------------------------
template <typename T>
class MusaSparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit MusaSparseTensorDenseMatMulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint_b", &adjoint_b_));
  }

  void Compute(OpKernelContext* context) override {
    // 1. 获取输入张量
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& dense_shape = context->input(2);
    const Tensor& dense = context->input(3);

    // 2. 解析输入维度
    const int64_t nnz = indices.dim_size(0);
    const int64_t ndims = indices.dim_size(1);
    const int64_t M = dense_shape.flat<int64_t>()(0);
    const int64_t K = dense_shape.flat<int64_t>()(1);
    const int64_t dense_rows = dense.dim_size(0);
    const int64_t N = dense.dim_size(1);

    // 3. 基础校验
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

    // 空输入直接返回空输出
    if (nnz == 0) {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({M, N}), &output));
      output->flat<T>().setZero();
      return;
    }

    // 4. 获取MUSA流
    musaStream_t raw_stream = GetMusaStreamByCtx(context);

    // 5. 使用 allocate_temp 分配设备端临时内存
    // 注意：allocate_temp 分配的内存生命周期仅在当前 Compute
    // 调用期间有效，无需手动 free

    Tensor d_row_counts_tensor;
    Tensor d_row_ptr_tensor;
    Tensor d_col_idx_tensor;
    Tensor d_csr_values_tensor;
    Tensor d_row_ptr_copy_tensor;

    // 分配 row_counts: [M] int32
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({M}),
                                                   &d_row_counts_tensor));

    // 分配 row_ptr: [M+1] int64
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_tensor));

    // 分配 col_idx: [nnz] int64
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({nnz}),
                                                   &d_col_idx_tensor));

    // 分配 csr_values: [nnz] float
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({nnz}),
                                                   &d_csr_values_tensor));

    // 分配 row_ptr_copy: [M+1] int64 (用于 FillCSR 内部修改)
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({M + 1}),
                                          &d_row_ptr_copy_tensor));

    // 获取设备指针
    int32_t* d_row_counts = d_row_counts_tensor.flat<int32_t>().data();
    int64_t* d_row_ptr = d_row_ptr_tensor.flat<int64_t>().data();
    int64_t* d_col_idx = d_col_idx_tensor.flat<int64_t>().data();
    float* d_csr_values = d_csr_values_tensor.flat<float>().data();
    int64_t* d_row_ptr_copy = d_row_ptr_copy_tensor.flat<int64_t>().data();

    // 初始化 row_counts 为 0
    // 使用 musaMemsetAsync 或者简单的 kernel 清零，这里保持原逻辑使用 memset
    musaMemsetAsync(d_row_counts, 0, M * sizeof(int32_t), raw_stream);

    // 6. COO -> CSR 转换流程

    // Step 6.1: 统计每行非零元素个数
    LaunchComputeRowCounts(indices.flat<int64_t>().data(), nnz, M, d_row_counts,
                           raw_stream);

    // Step 6.2: 对 row_counts 做 exclusive scan 得到 row_ptr
    // 由于原逻辑是在 Host 端做 Prefix Sum，我们需要将数据拷回 Host
    // 注意：同步操作会影响性能，但如果必须要在 Host
    // 做扫描，这是不可避免的开销。 更优解是实现 Device 端的 Prefix Sum
    // Kernel，但此处仅针对内存分配优化。

    std::vector<int32_t> h_row_counts(M);
    std::vector<int64_t> h_row_ptr(M + 1);

    // 等待 ComputeRowCounts 完成
    musaStreamSynchronize(raw_stream);

    musaMemcpy(h_row_counts.data(), d_row_counts, M * sizeof(int32_t),
               musaMemcpyDeviceToHost);

    h_row_ptr[0] = 0;
    for (int64_t i = 0; i < M; ++i) {
      h_row_ptr[i + 1] = h_row_ptr[i] + h_row_counts[i];
    }

    // 拷回设备到 allocate_temp 分配的 buffer
    musaMemcpyAsync(d_row_ptr, h_row_ptr.data(), (M + 1) * sizeof(int64_t),
                    musaMemcpyHostToDevice, raw_stream);

    // 拷贝到 copy buffer
    musaMemcpyAsync(d_row_ptr_copy, d_row_ptr, (M + 1) * sizeof(int64_t),
                    musaMemcpyDeviceToDevice, raw_stream);

    // Step 6.3: 填充 CSR 的 col_idx 和 values
    LaunchFillCSR(indices.flat<int64_t>().data(), values.flat<float>().data(),
                  nnz,
                  d_row_ptr_copy,  // 使用 copy，因为 FillCSR
                                   // 可能会修改它（如果是原子加逻辑）
                  d_col_idx, d_csr_values, raw_stream);

    // 7. 分配输出张量
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({M, N}), &output));

    // 8. 调用核心矩阵乘法核函数
    LaunchSparseDenseMatMul(d_row_ptr, d_col_idx, d_csr_values,
                            dense.flat<float>().data(), M, K, N,
                            output->flat<float>().data(), raw_stream);

    // 9. 释放临时内存
    // 使用 allocate_temp 后，不需要手动 musaFree。
    // Tensor 对象在函数退出时自动销毁，底层内存由 TensorFlow  allocator
    // 管理回收。

    // 10. 检查错误
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

// ----------------------------------------------------------------------------
// 算子注册（简化版，注册 float 类型）
// ----------------------------------------------------------------------------
#define REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL(T)                   \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseTensorDenseMatMul").Device("MUSA").TypeConstraint<T>("T"), \
      MusaSparseTensorDenseMatMulOp<T>);

REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL(float);

#undef REGISTER_MUSA_SPARSE_TENSOR_DENSE_MATMUL_KERNEL

}  // namespace musa
}  // namespace tensorflow