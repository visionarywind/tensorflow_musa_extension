#include <cstdlib>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"
#include <sstream>

namespace tensorflow {
namespace musa {

namespace {
void DumpMusaTensorToHost(OpKernelContext* ctx, const Tensor& device_tensor, const string& name) {
  // 1. 空张量直接打印
  if (device_tensor.NumElements() == 0) {
    LOG(ERROR) << "[" << name << "] Empty Tensor, Shape: " << device_tensor.shape().DebugString();
    return;
  }

  // 2. 仅支持FLOAT类型（Adam/Multiply核心类型，可扩展）
  OP_REQUIRES(ctx, device_tensor.dtype() == DT_FLOAT,
              errors::InvalidArgument("Dump only supports FLOAT for now"));

  // 3. 创建CPU(Host)张量，用于接收MUSA设备数据
  Tensor host_tensor;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true); // 强制分配在CPU
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, device_tensor.shape(), &host_tensor, cpu_alloc));

  // 4. 获取MUSA流，执行设备→Host拷贝
  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(
      host_tensor.data(),                // Host目标地址
      device_tensor.data(),              // MUSA设备源地址
      device_tensor.TotalBytes(),        // 总字节数
      musaMemcpyDeviceToHost,            // 拷贝方向：MUSA → CPU
      stream
  );

  // 5. 同步流，确保拷贝完成
  musaStreamSynchronize(stream);
  OP_REQUIRES(ctx, err == musaSuccess,
              errors::Internal("Dump musaMemcpy failed: ", musaGetErrorString(err)));

  // 6. 打印关键信息（形状、设备地址、Host地址、前10个数值）
  const float* host_data = host_tensor.flat<float>().data();
  LOG(ERROR) << "==================================================";
  LOG(ERROR) << "Dump Tensor: " << name << ", Shape: " << device_tensor.shape().DebugString() << ", MUSA Device Addr: " << device_tensor.data() << ", Host Addr: " << host_tensor.data();
  LOG(ERROR) << "Data: ";

  std::stringstream ss;
  for (int i = 0; i < (int)host_tensor.NumElements(); ++i) {
    ss << "\t" << host_data[i];
  }
  LOG(ERROR) << ss.str();
  LOG(ERROR) << "==================================================";
}
}

template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  // Multiply is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    LOG(ERROR) << "Multiply called " << in0.data() << in1.data();
    DumpMusaTensorToHost(ctx, in0, "left");
    DumpMusaTensorToHost(ctx, in0, "right");


    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for Mul: ", in0.shape().DebugString(),
                    " and ", in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;

      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // if (in0.shape() == output_shape) {
    //   const std::vector<int> forwardable_input_indices = {0};
    //   OP_REQUIRES_OK(
    //       ctx, ctx->forward_input_or_allocate_output(
    //                forwardable_input_indices, 0, output_shape, &output));
    // } else if (in1.shape() == output_shape) {
    //   const std::vector<int> forwardable_input_indices = {1};
    //   OP_REQUIRES_OK(
    //       ctx, ctx->forward_input_or_allocate_output(
    //                forwardable_input_indices, 0, output_shape, &output));
    // } else {
    //   OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    // }

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL);

    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    // musaStream_t stream = GetMusaStreamByCtx(ctx);
    // ctx->MaintainLifetimeOnStream(&in0, stream);
    // ctx->MaintainLifetimeOnStream(&in1, stream);
    // ctx->MaintainLifetimeOnStream(output, stream);

    auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);
    DumpMusaTensorToHost(ctx, *output, "output");

    OP_REQUIRES(ctx, status == mStatus::SUCCESS,
                errors::Internal("MUSA Multiply execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_MULTIPLY(TYPE)                        \
  REGISTER_KERNEL_BUILDER(                                  \
      Name("Mul").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaMultiplyOp<TYPE>);

REGISTER_MUSA_MULTIPLY(float);
REGISTER_MUSA_MULTIPLY(Eigen::half);
REGISTER_MUSA_MULTIPLY(bfloat16);
REGISTER_MUSA_MULTIPLY(int32);
REGISTER_MUSA_MULTIPLY(int64);

#undef REGISTER_MUSA_MULTIPLY

}  // namespace musa
}  // namespace tensorflow
