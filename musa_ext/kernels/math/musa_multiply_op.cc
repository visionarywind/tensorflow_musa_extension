#include <cstdlib>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>

namespace tensorflow {
namespace musa {

namespace {
void DumpMusaTensorToHost(OpKernelContext* ctx, const Tensor& device_tensor,
                          const string& name) {
  if (device_tensor.NumElements() == 0) {
    LOG(ERROR) << std::this_thread::get_id() << "[Dump] " << name
               << " | Empty Tensor | Shape: "
               << device_tensor.shape().DebugString();
    return;
  }

  std::stringstream ss;
  const DataType dtype = device_tensor.dtype();
  ss << std::this_thread::get_id()
     << "=================================================="
     << "[Dump] " << name << " | Type: " << DataTypeString(dtype)
     << " | Shape: " << device_tensor.shape().DebugString()
     << " | Device Addr: " << device_tensor.data();

  Tensor host_tensor;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, device_tensor.shape(),
                                         &host_tensor, cpu_alloc));

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(host_tensor.data(), device_tensor.data(),
                                    device_tensor.TotalBytes(),
                                    musaMemcpyDeviceToHost, stream);
  musaStreamSynchronize(stream);
  OP_REQUIRES(
      ctx, err == musaSuccess,
      errors::Internal("Dump musaMemcpy failed: ", musaGetErrorString(err)));

  const int64_t num_elems = host_tensor.NumElements();

  ss << std::this_thread::get_id() << "\n\tData:";
  switch (dtype) {
    case DT_INT32: {
      int32 mn = INT_MAX, mx = INT_MIN;
      const int32* data = host_tensor.flat<int32>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
        ss << data[i] << "\t";
      }
      ss << "min - " << mn << ", max - " << mx << "\t";
      break;
    }
    case DT_INT64: {
      int64_t mn = INT64_MAX, mx = INT64_MIN;
      const int64* data = host_tensor.flat<int64>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
        ss << data[i] << "\t";
      }
      ss << "min - " << mn << ", max - " << mx << "\t";
      break;
    }
    case DT_FLOAT: {
      float mn = FLT_MAX;     // 最大正数
      float mx = FLT_MIN;
      const float* data = host_tensor.flat<float>().data();
      for (int64_t i = 0; i < std::max((int64_t)100, num_elems); ++i) {
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
      }
      ss << "min - " << mn << ", max - " << mx << "\t";
      break;
    }
    default: {
      LOG(ERROR) << "Unsupported dtype: " << DataTypeString(dtype);
      return;
    }
  }

  LOG(ERROR) << ss.str();
}
}  // namespace 

template <typename T>
class MusaMultiplyOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;

  // Multiply is element-wise and computationally lightweight
  // Mark as inexpensive to enable inline scheduling
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    LOG(ERROR) << "[MUSA Debug] Thread: " << std::this_thread::get_id() 
              << " | Op: " << __FILE__ 
              << " | Method: " << __FUNCTION__;
  }

    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    BCast bcast(BCast::Vec(in0.shape().dim_sizes()),
                BCast::Vec(in1.shape().dim_sizes()));

    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument(
                    "Incompatible shapes for Mul: ", in0.shape().DebugString(),
                    " and ", in1.shape().DebugString()));

    TensorShape output_shape = BCast::ToShape(bcast.output_shape());

    Tensor* output = nullptr;
    if (in0.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {0};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else if (in1.shape() == output_shape) {
      const std::vector<int> forwardable_input_indices = {1};
      OP_REQUIRES_OK(
          ctx, ctx->forward_input_or_allocate_output(
                   forwardable_input_indices, 0, output_shape, &output));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    }

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mBinary binary_op;
    binary_op.SetMode(::musa::dnn::Binary::Mode::MUL);

    mTensor mt_in0 = CreateMTensor(in0, format_);
    mTensor mt_in1 = CreateMTensor(in1, format_);
    mTensor mt_out = CreateMTensor(*output, format_);

    auto status = binary_op.Run(handle, mt_out, mt_in0, mt_in1);
    DumpMusaTensorToHost(ctx, in0, "in0");
    DumpMusaTensorToHost(ctx, in1, "in1");
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
