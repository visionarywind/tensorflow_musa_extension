#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {
namespace{
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
void MusaNegKernelLauncher(const void* in, void* out, int size,
                           musaStream_t stream);

template <typename T>
class MusaNegOp : public MusaOpKernel {
 public:
  explicit MusaNegOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  // Neg is element-wise - lightweight
  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    LOG(ERROR) << "[MUSA Debug] Thread: " << std::this_thread::get_id() 
              << " | Op: " << __FILE__ 
              << " | Method: " << __FUNCTION__;
  }

    const Tensor& input = ctx->input(0);
    DumpMusaTensorToHost(ctx, input, "neg input");

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    MusaNegKernelLauncher<T>(input.tensor_data().data(),
                             const_cast<char*>(output->tensor_data().data()),
                             input.NumElements(), stream);
  }
};

#define REGISTER_MUSA_NEG(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Neg").Device("MUSA").TypeConstraint<TYPE>("T"), MusaNegOp<TYPE>)

REGISTER_MUSA_NEG(float);
REGISTER_MUSA_NEG(double);
REGISTER_MUSA_NEG(int32);
REGISTER_MUSA_NEG(int64);
REGISTER_MUSA_NEG(Eigen::half);
REGISTER_MUSA_NEG(bfloat16);

#undef REGISTER_MUSA_NEG

}  // namespace musa
}  // namespace tensorflow
