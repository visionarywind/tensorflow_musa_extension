#include <new>

#include "../utils_op.h"
#include "mu/device/musa_memcpy.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

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
class MusaReshapeOp : public MusaOpKernel {
 public:
  explicit MusaReshapeOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    LOG(ERROR) << "[MUSA Debug] Thread: " << std::this_thread::get_id() 
              << " | Op: " << __FILE__ 
              << " | Method: " << __FUNCTION__;
  }

    const Tensor& input = ctx->input(0);
    const Tensor& sizes = ctx->input(1);

    DumpMusaTensorToHost(ctx, input, "input");

    TensorShape shape;
    int64 unknown_index = -1;
    int64 product = 1;

    if (sizes.dtype() == DT_INT32) {
      auto vec = sizes.flat<int32>();
      for (int i = 0; i < vec.size(); ++i) {
        int64 size = static_cast<int64>(vec(i));
        if (size == -1) {
          OP_REQUIRES(ctx, unknown_index == -1,
                      errors::InvalidArgument(
                          "Only one input size may be -1, not both ",
                          unknown_index, " and ", i));
          unknown_index = i;
          shape.AddDim(1);
        } else {
          OP_REQUIRES(ctx, size >= 0,
                      errors::InvalidArgument(
                          "Dimension size must be non-negative, got ", size));
          shape.AddDim(size);
          product *= size;
        }
      }
    } else if (sizes.dtype() == DT_INT64) {
      auto vec = sizes.flat<int64>();
      for (int i = 0; i < vec.size(); ++i) {
        int64 size = vec(i);
        if (size == -1) {
          OP_REQUIRES(ctx, unknown_index == -1,
                      errors::InvalidArgument(
                          "Only one input size may be -1, not both ",
                          unknown_index, " and ", i));
          unknown_index = i;
          shape.AddDim(1);
        } else {
          OP_REQUIRES(ctx, size >= 0,
                      errors::InvalidArgument(
                          "Dimension size must be non-negative, got ", size));
          shape.AddDim(size);
          product *= size;
        }
      }
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::InvalidArgument("Shape tensor must be int32 or int64"));
    }

    if (unknown_index != -1) {
      int64 input_num_elements = input.NumElements();
      if (input_num_elements > 0) {
        OP_REQUIRES(ctx, product > 0,
                    errors::InvalidArgument(
                        "Cannot infer -1 dimension with zero product"));
        OP_REQUIRES(ctx, input_num_elements % product == 0,
                    errors::InvalidArgument(
                        "Input has ", input_num_elements,
                        " elements, which isn't divisible by ", product));
        shape.set_dim(unknown_index, input_num_elements / product);
      } else {
        shape.set_dim(unknown_index, 0);
      }
    }

    OP_REQUIRES(ctx, input.NumElements() == shape.num_elements(),
                errors::InvalidArgument("Input has ", input.NumElements(),
                                        " elements, but target shape has ",
                                        shape.num_elements(), " elements."));

    Tensor output;
    bool success = output.CopyFrom(input, shape);
    OP_REQUIRES(ctx, success,
                errors::Internal("MUSA Reshape: Tensor::CopyFrom failed."));

    ctx->set_output(0, output);
  }
};

#define REGISTER_MUSA_RESHAPE(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Reshape").Device("MUSA").TypeConstraint<TYPE>("T").HostMemory( \
          "shape"),                                                        \
      MusaReshapeOp<TYPE>)

REGISTER_MUSA_RESHAPE(float);
REGISTER_MUSA_RESHAPE(Eigen::half);
REGISTER_MUSA_RESHAPE(bfloat16);
REGISTER_MUSA_RESHAPE(double);
REGISTER_MUSA_RESHAPE(int32);
REGISTER_MUSA_RESHAPE(int64);
REGISTER_MUSA_RESHAPE(bool);

#undef REGISTER_MUSA_RESHAPE

}  // namespace musa
}  // namespace tensorflow