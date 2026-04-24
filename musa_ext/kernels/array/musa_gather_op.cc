#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "../utils_op.h"
#include "../../mu/kernel_register.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

// ============================================================================
// Custom Kernel Launcher Declarations
// ============================================================================

extern "C" {
void LaunchGatherV2FloatInt32(const float* params, const int* indices, float* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2FloatInt64(const float* params, const int64_t* indices, float* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int64_t limit,
                              musaStream_t stream);
void LaunchGatherV2DoubleInt32(const double* params, const int* indices, double* output,
                               int64_t batch_size, int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride, int limit,
                               musaStream_t stream);
void LaunchGatherV2DoubleInt64(const double* params, const int64_t* indices, double* output,
                               int64_t batch_size, int64_t axis_size, int64_t inner_size,
                               int64_t indices_size, int64_t params_stride, int64_t limit,
                               musaStream_t stream);
void LaunchGatherV2Int32Int32(const int* params, const int* indices, int* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2Int32Int64(const int* params, const int64_t* indices, int* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int64_t limit,
                              musaStream_t stream);
void LaunchGatherV2Int64Int32(const int64_t* params, const int* indices, int64_t* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int limit,
                              musaStream_t stream);
void LaunchGatherV2Int64Int64(const int64_t* params, const int64_t* indices, int64_t* output,
                              int64_t batch_size, int64_t axis_size, int64_t inner_size,
                              int64_t indices_size, int64_t params_stride, int64_t limit,
                              musaStream_t stream);
void LaunchGatherV2BoolInt32(const bool* params, const int* indices, bool* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int limit,
                             musaStream_t stream);
void LaunchGatherV2BoolInt64(const bool* params, const int64_t* indices, bool* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int64_t limit,
                             musaStream_t stream);
void LaunchGatherV2HalfInt32(const void* params, const int* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int limit,
                             musaStream_t stream);
void LaunchGatherV2HalfInt64(const void* params, const int64_t* indices, void* output,
                             int64_t batch_size, int64_t axis_size, int64_t inner_size,
                             int64_t indices_size, int64_t params_stride, int64_t limit,
                             musaStream_t stream);
void LaunchGatherV2BFloat16Int32(const void* params, const int* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, int limit,
                                 musaStream_t stream);
void LaunchGatherV2BFloat16Int64(const void* params, const int64_t* indices, void* output,
                                 int64_t batch_size, int64_t axis_size, int64_t inner_size,
                                 int64_t indices_size, int64_t params_stride, int64_t limit,
                                 musaStream_t stream);
}

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

// ============================================================================
// Optimized Gather Op Implementation
// ============================================================================

template <typename T, typename IndexT>
class MusaGatherOp : public MusaOpKernel {
 public:
  explicit MusaGatherOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {
    axis_ = 0;
    has_axis_input_ = false;
  }

  // Gather is computationally intensive due to irregular memory access
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    LOG(ERROR) << "[MUSA Debug] Thread: " << std::this_thread::get_id() 
              << " | Op: " << __FILE__ 
              << " | Method: " << __FUNCTION__;
  }


    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    DumpMusaTensorToHost(ctx, params, "params");
    DumpMusaTensorToHost(ctx, indices, "indices");


    int64_t axis = axis_;
    if (ctx->num_inputs() >= 3) {
      const Tensor& axis_tensor = ctx->input(2);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(axis_tensor.shape()),
                  errors::InvalidArgument("axis must be a scalar"));

      if (axis_tensor.dtype() == DT_INT32) {
        axis = static_cast<int64_t>(axis_tensor.scalar<int32>()());
      } else if (axis_tensor.dtype() == DT_INT64) {
        axis = axis_tensor.scalar<int64>()();
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("axis must be int32 or int64"));
      }
      has_axis_input_ = true;
    }

    const int64_t params_dims = params.dims();
    if (axis < 0) {
      axis += params_dims;
    }

    OP_REQUIRES(
        ctx, axis >= 0 && axis < params_dims,
        errors::InvalidArgument("Expected axis in the range [", -params_dims,
                                ", ", params_dims, "), but got ", axis));

    OP_REQUIRES(ctx, indices.dtype() == DT_INT32 || indices.dtype() == DT_INT64,
                errors::InvalidArgument("indices must be int32 or int64"));

    // Build output shape
    TensorShape output_shape;
    for (int64_t i = 0; i < axis; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }
    for (int64_t i = 0; i < indices.dims(); ++i) {
      output_shape.AddDim(indices.dim_size(i));
    }
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      output_shape.AddDim(params.dim_size(i));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    if (output->NumElements() == 0) return;

    // Compute dimensions for kernel launch
    const int64_t limit = params.dim_size(axis);

    int64_t batch_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      batch_size *= params.dim_size(i);
    }

    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < params_dims; ++i) {
      inner_size *= params.dim_size(i);
    }

    const int64_t indices_size = indices.NumElements();
    const int64_t params_stride = limit * inner_size;

    // Get stream for async execution
    musaStream_t stream = GetMusaStreamByCtx(ctx);

    // Launch optimized custom kernel
    LaunchKernel(
        params.flat<T>().data(),
        indices.flat<IndexT>().data(),
        output->flat<T>().data(),
        batch_size,
        limit,
        inner_size,
        indices_size,
        params_stride,
        static_cast<IndexT>(limit),
        stream);
    
    DumpMusaTensorToHost(ctx, indices, "indices");
  }

 private:
  int64_t axis_;
  bool has_axis_input_;

  // Type-specific kernel launcher
  void LaunchKernel(const T* params, const IndexT* indices, T* output,
                    int64_t batch_size, int64_t axis_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, IndexT limit,
                    musaStream_t stream);
};

// ============================================================================
// Launcher Specializations
// ============================================================================

#define DEFINE_GATHER_LAUNCHER(T, IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<T, IndexT>::LaunchKernel( \
      const T* params, const IndexT* indices, T* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(params, indices, output, batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER(float, int32, LaunchGatherV2FloatInt32)
DEFINE_GATHER_LAUNCHER(float, int64, LaunchGatherV2FloatInt64)
DEFINE_GATHER_LAUNCHER(double, int32, LaunchGatherV2DoubleInt32)
DEFINE_GATHER_LAUNCHER(double, int64, LaunchGatherV2DoubleInt64)
DEFINE_GATHER_LAUNCHER(int32, int32, LaunchGatherV2Int32Int32)
DEFINE_GATHER_LAUNCHER(int32, int64, LaunchGatherV2Int32Int64)
DEFINE_GATHER_LAUNCHER(int64, int32, LaunchGatherV2Int64Int32)
DEFINE_GATHER_LAUNCHER(int64, int64, LaunchGatherV2Int64Int64)
DEFINE_GATHER_LAUNCHER(bool, int32, LaunchGatherV2BoolInt32)
DEFINE_GATHER_LAUNCHER(bool, int64, LaunchGatherV2BoolInt64)

// Half specialization
#define DEFINE_GATHER_LAUNCHER_HALF(IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<Eigen::half, IndexT>::LaunchKernel( \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER_HALF(int32, LaunchGatherV2HalfInt32)
DEFINE_GATHER_LAUNCHER_HALF(int64, LaunchGatherV2HalfInt64)

// BFloat16 specialization
#define DEFINE_GATHER_LAUNCHER_BF16(IndexT, launcher_func) \
  template <> \
  void MusaGatherOp<bfloat16, IndexT>::LaunchKernel( \
      const bfloat16* params, const IndexT* indices, bfloat16* output, \
      int64_t batch_size, int64_t axis_size, int64_t inner_size, \
      int64_t indices_size, int64_t params_stride, IndexT limit, \
      musaStream_t stream) { \
    launcher_func(reinterpret_cast<const void*>(params), indices, \
                  reinterpret_cast<void*>(output), batch_size, axis_size, inner_size, \
                  indices_size, params_stride, limit, stream); \
  }

DEFINE_GATHER_LAUNCHER_BF16(int32, LaunchGatherV2BFloat16Int32)
DEFINE_GATHER_LAUNCHER_BF16(int64, LaunchGatherV2BFloat16Int64)

#undef DEFINE_GATHER_LAUNCHER
#undef DEFINE_GATHER_LAUNCHER_HALF
#undef DEFINE_GATHER_LAUNCHER_BF16

// ============================================================================
// Kernel Registration
// ============================================================================

#define REGISTER_GATHER_V2_FULL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int32>("Tindices") \
                              .TypeConstraint<int64>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int32>);               \
  REGISTER_KERNEL_BUILDER(Name("GatherV2")                       \
                              .Device("MUSA")                    \
                              .TypeConstraint<T>("Tparams")      \
                              .TypeConstraint<int64>("Tindices") \
                              .TypeConstraint<int32>("Taxis")    \
                              .HostMemory("axis"),               \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V2_FULL(float);
REGISTER_GATHER_V2_FULL(double);
REGISTER_GATHER_V2_FULL(int32);
REGISTER_GATHER_V2_FULL(int64);
REGISTER_GATHER_V2_FULL(bool);
REGISTER_GATHER_V2_FULL(Eigen::half);
REGISTER_GATHER_V2_FULL(bfloat16);

#undef REGISTER_GATHER_V2_FULL

#define REGISTER_GATHER_V1(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaGatherOp<T, int32>);                \
  REGISTER_KERNEL_BUILDER(Name("Gather")                          \
                              .Device("MUSA")                     \
                              .TypeConstraint<T>("Tparams")       \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaGatherOp<T, int64>);

REGISTER_GATHER_V1(float);
REGISTER_GATHER_V1(double);
REGISTER_GATHER_V1(int32);
REGISTER_GATHER_V1(int64);
REGISTER_GATHER_V1(bool);
REGISTER_GATHER_V1(Eigen::half);
REGISTER_GATHER_V1(bfloat16);

#undef REGISTER_GATHER_V1

}  // namespace musa
}  // namespace tensorflow
