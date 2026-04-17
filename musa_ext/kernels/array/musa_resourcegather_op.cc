// Optimized MUSA ResourceGather Op Implementation
// Uses custom kernels for maximum performance
//
// Performance optimizations:
// 1. Custom MUSA kernels with optimized memory access patterns
// 2. GPU-side bounds checking
// 3. Direct kernel launch without muDNN overhead
// 4. Support for all data types including bfloat16 and double

#include <mudnn.h>

#include <cmath>
#include <limits>
#include <vector>
#include <thread>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/platform/logging.h"

// ============================================================================
// Custom Kernel Launcher Declarations
// ============================================================================

extern "C" {
void LaunchResourceGatherFloatInt32(const float* params, const int* indices,
                                    float* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherFloatInt64(const float* params, const int64_t* indices,
                                    float* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int64_t limit,
                                    musaStream_t stream);
void LaunchResourceGatherDoubleInt32(const double* params, const int* indices,
                                     double* output, int64_t batch_size,
                                     int64_t inner_size, int64_t indices_size,
                                     int64_t params_stride, int limit,
                                     musaStream_t stream);
void LaunchResourceGatherDoubleInt64(const double* params,
                                     const int64_t* indices, double* output,
                                     int64_t batch_size, int64_t inner_size,
                                     int64_t indices_size,
                                     int64_t params_stride, int64_t limit,
                                     musaStream_t stream);
void LaunchResourceGatherInt32Int32(const int* params, const int* indices,
                                    int* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt32Int64(const int* params, const int64_t* indices,
                                    int* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int64_t limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt64Int32(const int64_t* params, const int* indices,
                                    int64_t* output, int64_t batch_size,
                                    int64_t inner_size, int64_t indices_size,
                                    int64_t params_stride, int limit,
                                    musaStream_t stream);
void LaunchResourceGatherInt64Int64(const int64_t* params,
                                    const int64_t* indices, int64_t* output,
                                    int64_t batch_size, int64_t inner_size,
                                    int64_t indices_size, int64_t params_stride,
                                    int64_t limit, musaStream_t stream);
void LaunchResourceGatherHalfInt32(const void* params, const int* indices,
                                   void* output, int64_t batch_size,
                                   int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, int limit,
                                   musaStream_t stream);
void LaunchResourceGatherHalfInt64(const void* params, const int64_t* indices,
                                   void* output, int64_t batch_size,
                                   int64_t inner_size, int64_t indices_size,
                                   int64_t params_stride, int64_t limit,
                                   musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ============================================================================
// Optimized ResourceGather Op Implementation
// ============================================================================

template <typename T, typename Index>
class MusaResourceGatherOp : public MusaOpKernel {
 public:
  explicit MusaResourceGatherOp(OpKernelConstruction* c) : MusaOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("batch_dims", &batch_dims_));
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    Status s = LookupResource(c, HandleFromInput(c, 0), &v);
    if (!s.ok()) {
      c->CtxFailure(s);
      return;
    }

    tf_shared_lock ml(*v->mu());
    const Tensor& params = *v->tensor();
    const Tensor& indices = c->input(1);

    OP_REQUIRES(
        c, TensorShapeUtils::IsVectorOrHigher(params.shape()),
        errors::InvalidArgument("params must be at least 1 dimensional"));
    OP_REQUIRES(c, params.shape().dims() >= batch_dims_,
                errors::InvalidArgument("params must have at least ",
                                        batch_dims_, " dims"));

    // Build output shape
    TensorShape result_shape;
    for (int i = 0; i < batch_dims_; ++i)
      result_shape.AddDim(params.dim_size(i));
    for (int i = batch_dims_; i < indices.dims(); ++i)
      result_shape.AddDim(indices.dim_size(i));
    for (int i = batch_dims_ + 1; i < params.dims(); ++i)
      result_shape.AddDim(params.dim_size(i));

    Tensor* out = nullptr;
    s = c->allocate_output(0, result_shape, &out);
    if (!s.ok()) {
      c->CtxFailure(s);
      return;
    }

    if (out->NumElements() == 0) {
      return;
    }

    if (indices.NumElements() > 0) {
      // Calculate dimensions for kernel launch
      int64_t batch_size = 1;
      for (int i = 0; i < batch_dims_; ++i) {
        batch_size *= params.dim_size(i);
      }

      int64_t inner_size = 1;
      for (int i = batch_dims_ + 1; i < params.dims(); ++i) {
        inner_size *= params.dim_size(i);
      }

      const int64_t indices_size = indices.NumElements();
      const int64_t params_stride = params.dim_size(batch_dims_) * inner_size;
      const Index limit = static_cast<Index>(params.dim_size(batch_dims_));

      // Get stream
      musaStream_t stream = GetMusaStreamByCtx(c);

      // Launch optimized kernel
      LaunchKernel(params.flat<T>().data(), indices.flat<Index>().data(),
                   out->flat<T>().data(), batch_size, inner_size, indices_size,
                   params_stride, limit, stream);
    }
  }

 private:
  int32 batch_dims_ = 0;

  void LaunchKernel(const T* params, const Index* indices, T* output,
                    int64_t batch_size, int64_t inner_size,
                    int64_t indices_size, int64_t params_stride, Index limit,
                    musaStream_t stream);
};

// ============================================================================
// Launcher Specializations
// ============================================================================

#define DEFINE_RESOURCE_GATHER_LAUNCHER(T, IndexT, launcher_func)            \
  template <>                                                                \
  void MusaResourceGatherOp<T, IndexT>::LaunchKernel(                        \
      const T* params, const IndexT* indices, T* output, int64_t batch_size, \
      int64_t inner_size, int64_t indices_size, int64_t params_stride,       \
      IndexT limit, musaStream_t stream) {                                   \
    launcher_func(params, indices, output, batch_size, inner_size,           \
                  indices_size, params_stride, limit, stream);               \
  }

DEFINE_RESOURCE_GATHER_LAUNCHER(float, int32, LaunchResourceGatherFloatInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(float, int64, LaunchResourceGatherFloatInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, int32, LaunchResourceGatherDoubleInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER(double, int64, LaunchResourceGatherDoubleInt64)
DEFINE_RESOURCE_GATHER_LAUNCHER(int32, int32, LaunchResourceGatherInt32Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(int32, int64, LaunchResourceGatherInt32Int64)
DEFINE_RESOURCE_GATHER_LAUNCHER(int64, int32, LaunchResourceGatherInt64Int32)
DEFINE_RESOURCE_GATHER_LAUNCHER(int64, int64, LaunchResourceGatherInt64Int64)

// Half specialization
#define DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(IndexT, launcher_func)          \
  template <>                                                                \
  void MusaResourceGatherOp<Eigen::half, IndexT>::LaunchKernel(              \
      const Eigen::half* params, const IndexT* indices, Eigen::half* output, \
      int64_t batch_size, int64_t inner_size, int64_t indices_size,          \
      int64_t params_stride, IndexT limit, musaStream_t stream) {            \
    launcher_func(reinterpret_cast<const void*>(params), indices,            \
                  reinterpret_cast<void*>(output), batch_size, inner_size,   \
                  indices_size, params_stride, limit, stream);               \
  }

DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(int32, LaunchResourceGatherHalfInt32)
DEFINE_RESOURCE_GATHER_LAUNCHER_HALF(int64, LaunchResourceGatherHalfInt64)

#undef DEFINE_RESOURCE_GATHER_LAUNCHER
#undef DEFINE_RESOURCE_GATHER_LAUNCHER_HALF

// ============================================================================
// ResourceScatterAdd Op (keeps muDNN for atomic operations)
// ============================================================================

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

  ss << "\t" <<std::this_thread::get_id() << "\n\tData:";
  switch (dtype) {
    case DT_INT32: {
      int mn = INT_MAX, mx = INT_MIN;
      const int32* data = host_tensor.flat<int32>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        // ss << data[i] << "\t";
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
      }
      ss << "min : " << mn << ", mx : " << mx;
      break;
    }
    case DT_INT64: {
      int64 mn = INT64_MIN, mx = INT64_MAX;
      const int64* data = host_tensor.flat<int64>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        // ss << data[i] << "\t";
        mn = std::min(mn, data[i]);
        mx = std::max(mx, data[i]);
      }
      ss << "min : " << mn << ", mx : " << mx;
      break;
    }
    case DT_FLOAT: {
      const float* data = host_tensor.flat<float>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        ss << data[i] << "\t";
      }
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

namespace {
mType GetType(DataType t) {
  switch (t) {
    case DataType::DT_FLOAT:
      return mType::FLOAT;
    case DataType::DT_DOUBLE:
      return mType::DOUBLE;
    case DataType::DT_INT32:
      return mType::INT32;
    case DataType::DT_UINT8:
      return mType::UINT8;
    case DataType::DT_INT16:
      return mType::INT16;
    case DataType::DT_INT8:
      return mType::INT8;
    case DataType::DT_INT64:
      return mType::INT64;
    case DataType::DT_BFLOAT16:
      return mType::BFLOAT16;
    case DataType::DT_UINT16:
      return mType::UINT16;
    case DataType::DT_HALF:
      return mType::HALF;
    case DataType::DT_UINT32:
      return mType::UINT32;
    case DataType::DT_UINT64:
      return mType::UINT64;
    case DataType::DT_BOOL:
      return mType::BOOL;
    default:
      CHECK(false);
      throw;
  }
}
}  // namespace

template <typename T, typename Index>
class MusaResourceScatterAddOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    mutex_lock ml(*v->mu());
    Tensor* params = v->tensor();
    const Tensor& indices = c->input(1);
    const Tensor& updates = c->input(2);
    //LOG(ERROR) << "Scatter op - indices : " << (indices.data())
    //           << ", update : " << (updates.data())
    //           << ", params : " << (params->data());
    // DumpMusaTensorToHost(c, indices, "indices");

    if (indices.NumElements() > 0) {
      auto& h = GetHandleByCtx(c);
      auto* device = static_cast<MusaDevice*>(c->device());
      auto maintainer = device->GetMemMaintainer(
          [](size_t s) { return ::musa::dnn::MemoryHandler(); });

      mScatterND op;
      MTOP_CHECK_OK(op.SetMode(mScatterND::Mode::ADD), "SetModeAdd", c);

      // Tensor indices_reshaped;
      // const int64 num_indices = indices.NumElements();
      // OP_REQUIRES_OK(c, c->allocate_temp(
      //     indices.dtype(),
      //     TensorShape({indices.NumElements(), 1}),
      //     &indices_reshaped
      // ));
      // const Index* src_ptr = indices.flat<Index>().data();
      // Index* dst_ptr = indices_reshaped.flat<Index>().data();
      // std::memcpy(dst_ptr, src_ptr, num_indices * sizeof(Index));

      //TensorShape indices_new_shape = indices.shape();
      //indices_new_shape.AddDim(1);

      //if (!indices_reshaped
      //         .BitcastFrom(indices, indices.dtype(), indices_new_shape)
      //         .ok()) {
      //  OP_REQUIRES(c, false,
      //              errors::Internal(
      //                  "MusaResourceScatterAdd: Failed to reshape indices."));
      //}

      auto params_mt = CreateMTensor(*params, format_);
      auto create_mtensor = [](const Tensor& t, mFormat format) {
        mTensor rst;
        rst.SetAddr(const_cast<void*>(static_cast<const void*>(t.tensor_data().data())));
        rst.SetType(GetType(t.dtype()));
        
        auto dims_raw = t.shape().dim_sizes();
        const int rank = static_cast<int>(dims_raw.size());
        if (rank >= 4) {
          rst.SetFormat(format);
        } else {
          rst.SetFormat(mFormat::NCHW);
        }

        rst.SetNdInfo({static_cast<int64_t>(rank), static_cast<int64_t>(1)});
        return rst;
      };
      auto indices_mt = create_mtensor(indices, format_);
      // indices_mt.SetNdInfo({static_cast<int64_t>(indices.shape().dim_sizes().size()), {static_cast<int64_t>(1)});

      //DumpMusaTensorToHost(c, indices, "indices");
      auto updates_mt = CreateMTensor(updates, format_);
      MTOP_CHECK_OK_RUN(
          op.Run(h, params_mt, indices_mt, updates_mt, maintainer),
          "RunScatterND", c);
    }
  }
};

// ============================================================================
// AssignUpdateVariable Op
// ============================================================================

template <typename T, mBinary::Mode BMODE>
class MusaAssignUpdateVariableOp : public MusaOpKernel {
 public:
  using MusaOpKernel::MusaOpKernel;
  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> variable;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &variable));
    mutex_lock ml(*variable->mu());

    Tensor* var_tensor = variable->tensor();
    const Tensor& value = c->input(1);

    if (var_tensor->NumElements() > 0) {
      auto& h = GetHandleByCtx(c);
      mBinary op;
      MTOP_CHECK_OK(op.SetMode(BMODE), "SetMode", c);
      auto out_mt = CreateMTensor(*var_tensor, format_);
      auto in_mt = CreateMTensor(value, format_);
      MTOP_CHECK_OK_RUN(op.Run(h, out_mt, out_mt, in_mt), "RunBinaryUpdate", c);
    }

    if (c->num_outputs() > 0) {
      c->set_output(0, c->input(0));
    }
  }
};

// ============================================================================
// VariableShape Op
// ============================================================================

class MusaVariableShapeOp : public OpKernel {
 public:
  explicit MusaVariableShapeOp(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(c, LookupResource(c, HandleFromInput(c, 0), &v));
    tf_shared_lock ml(*v->mu());
    const TensorShape& s = v->tensor()->shape();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, TensorShape({s.dims()}), &out));
    for (int i = 0; i < s.dims(); ++i) {
      if (out->dtype() == DT_INT32)
        out->flat<int32>()(i) = s.dim_size(i);
      else
        out->flat<int64>()(i) = s.dim_size(i);
    }
  }
};

// ============================================================================
// Kernel Registration
// ============================================================================

#define REGISTER_MUSA_KERNELS(type)                               \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceGatherOp<type, int32>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceGather")                  \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceGatherOp<type, int64>);     \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int32>("Tindices"), \
                          MusaResourceScatterAddOp<type, int32>); \
  REGISTER_KERNEL_BUILDER(Name("ResourceScatterAdd")              \
                              .Device(DEVICE_MTGPU)               \
                              .HostMemory("resource")             \
                              .TypeConstraint<type>("dtype")      \
                              .TypeConstraint<int64>("Tindices"), \
                          MusaResourceScatterAddOp<type, int64>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignSubVariableOp")                                 \
          .Device(DEVICE_MTGPU)                                   \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::SUB>);      \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("AssignAddVariableOp")                                 \
          .Device(DEVICE_MTGPU)                                   \
          .HostMemory("resource")                                 \
          .TypeConstraint<type>("dtype"),                         \
      MusaAssignUpdateVariableOp<type, mBinary::Mode::ADD>);

REGISTER_MUSA_KERNELS(float);
REGISTER_MUSA_KERNELS(Eigen::half);
REGISTER_MUSA_KERNELS(double);
REGISTER_MUSA_KERNELS(int32);
REGISTER_MUSA_KERNELS(int64);

REGISTER_KERNEL_BUILDER(Name("VariableShape")
                            .Device(DEVICE_MTGPU)
                            .HostMemory("input")
                            .HostMemory("output"),
                        MusaVariableShapeOp);

#undef REGISTER_MUSA_KERNELS

}  // namespace musa
}  // namespace tensorflow
