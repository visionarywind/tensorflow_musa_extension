#include <list>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_device.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_memset.h"

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
      const int32* data = host_tensor.flat<int32>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        ss << data[i] << "\t";
      }
      break;
    }
    case DT_INT64: {
      const int64* data = host_tensor.flat<int64>().data();
      for (int64_t i = 0; i < num_elems; ++i) {
        ss << data[i] << "\t";
      }
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

template <typename T, typename OutIdxT>
class MusaUniqueOp : public MusaOpKernel {
 public:
  explicit MusaUniqueOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    OP_REQUIRES(ctx, input.dims() <= 1,
                errors::InvalidArgument("Unique only supports 1D tensor, got ", input.dims(), "D"));

    const int64_t input_num = input.NumElements();
    if (input_num == 0) {
      ctx->set_output(0, Tensor(input.dtype(), TensorShape()));
      ctx->set_output(1, Tensor(DataTypeToEnum<OutIdxT>::value, TensorShape()));
      return;
    }

    Tensor temp_out_values;
    Tensor *temp_out_indices;
    Tensor temp_counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &temp_out_values));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &temp_out_indices));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<OutIdxT>::value, input.shape(), &temp_counts));
    std::vector<Tensor> workspace_tensors;
    auto mem_alloc_func = [ctx, &workspace_tensors](size_t size) -> ::musa::dnn::MemoryHandler {
      if (size == 0) return nullptr;
      Tensor temp;
      Status s = ctx->allocate_temp(DT_UINT8, TensorShape({static_cast<int64_t>(size)}), &temp);
      if (!s.ok()) return nullptr;
      workspace_tensors.emplace_back(temp);
      return ::musa::dnn::MemoryHandler(temp.flat<uint8_t>().data(), [](void*) {});
    };
    auto* musa_device = static_cast<MusaDevice*>(ctx->device());
    auto maintainer = musa_device->GetMemMaintainer(mem_alloc_func);

    ::musa::dnn::Tensor t_in          = CreateMTensor(input);
    ::musa::dnn::Tensor t_out_val     = CreateMTensor(temp_out_values);
    ::musa::dnn::Tensor t_out_indices = CreateMTensor(*temp_out_indices);
    ::musa::dnn::Tensor t_counts      = CreateMTensor(temp_counts);

    ::musa::dnn::Unique op;
    ::musa::dnn::Status mode_status = op.SetMode(::musa::dnn::Unique::Mode::UNSORTED);
    if (mode_status != ::musa::dnn::Status::SUCCESS) {
      ctx->SetStatus(errors::Internal("Unique SetMode failed"));
      return;
    }
    auto& handle = GetHandleByCtx(ctx);
    op.Run(handle, t_out_val, t_out_indices, t_counts, t_in, maintainer);

    musaStream_t stream = GetMusaStreamByCtx(ctx);
    std::vector<int64_t> nd_info;
    t_counts.GetNdInfo(nd_info);
    int64_t num_unique = nd_info[0];
    Tensor* out_values = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_unique}), &out_values));
    musaMemcpyAsync(out_values->data(), temp_out_values.data(),
                    num_unique * sizeof(T), musaMemcpyDeviceToDevice, stream);

    DumpMusaTensorToHost(ctx, out_values, "out_values");
    DumpMusaTensorToHost(ctx, *temp_out_indices, "*temp_out_indices");
    DumpMusaTensorToHost(ctx, temp_out_values, "temp_out_values");
    GetDeviceByCtx(ctx)->event_mgr()->ThenExecute(stream, [temp_out_values]() { });
  }
};

#define REGISTER_MUSA_UNIQUE(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_MTGPU)              \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          MusaUniqueOp<type, int32>);            \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_MTGPU)              \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          MusaUniqueOp<type, int64>);

REGISTER_MUSA_UNIQUE(float);
REGISTER_MUSA_UNIQUE(double);
REGISTER_MUSA_UNIQUE(int32);
REGISTER_MUSA_UNIQUE(int64);
REGISTER_MUSA_UNIQUE(Eigen::half);
REGISTER_MUSA_UNIQUE(bfloat16);

}  // namespace musa
}  // namespace tensorflow
