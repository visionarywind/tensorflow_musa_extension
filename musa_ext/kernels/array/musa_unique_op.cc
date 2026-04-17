#include <list>
#include <vector>

#include "../utils_op.h"
#include "mu/device/musa_device.h"
#include "mu/device/musa_memcpy.h"
#include "mu/device/musa_memset.h"

namespace tensorflow {
namespace musa {

namespace {
void DumpMusaTensorToHost(OpKernelContext* ctx, const Tensor& device_tensor, const string& name) {
  if (device_tensor.NumElements() == 0) {
    LOG(ERROR) << "[Dump] " << name << " | Empty Tensor";
    return;
  }

  const DataType dtype = device_tensor.dtype();
  LOG(ERROR) << "==================================================";
  LOG(ERROR) << "[Dump] " << name
             << " | Type: " << DataTypeString(dtype)
             << " | Shape: " << device_tensor.shape().DebugString();

  Tensor host_tensor;
  AllocatorAttributes cpu_alloc;
  cpu_alloc.set_on_host(true);
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(dtype, device_tensor.shape(), &host_tensor, cpu_alloc));

  musaStream_t stream = GetMusaStreamByCtx(ctx);
  musaError_t err = musaMemcpyAsync(
      host_tensor.data(), device_tensor.data(),
      device_tensor.TotalBytes(), musaMemcpyDeviceToHost, stream
  );
  musaStreamSynchronize(stream);
  OP_REQUIRES(ctx, err == musaSuccess, errors::Internal("Dump拷贝失败"));

  std::stringstream ss;
  ss << "Data: ";
  const int64_t max_elems = host_tensor.NumElements();
  switch (dtype) {
    case DT_INT32: {
      const int32* data = host_tensor.flat<int32>().data();
      for (int64_t i = 0; i < max_elems; ++i) ss << data[i] << "\t";
      break;
    }
    case DT_INT64: {
      const int64* data = host_tensor.flat<int64>().data();
      for (int64_t i = 0; i < max_elems; ++i) ss << data[i] << "\t";
      break;
    }
    case DT_FLOAT: {
      const float* data = host_tensor.flat<float>().data();
      for (int64_t i = 0; i < max_elems; ++i) ss << data[i] << "\t";
      break;
    }
    default:
      ss << "unsupported type";
  }
  LOG(ERROR) << ss.str();
  LOG(ERROR) << "==================================================\n";
}
}

template <typename T, typename OutIdxT>
class MusaUniqueOp : public MusaOpKernel {
 public:
  explicit MusaUniqueOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input = ctx->input(0);
    DumpMusaTensorToHost(ctx, input, "input");
    OP_REQUIRES(ctx, input.dims() <= 1,
                errors::InvalidArgument("Unique only supports 1D tensor, got ", input.dims(), "D"));

    const int64_t input_num = input.NumElements();
    musaStream_t stream = GetMusaStreamByCtx(ctx);
    auto& handle = GetHandleByCtx(ctx);
    auto* musa_device = static_cast<MusaDevice*>(ctx->device());
    OP_REQUIRES(ctx, musa_device != nullptr, errors::Internal("MusaDevice is null"));

    if (input_num == 0) {
      ctx->set_output(0, Tensor(input.dtype(), TensorShape()));
      ctx->set_output(1, Tensor(DataTypeToEnum<OutIdxT>::value, TensorShape()));
      return;
    }

    Tensor temp_out_values;
    Tensor temp_out_indices;
    Tensor temp_counts;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(input.dtype(), input.shape(), &temp_out_values));
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<OutIdxT>::value, input.shape(), &temp_out_indices));
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
    auto maintainer = musa_device->GetMemMaintainer(mem_alloc_func);

    ::musa::dnn::Tensor t_in          = CreateMTensor(input);
    ::musa::dnn::Tensor t_out_val     = CreateMTensor(temp_out_values);
    ::musa::dnn::Tensor t_out_indices = CreateMTensor(temp_out_indices);
    ::musa::dnn::Tensor t_counts      = CreateMTensor(temp_counts);

    ::musa::dnn::Unique op;
    ::musa::dnn::Status mode_status = op.SetMode(::musa::dnn::Unique::Mode::UNSORTED);
    if (mode_status != ::musa::dnn::Status::SUCCESS) {
      ctx->SetStatus(errors::Internal("Unique SetMode failed"));
      return;
    }
    mStatus status = op.Run(handle, t_out_val, t_out_indices, t_counts, t_in, maintainer);

    musaStreamSynchronize(stream);
    if (status != ::musa::dnn::Status::SUCCESS) {
      ctx->SetStatus(errors::Internal("muDNN Unique Run failed"));
      return;
    }

    std::vector<OutIdxT> counts_host(input_num);
    musaMemcpyAsync(counts_host.data(), temp_counts.flat<OutIdxT>().data(),
                    input_num * sizeof(OutIdxT), musaMemcpyDeviceToHost, stream);
    musaStreamSynchronize(stream);

    int64_t num_unique = 0;
    for (int64_t i = 0; i < input_num; ++i) {
      if (counts_host[i] == 0) break;
      num_unique++;
    }

    // 分配最终输出张量
    Tensor* out_values = nullptr;
    Tensor* out_indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_unique}), &out_values));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &out_indices));

    // 只拷贝有效数据，无脏数据
    musaMemcpyAsync(out_values->data(), temp_out_values.data(),
                    num_unique * sizeof(T), musaMemcpyDeviceToDevice, stream);
    musaMemcpyAsync(out_indices->data(), temp_out_indices.data(),
                    input_num * sizeof(OutIdxT), musaMemcpyDeviceToDevice, stream);
    musaStreamSynchronize(stream);

    // 打印结果
    DumpMusaTensorToHost(ctx, *out_values, "unique_values");
    DumpMusaTensorToHost(ctx, *out_indices, "unique_indices");
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
