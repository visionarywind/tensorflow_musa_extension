#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../utils_op.h"

#ifndef MAX_DIM
#define MAX_DIM 4  // NHWC 4d support only
#endif

extern "C" {
    void nd_pad_kernel_launcher_float(const float *, float *, const int, const int64_t *, const int64_t *, const int64_t *, const int64_t *, const float, const int64_t, const musaStream_t);
    void nd_pad_kernel_launcher_double(const double *, double *, const int, const int64_t *, const int64_t *, const int64_t *, const int64_t *, const double, const int64_t, const musaStream_t);
    void nd_pad_kernel_launcher_int32(const int32_t *, int32_t *, const int, const int64_t *, const int64_t *, const int64_t *, const int64_t *, const int32_t, const int64_t, const musaStream_t);
    void nd_pad_kernel_launcher_int64(const int64_t *, int64_t *, const int, const int64_t *, const int64_t *, const int64_t *, const int64_t *, const int64_t, const int64_t, const musaStream_t);
    void nd_pad_kernel_launcher_uint8(const uint8_t *, uint8_t *, const int, const int64_t *, const int64_t *, const int64_t *, const int64_t *, const uint8_t, const int64_t, const musaStream_t);
}

namespace tensorflow {
namespace musa {

namespace {

template <typename ValueT>
std::enable_if_t<std::is_integral<ValueT>::value, mStatus> SetPadValue(
    mPad& pad, ValueT value) {
  return pad.SetValue(static_cast<int64_t>(value));
}

template <typename ValueT>
std::enable_if_t<!std::is_integral<ValueT>::value, mStatus> SetPadValue(
    mPad& pad, ValueT value) {
  return pad.SetValue(static_cast<double>(value));
}

struct FloatTag {};
struct DoubleTag {};
struct Int32Tag {};
struct Int64Tag {};
struct Uint8Tag {};
struct UnsupportedTag {};

// 辅助函数：获取类型标签
template <typename T>
struct TypeTag {
    using type = UnsupportedTag;
};

template <> struct TypeTag<float> { using type = FloatTag; };
template <> struct TypeTag<double> { using type = DoubleTag; };
template <> struct TypeTag<int32_t> { using type = Int32Tag; };
template <> struct TypeTag<int64_t> { using type = Int64Tag; };
template <> struct TypeTag<uint8_t> { using type = Uint8Tag; };

// 具体的实现函数，通过标签区分
void CallPadLauncherImpl(const float *input_data, float *output_data, const int dims,
                         const int64_t *in_dims, const int64_t *out_dims,
                         const int64_t *pad_before, const int64_t *pad_after,
                         const float pad_value, const int64_t total_out_elements,
                         const musaStream_t stream, FloatTag) {
    nd_pad_kernel_launcher_float(input_data, output_data, dims, in_dims, out_dims, 
                                 pad_before, pad_after, pad_value, total_out_elements, stream);
}

void CallPadLauncherImpl(const double *input_data, double *output_data, const int dims,
                         const int64_t *in_dims, const int64_t *out_dims,
                         const int64_t *pad_before, const int64_t *pad_after,
                         const double pad_value, const int64_t total_out_elements,
                         const musaStream_t stream, DoubleTag) {
    nd_pad_kernel_launcher_double(input_data, output_data, dims, in_dims, out_dims, 
                                  pad_before, pad_after, pad_value, total_out_elements, stream);
}

void CallPadLauncherImpl(const int32_t *input_data, int32_t *output_data, const int dims,
                         const int64_t *in_dims, const int64_t *out_dims,
                         const int64_t *pad_before, const int64_t *pad_after,
                         const int32_t pad_value, const int64_t total_out_elements,
                         const musaStream_t stream, Int32Tag) {
    nd_pad_kernel_launcher_int32(input_data, output_data, dims, in_dims, out_dims, 
                                 pad_before, pad_after, pad_value, total_out_elements, stream);
}

void CallPadLauncherImpl(const int64_t *input_data, int64_t *output_data, const int dims,
                         const int64_t *in_dims, const int64_t *out_dims,
                         const int64_t *pad_before, const int64_t *pad_after,
                         const int64_t pad_value, const int64_t total_out_elements,
                         const musaStream_t stream, Int64Tag) {
    nd_pad_kernel_launcher_int64(input_data, output_data, dims, in_dims, out_dims, 
                                 pad_before, pad_after, pad_value, total_out_elements, stream);
}

void CallPadLauncherImpl(const uint8_t *input_data, uint8_t *output_data, const int dims,
                         const int64_t *in_dims, const int64_t *out_dims,
                         const int64_t *pad_before, const int64_t *pad_after,
                         const uint8_t pad_value, const int64_t total_out_elements,
                         const musaStream_t stream, Uint8Tag) {
    nd_pad_kernel_launcher_uint8(input_data, output_data, dims, in_dims, out_dims, 
                                 pad_before, pad_after, pad_value, total_out_elements, stream);
}

// 处理不支持的类型
void CallPadLauncherImpl(const void *, void *, const int, const int64_t *, const int64_t *,
                         const int64_t *, const int64_t *, const int64_t, const int64_t,
                         const musaStream_t, UnsupportedTag) {
    // 这里可以抛出异常或记录错误，但由于是 void 返回，通常在编译期通过 static_assert 拦截更好
    // 在 C++14 中，我们可以依赖下面的主模板中的 static_assert
}

template <typename T>
void CallPadLauncher(const T *input_data, T *output_data, const int dims,
                     const int64_t *in_dims, const int64_t *out_dims,
                     const int64_t *pad_before, const int64_t *pad_after,
                     const T pad_value, const int64_t total_out_elements,
                     const musaStream_t stream) {
    // 确保类型是支持的，否则编译失败
    static_assert(!std::is_same<typename TypeTag<T>::type, UnsupportedTag>::value, 
                  "Unsupported type for nd_pad_kernel_launcher");
    
    // 调用对应的实现
    CallPadLauncherImpl(input_data, output_data, dims, in_dims, out_dims, 
                        pad_before, pad_after, pad_value, total_out_elements, 
                        stream, typename TypeTag<T>::type());
}

} // namespace

template <typename T, typename Tpadding>
class MusaPadOp : public OpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction* context) : OpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& paddings = context->input(1);
    const int dims = input.dims();

    static constexpr int kMaxDims = 8;
    OP_REQUIRES(context, dims >= 1 && dims <= kMaxDims,
                errors::Unimplemented(
                    "MusaPadGeneric supports 1-8D input, got ", dims, "D"));
    OP_REQUIRES(
        context,
        paddings.dims() == 2 && paddings.dim_size(0) == dims &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("Paddings must be [", dims, ", 2], got ",
                                paddings.shape().DebugString()));

    T pad_value = T(0);
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(constant_values.shape()),
                  errors::InvalidArgument("constant_values must be scalar"));
      pad_value = constant_values.scalar<T>()();
    }

    typename TTypes<Tpadding>::ConstMatrix paddings_mat =
        paddings.matrix<Tpadding>();
    int64_t pad_before[8], pad_after[8];
    for (int d = 0; d < dims; ++d) {
      pad_before[d] = static_cast<int64_t>(paddings_mat(d, 0));
      pad_after[d] = static_cast<int64_t>(paddings_mat(d, 1));
      OP_REQUIRES(context, pad_before[d] >= 0 && pad_after[d] >= 0,
                  errors::InvalidArgument("Paddings must be non-negative"));
    }

    TensorShape out_shape;
    int64_t out_size = 1;
    int64_t input_dims[8], out_dims[8];
    for (int d = 0; d < dims; ++d) {
      int64_t out_dim = input.dim_size(d) + pad_before[d] + pad_after[d];
      input_dims[d] = input.dim_size(d);
      out_dims[d] = out_dim;
      out_shape.AddDim(out_dim);
      out_size *= out_dim;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    if (out_size == 0) return;

    Tensor pad_before_dev, pad_after_dev, pad_input_dims, pad_out_dims;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DT_INT64, TensorShape({dims}), &pad_before_dev));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT64, TensorShape({dims}), &pad_after_dev));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT64, TensorShape({dims}), &pad_input_dims));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_INT64, TensorShape({dims}), &pad_out_dims));

    musaStream_t stream = GetMusaStreamByCtx(context);
    musaMemcpyAsync(pad_before_dev.flat<int64_t>().data(),
                               pad_before, dims * sizeof(int64_t),
                               musaMemcpyHostToDevice, stream);
    musaMemcpyAsync(pad_after_dev.flat<int64_t>().data(), pad_after,
                               dims * sizeof(int64_t), musaMemcpyHostToDevice,
                               stream);
    musaMemcpyAsync(pad_input_dims.flat<int64_t>().data(), input_dims,
                               dims * sizeof(int64_t), musaMemcpyHostToDevice,
                               stream);
    musaMemcpyAsync(pad_out_dims.flat<int64_t>().data(), out_dims,
                               dims * sizeof(int64_t), musaMemcpyHostToDevice,
                               stream);

    CallPadLauncher<T>(
        input.flat<T>().data(), output->flat<T>().data(), dims,
        pad_input_dims.flat<int64_t>().data(),
        pad_out_dims.flat<int64_t>().data(),
        pad_before_dev.flat<int64_t>().data(),
        pad_after_dev.flat<int64_t>().data(), pad_value, out_size, stream);

    // 错误检查
    musaError_t err = musaGetLastError();
    OP_REQUIRES(context, err == musaSuccess,
                errors::Internal("MUSA Pad kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

#define REGISTER_MUSA_PAD_TYPE(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(Name("Pad")                                         \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          MusaPadOp<TYPE, int32>);                             \
  REGISTER_KERNEL_BUILDER(Name("Pad")                                         \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int64>("Tpaddings")              \
                              .HostMemory("paddings"),                         \
                          MusaPadOp<TYPE, int64>);                             \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                                       \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int32>("Tpaddings")              \
                              .HostMemory("paddings")                          \
                              .HostMemory("constant_values"),                  \
                          MusaPadOp<TYPE, int32>);                             \
  REGISTER_KERNEL_BUILDER(Name("PadV2")                                       \
                              .Device(DEVICE_MTGPU)                            \
                              .TypeConstraint<TYPE>("T")                       \
                              .TypeConstraint<int64>("Tpaddings")              \
                              .HostMemory("paddings")                          \
                              .HostMemory("constant_values"),                  \
                          MusaPadOp<TYPE, int64>);

REGISTER_MUSA_PAD_TYPE(float);
REGISTER_MUSA_PAD_TYPE(int32);
REGISTER_MUSA_PAD_TYPE(int64);
REGISTER_MUSA_PAD_TYPE(double);
REGISTER_MUSA_PAD_TYPE(uint8);

#undef REGISTER_MUSA_PAD_TYPE

}  // namespace musa
}  // namespace tensorflow
