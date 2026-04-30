#include <type_traits>
#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/padding.h"
#include "utils/logging.h"

using namespace tensorflow;

// ----------------------------------------------------------------------------
// 核函数启动器声明：从.mu.cc中导入对外暴露的C接口
// ----------------------------------------------------------------------------
extern "C" {
// 启动器声明宏：简化重复的声明代码
#define DECLARE_LAUNCHER(T, IndexT, Name)                                      \
  void Name(const T* images, T* patches, int64_t batch_size, int64_t in_h,     \
            int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w,          \
            int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,        \
            int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left, \
            musaStream_t stream);

// 基础类型启动器声明
DECLARE_LAUNCHER(float, int, LaunchExtractImagePatchesFloatInt32)
DECLARE_LAUNCHER(float, int64_t, LaunchExtractImagePatchesFloatInt64)
DECLARE_LAUNCHER(double, int, LaunchExtractImagePatchesDoubleInt32)
DECLARE_LAUNCHER(double, int64_t, LaunchExtractImagePatchesDoubleInt64)
DECLARE_LAUNCHER(int32_t, int, LaunchExtractImagePatchesInt32Int32)
DECLARE_LAUNCHER(int32_t, int64_t, LaunchExtractImagePatchesInt32Int64)
DECLARE_LAUNCHER(int64_t, int, LaunchExtractImagePatchesInt64Int32)
DECLARE_LAUNCHER(int64_t, int64_t, LaunchExtractImagePatchesInt64Int64)
DECLARE_LAUNCHER(uint8_t, int, LaunchExtractImagePatchesUInt8Int32)
DECLARE_LAUNCHER(uint8_t, int64_t, LaunchExtractImagePatchesUInt8Int64)
DECLARE_LAUNCHER(bool, int, LaunchExtractImagePatchesBoolInt32)
DECLARE_LAUNCHER(bool, int64_t, LaunchExtractImagePatchesBoolInt64)

// FP16启动器声明
void LaunchExtractImagePatchesHalfInt32(
    const void* images, void* patches, int64_t batch_size, int64_t in_h,
    int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w, int64_t kH,
    int64_t kW, int64_t stride_h, int64_t stride_w, int64_t rate_h,
    int64_t rate_w, int64_t pad_top, int64_t pad_left, musaStream_t stream);
void LaunchExtractImagePatchesHalfInt64(
    const void* images, void* patches, int64_t batch_size, int64_t in_h,
    int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w, int64_t kH,
    int64_t kW, int64_t stride_h, int64_t stride_w, int64_t rate_h,
    int64_t rate_w, int64_t pad_top, int64_t pad_left, musaStream_t stream);

// BF16启动器声明
void LaunchExtractImagePatchesBFloat16Int32(
    const void* images, void* patches, int64_t batch_size, int64_t in_h,
    int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w, int64_t kH,
    int64_t kW, int64_t stride_h, int64_t stride_w, int64_t rate_h,
    int64_t rate_w, int64_t pad_top, int64_t pad_left, musaStream_t stream);
void LaunchExtractImagePatchesBFloat16Int64(
    const void* images, void* patches, int64_t batch_size, int64_t in_h,
    int64_t in_w, int64_t in_c, int64_t out_h, int64_t out_w, int64_t kH,
    int64_t kW, int64_t stride_h, int64_t stride_w, int64_t rate_h,
    int64_t rate_w, int64_t pad_top, int64_t pad_left, musaStream_t stream);

#undef DECLARE_LAUNCHER
}

namespace tensorflow {
namespace musa {
namespace {

// ----------------------------------------------------------------------------
// 1. 定义标签结构体
// ----------------------------------------------------------------------------
struct FloatTag {};
struct DoubleTag {};
struct Int32Tag {};
struct Int64Tag {};
struct HalfTag {};
struct UInt8Tag {};
struct BoolTag {};

template <typename T>
struct TypeTag {
  using type = void;
};

template <> struct TypeTag<float> { using type = FloatTag; };
template <> struct TypeTag<double> { using type = DoubleTag; };
template <> struct TypeTag<int32_t> { using type = Int32Tag; };
template <> struct TypeTag<int64_t> { using type = Int64Tag; };
template <> struct TypeTag<Eigen::half> { using type = HalfTag; };
template <> struct TypeTag<uint8_t> { using type = UInt8Tag; };
template <> struct TypeTag<bool> { using type = BoolTag; };

// ----------------------------------------------------------------------------
// 2. 核心实现：针对每种类型和索引类型的独立重载函数
//    注意：这些不是模板函数，而是普通函数重载。
//    编译器只会实例化被调用的那个具体函数。
// ----------------------------------------------------------------------------

// --- Float Implementations ---
void LaunchImpl(FloatTag, std::true_type, /*is_int32*/
                const float* images, float* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesFloatInt32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(FloatTag, std::false_type, /*is_int64*/
                const float* images, float* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesFloatInt64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- Double Implementations ---
void LaunchImpl(DoubleTag, std::true_type,
                const double* images, double* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesDoubleInt32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(DoubleTag, std::false_type,
                const double* images, double* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesDoubleInt64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- Int32 Implementations ---
void LaunchImpl(Int32Tag, std::true_type,
                const int32_t* images, int32_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesInt32Int32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(Int32Tag, std::false_type,
                const int32_t* images, int32_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesInt32Int64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- Int64 Implementations ---
void LaunchImpl(Int64Tag, std::true_type,
                const int64_t* images, int64_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesInt64Int32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(Int64Tag, std::false_type,
                const int64_t* images, int64_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesInt64Int64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- Half Implementations ---
void LaunchImpl(HalfTag, std::true_type,
                const Eigen::half* images, Eigen::half* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesHalfInt32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(HalfTag, std::false_type,
                const Eigen::half* images, Eigen::half* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesHalfInt64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- UInt8 Implementations ---
void LaunchImpl(UInt8Tag, std::true_type,
                const uint8_t* images, uint8_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesUInt8Int32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(UInt8Tag, std::false_type,
                const uint8_t* images, uint8_t* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesUInt8Int64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// --- Bool Implementations ---
void LaunchImpl(BoolTag, std::true_type,
                const bool* images, bool* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesBoolInt32(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

void LaunchImpl(BoolTag, std::false_type,
                const bool* images, bool* patches,
                int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
                int64_t out_h, int64_t out_w,
                int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
                int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
                musaStream_t stream) {
  LaunchExtractImagePatchesBoolInt64(images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left, stream);
}

// ----------------------------------------------------------------------------
// 3. 统一入口：通过标签分发调用具体的重载函数
// ----------------------------------------------------------------------------
template <typename T, typename Tpadding>
void LaunchExtractImagePatchesDispatcher(
    const T* images_ptr, T* patches_ptr,
    int64_t batch_size, int64_t in_h, int64_t in_w, int64_t in_c,
    int64_t out_h, int64_t out_w,
    int64_t kH, int64_t kW, int64_t stride_h, int64_t stride_w,
    int64_t rate_h, int64_t rate_w, int64_t pad_top, int64_t pad_left,
    musaStream_t stream) {
  
  using Tag = typename TypeTag<T>::type;
  using IsInt32 = typename std::is_same<Tpadding, int32>::type; // std::true_type or std::false_type
  
  // 这里没有 if-else，直接调用 LaunchImpl。
  // 编译器会根据 Tag 和 IsInt32 的类型，在编译期解析出唯一匹配的重载函数。
  // 其他重载函数根本不会被实例化，因此不会产生类型转换错误。
  LaunchImpl(Tag{}, IsInt32{},
             images_ptr, patches_ptr,
             batch_size, in_h, in_w, in_c,
             out_h, out_w,
             kH, kW, stride_h, stride_w,
             rate_h, rate_w, pad_top, pad_left,
             stream);
}
}
// ----------------------------------------------------------------------------
// MusaExtractImagePatchesOp 主类
// 作用：对接TF算子框架，处理属性解析、参数校验、内存分配、核函数调用
// 模板参数：T=数据类型，Tpadding=padding参数的索引类型（int32/int64）
// ----------------------------------------------------------------------------
template <typename T>
class MusaExtractImagePatchesOp : public OpKernel {
 public:
  // 构造函数：算子初始化，解析属性，参数校验
  explicit MusaExtractImagePatchesOp(OpKernelConstruction* context)
      : OpKernel(context) {
    // 1. 从算子定义中读取属性
    OP_REQUIRES_OK(context, context->GetAttr("ksizes", &ksizes_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(context, context->GetAttr("rates", &rates_));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_type_));

    // 2. TF强制校验：属性必须是4维数组
    OP_REQUIRES(context, ksizes_.size() == 4,
                errors::InvalidArgument("ksizes must be 4 elements, got ",
                                        ksizes_.size()));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("strides must be 4 elements, got ",
                                        strides_.size()));
    OP_REQUIRES(context, rates_.size() == 4,
                errors::InvalidArgument("rates must be 4 elements, got ",
                                        rates_.size()));

    // 3. TF强制校验：batch和channel维度的ksize/stride/rate必须为1
    OP_REQUIRES(
        context, ksizes_[0] == 1 && ksizes_[3] == 1,
        errors::InvalidArgument("Batch and channel ksize must be 1, got [",
                                ksizes_[0], ", ", ksizes_[3], "]"));
    OP_REQUIRES(
        context, strides_[0] == 1 && strides_[3] == 1,
        errors::InvalidArgument("Batch and channel stride must be 1, got [",
                                strides_[0], ", ", strides_[3], "]"));
    OP_REQUIRES(
        context, rates_[0] == 1 && rates_[3] == 1,
        errors::InvalidArgument("Batch and channel rate must be 1, got [",
                                rates_[0], ", ", rates_[3], "]"));

    // 4. 提取核心参数，后续Compute复用
    kH_ = ksizes_[1];
    kW_ = ksizes_[2];
    stride_h_ = strides_[1];
    stride_w_ = strides_[2];
    rate_h_ = rates_[1];
    rate_w_ = rates_[2];

    // 5. 合法性校验：参数必须为正整数
    OP_REQUIRES(context, kH_ > 0 && kW_ > 0,
                errors::InvalidArgument("Kernel size must be positive, got kH=",
                                        kH_, ", kW=", kW_));
    OP_REQUIRES(context, stride_h_ > 0 && stride_w_ > 0,
                errors::InvalidArgument(
                    "Stride must be positive, got stride_h=", stride_h_,
                    ", stride_w=", stride_w_));
    OP_REQUIRES(context, rate_h_ > 0 && rate_w_ > 0,
                errors::InvalidArgument("Rate must be positive, got rate_h=",
                                        rate_h_, ", rate_w=", rate_w_));
  }

  // 核心计算函数：每次算子执行都会调用
  void Compute(OpKernelContext* context) override {
    LOG(ERROR) << "MusaExtractImagePatchesOp entered";

    // 1. 获取输入张量
    const Tensor& images = context->input(0);
    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("Input must be 4D NHWC tensor, got ",
                                        images.dims(), "D"));

    // 2. 解析输入维度
    const int64_t batch_size = images.dim_size(0);
    const int64_t in_h = images.dim_size(1);
    const int64_t in_w = images.dim_size(2);
    const int64_t in_c = images.dim_size(3);

    // 3. 计算有效核尺寸（含膨胀率）
    const int64_t effective_kH = kH_ + (kH_ - 1) * (rate_h_ - 1);
    const int64_t effective_kW = kW_ + (kW_ - 1) * (rate_w_ - 1);

    // 4. 计算输出尺寸与Padding，使用TF原生函数，100%对齐官方逻辑
    int64_t out_h, out_w;
    int64_t pad_top, pad_bottom, pad_left, pad_right;
    auto compute_output_and_padding =
        [](int64_t input_size, int64_t filter_size, int64_t stride,
           Padding padding_type, int64_t* output_size, int64_t* pad_before,
           int64_t* pad_after) {
          if (padding_type == Padding::VALID) {
            // VALID: 不填充，输出大小 = (输入 - 核 + 步长) / 步长
            *output_size = (input_size - filter_size + stride) / stride;
            *pad_before = 0;
            *pad_after = 0;
          } else {
            // SAME: 填充以保持输出大小 = 输入 / 步长 (向上取整)
            *output_size = (input_size + stride - 1) / stride;

            // 计算需要的总填充量
            int64_t padding_needed =
                std::max(static_cast<int64_t>(0), (*output_size - 1) * stride +
                                                      filter_size - input_size);

            // 均匀分配填充：前半部分在前，后半部分在后
            *pad_before = padding_needed / 2;
            *pad_after = padding_needed - *pad_before;
          }
        };

    // 计算高度方向
    compute_output_and_padding(in_h, effective_kH, stride_h_, padding_type_,
                               &out_h, &pad_top, &pad_bottom);

    // 计算宽度方向
    compute_output_and_padding(in_w, effective_kW, stride_w_, padding_type_,
                               &out_w, &pad_left, &pad_right);

    // 5. 计算输出形状并分配设备内存
    const int64_t patch_depth = kH_ * kW_ * in_c;
    TensorShape output_shape({batch_size, out_h, out_w, patch_depth});
    Tensor* patches = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &patches));
    // 空张量直接返回，避免后续开销
    if (output_shape.num_elements() == 0) return;

    // 6. 获取MUSA原生流：复用TF的设备流，实现异步执行，与计算图完美融合
    musaStream_t raw_stream = GetMusaStreamByCtx(context);

    // 7. 获取设备端数据指针
    auto images_flat = images.flat<T>();
    auto patches_flat = patches->flat<T>();
    const T* images_ptr = images_flat.data();
    T* patches_ptr = patches_flat.data();

    // 8. 类型分发：根据模板参数调用对应类型的启动器
    // C++14 兼容写法：使用运行时 if，编译器会优化掉未命中的分支
    LaunchExtractImagePatchesDispatcher<T, int64_t>(
        images_ptr, patches_ptr, batch_size, in_h, in_w, in_c, out_h, out_w,
        kH_, kW_, stride_h_, stride_w_, rate_h_, rate_w_, pad_top, pad_left,
        raw_stream);

    // 9. 检查核函数启动错误，避免静默失败
    musaError_t err = musaGetLastError();
    OP_REQUIRES(
        context, err == musaSuccess,
        errors::Internal("MUSA ExtractImagePatches kernel launch failed: ",
                         musaGetErrorString(err)));
  }

 private:
  // 算子属性缓存
  std::vector<int32> ksizes_;
  std::vector<int32> strides_;
  std::vector<int32> rates_;
  Padding padding_type_;
  // 核心参数缓存，避免Compute中重复计算
  int64_t kH_, kW_;
  int64_t stride_h_, stride_w_;
  int64_t rate_h_, rate_w_;
};

// ----------------------------------------------------------------------------
// 算子注册：将算子注册到TF的MUSA设备上
// ----------------------------------------------------------------------------
#define REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(T)       \
  REGISTER_KERNEL_BUILDER(Name("ExtractImagePatches")                 \
                              .Device("MUSA")                         \
                              .TypeConstraint<T>("T"),                 \
                          MusaExtractImagePatchesOp<T>);

// 注册TF原生支持的所有数据类型，与官方算子完全对齐
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(float)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(double)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(int32_t)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(int64_t)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(uint8_t)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(bool)
REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL(Eigen::half)

// 清理宏定义
#undef REGISTER_MUSA_EXTRACT_IMAGE_PATCHES_KERNEL
}  // namespace musa
}  // namespace tensorflow
