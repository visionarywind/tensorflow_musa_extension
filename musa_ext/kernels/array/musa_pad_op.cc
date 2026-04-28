#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "../utils_op.h"

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

struct CompressionPlan {
  std::vector<int64_t> merged_input_sizes;
  std::vector<int> kept_dims;
  std::vector<int> reversed_kept_pad_pairs;
  bool is_valid;
};
CompressionPlan GenerateCompressionPlan(
    int dims, const std::vector<std::pair<int, int>>& pad_pairs_per_dim) {
  CompressionPlan plan;
  plan.is_valid = false;

  std::vector<bool> needs_padding(dims, false);
  int num_pad_dims = 0;
  for (int d = 0; d < dims; ++d) {
    if (pad_pairs_per_dim[d].first != 0 || pad_pairs_per_dim[d].second != 0) {
      needs_padding[d] = true;
      num_pad_dims++;
    }
  }
  if (num_pad_dims > 3) {
    return plan;
  }

  std::vector<int> kept_dims;
  int current_merge_start = -1;

  for (int d = 0; d < dims; ++d) {
    if (needs_padding[d]) {
      // 遇到需 Padding 维度：先处理之前的合并块
      if (current_merge_start != -1) {
        // 合并块标记：用 -1 表示
        kept_dims.push_back(-1);
        current_merge_start = -1;
      }
      // 保留此维度
      kept_dims.push_back(d);
    } else {
      // 无 Padding 维度：开始或继续合并
      if (current_merge_start == -1) {
        current_merge_start = d;
      }
    }
  }
  // 处理最后一个合并块
  if (current_merge_start != -1) {
    kept_dims.push_back(-1);
  }

  // 4. 检查合并后维度 ≤ 3
  if (kept_dims.size() > 3) {
    return plan;
  }

  // 5. 生成反序 Padding 对（适配 muDNN）
  // 先按正向顺序收集保留维度的 Padding
  std::vector<int> forward_pad_pairs;
  for (int d : kept_dims) {
    if (d != -1) {
      forward_pad_pairs.push_back(pad_pairs_per_dim[d].first);
      forward_pad_pairs.push_back(pad_pairs_per_dim[d].second);
    } else {
      forward_pad_pairs.push_back(0); // before = 0
      forward_pad_pairs.push_back(0); // after  = 0
    }
  }

  // 反转以适配 muDNN 的内外层顺序
    std::vector<int> reversed_pad_pairs;
    // 按压缩后维度从内到外：反向遍历 kept_dims
    for (int i = static_cast<int>(kept_dims.size()) - 1; i >= 0; --i) {
        int marker = kept_dims[i];
        if (marker != -1) {
            reversed_pad_pairs.push_back(pad_pairs_per_dim[marker].first);
            reversed_pad_pairs.push_back(pad_pairs_per_dim[marker].second);
        } else {
            reversed_pad_pairs.push_back(0);
            reversed_pad_pairs.push_back(0);
        }
    }

  // 6. 填充规划结果
  plan.kept_dims = kept_dims;
  plan.reversed_kept_pad_pairs = reversed_pad_pairs;
  plan.is_valid = true;

  return plan;
}
}  // namespace

template <typename T, typename Tpadding>
class MusaPadOp : public MusaOpKernel {
 public:
  explicit MusaPadOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  bool IsExpensive() override { return false; }

  void Compute(OpKernelContext* context) override {
    static bool enable_print = std::getenv("ENABLE_PRINT") == "1";
    const Tensor& input = context->input(0);
    const Tensor& paddings = context->input(1);
    const int dims = input.dims();
    if (enable_print) {
      LOG(ERROR) << "Input - dims: " << dims << ", shape: " << input.shape().DebugString();
      LOG(ERROR) << "Paddings - shape: " << paddings.shape().DebugString();
    }
    static constexpr int kMinDims = 0;
    static constexpr int kMaxDims = 8;

    OP_REQUIRES(context, kMinDims <= dims && dims <= kMaxDims,
                errors::Unimplemented("Inputs rank not in [", kMinDims, ",",
                                      kMaxDims, "]: ", dims));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsMatrix(paddings.shape()) &&
            paddings.dim_size(1) == 2,
        errors::InvalidArgument("Paddings must be a matrix with 2 columns: ",
                                paddings.shape().DebugString()));
    OP_REQUIRES(context, dims == paddings.dim_size(0),
                errors::InvalidArgument(
                    "The first dimension of paddings must be the rank of inputs ",
                    paddings.shape().DebugString(), " ",
                    input.shape().DebugString()));

    T pad_value = T();
    if (context->num_inputs() == 3) {
      const Tensor& constant_values = context->input(2);
      OP_REQUIRES(context, TensorShapeUtils::IsScalar(constant_values.shape()),
                  errors::InvalidArgument("Constant_values must be a scalar: ",
                                          constant_values.shape().DebugString()));
      pad_value = constant_values.scalar<T>()();
    }

    TensorShape output_shape;
    typename TTypes<Tpadding>::ConstMatrix paddings_matrix =
        paddings.matrix<Tpadding>();
    std::vector<std::pair<int, int>> pad_pairs_per_dim;
    pad_pairs_per_dim.reserve(dims);
    for (int d = 0; d < dims; ++d) {
      const int64_t before = static_cast<int64_t>(paddings_matrix(d, 0));
      const int64_t after = static_cast<int64_t>(paddings_matrix(d, 1));
      OP_REQUIRES(context, before >= 0 && after >= 0,
                  errors::InvalidArgument("Paddings must be non-negative: ",
                                          before, " ", after));
      OP_REQUIRES(context,
                  before <= std::numeric_limits<int>::max() &&
                      after <= std::numeric_limits<int>::max(),
                  errors::InvalidArgument(
                      "Paddings must fit in int32 for MUSA: ", before, " ",
                      after));
      pad_pairs_per_dim.emplace_back(static_cast<int>(before),
                                     static_cast<int>(after));
      const int64_t size_d = input.dim_size(d);
      OP_REQUIRES_OK(
          context,
          output_shape.AddDimWithStatus(before + size_d + after));
    }

    // 生成原始反序 Padding 对（用于 ≤3D 情况）
    std::vector<int> original_pad_pairs;
    original_pad_pairs.reserve(dims * 2);
    for (int d = dims - 1; d >= 0; --d) {
      original_pad_pairs.push_back(pad_pairs_per_dim[d].first);
      original_pad_pairs.push_back(pad_pairs_per_dim[d].second);
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output));
    if (output->NumElements() == 0) return;

    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);
    auto& handle = GetHandleByCtx(context);

    mPad pad_op;
    pad_op.SetMode(mPad::Mode::CONSTANT);
    SetPadValue(pad_op, pad_value);

    ::musa::dnn::Status status;
    const int kMaxSupportedDims = 3;

    if (dims > kMaxSupportedDims) {
      // 1. 生成修复后的压缩规划
      CompressionPlan plan = GenerateCompressionPlan(dims, pad_pairs_per_dim);

      OP_REQUIRES(context, plan.is_valid,
                  errors::Unimplemented(
                      "MUSA Pad only supports up to 3 dimensions with padding. "
                      "Got ",
                      dims, " dimensions with ",
                      [&]() {
                        int cnt = 0;
                        for (auto& p : pad_pairs_per_dim)
                          if (p.first || p.second) cnt++;
                        return cnt;
                      }(),
                      " padded dimensions, which cannot be compressed."));

      // 2. 根据规划计算实际的 3D 形状
      std::vector<int64_t> new_input_shape;
      std::vector<int64_t> new_output_shape;
      int current_original_dim = 0;

      for (int marker : plan.kept_dims) {
        if (marker != -1) {
          // 保留的需 Padding 维度
          int d = marker;
          int64_t in_size = input.dim_size(d);
          int64_t out_size = in_size + pad_pairs_per_dim[d].first + pad_pairs_per_dim[d].second;
          new_input_shape.push_back(in_size);
          new_output_shape.push_back(out_size);
          current_original_dim = d + 1;
        } else {
          // 合并块：合并从 current_original_dim 开始的连续无 Padding 维度
          int64_t merged_in_size = 1;
          int64_t merged_out_size = 1;
          
          while (current_original_dim < dims) {
            bool needs_pad = (pad_pairs_per_dim[current_original_dim].first != 0 ||
                             pad_pairs_per_dim[current_original_dim].second != 0);
            if (needs_pad) break;
            
            merged_in_size *= input.dim_size(current_original_dim);
            merged_out_size *= input.dim_size(current_original_dim);
            current_original_dim++;
          }
          
          new_input_shape.push_back(merged_in_size);
          new_output_shape.push_back(merged_out_size);
        }
      }

      // 3. 验证最终维度
      OP_REQUIRES(context, new_input_shape.size() <= kMaxSupportedDims,
                  errors::Internal("Compression failed: final dims ",
                                   new_input_shape.size(), " > 3"));

      if (enable_print) {
        LOG(ERROR) << "Compressed input shape: [" 
                  << [&]() { std::string s; for (auto v : new_input_shape) s += std::to_string(v) + ","; return s; }() 
                  << "]";
        LOG(ERROR) << "Compressed output shape: [" 
                  << [&]() { std::string s; for (auto v : new_output_shape) s += std::to_string(v) + ","; return s; }() 
                  << "]";
        LOG(ERROR) << "Reversed kept pad pairs: [" 
                  << [&]() { std::string s; for (auto v : plan.reversed_kept_pad_pairs) s += std::to_string(v) + ","; return s; }() 
                  << "]";
      }

      // 4. 创建重塑后的 mTensor 视图
      mTensor in_mt_reshaped = CreateMTensor(input, format_);
      mTensor out_mt_reshaped = CreateMTensor(*output, format_);
      
      in_mt_reshaped.SetNdInfo(static_cast<int>(new_input_shape.size()), new_input_shape.data());
      out_mt_reshaped.SetNdInfo(static_cast<int>(new_output_shape.size()), new_output_shape.data());

      // 5. 设置合适的 mTensor 格式（根据压缩后的维度）
      // 注意：请根据 muDNN 实际支持的格式调整以下映射
      if (new_input_shape.size() == 1) {
          in_mt_reshaped.SetFormat(::musa::dnn::Tensor::Format::NCW);
      } else if (new_input_shape.size() == 2) {
          in_mt_reshaped.SetFormat(::musa::dnn::Tensor::Format::NCHW);
      } else if (new_input_shape.size() == 3) {
          in_mt_reshaped.SetFormat(::musa::dnn::Tensor::Format::NCDHW);
      }

      // 6. 验证设置后的形状
      std::vector<int64_t> in_nd_info, out_nd_info;
      in_mt_reshaped.GetNdInfo(in_nd_info);
      out_mt_reshaped.GetNdInfo(out_nd_info);
      if (enable_print) {
        LOG(ERROR) << "in_mt_reshaped nd info size: " << in_nd_info.size();
        LOG(ERROR) << "out_mt_reshaped nd info size: " << out_nd_info.size();
      }

      // 7. 设置 Padding 并运行
      if (!plan.reversed_kept_pad_pairs.empty()) {
        pad_op.SetPaddingInfo(static_cast<int>(plan.reversed_kept_pad_pairs.size()),
                              plan.reversed_kept_pad_pairs.data());
      } else {
        pad_op.SetPaddingInfo(0, nullptr);
      }

      status = pad_op.Run(handle, out_mt_reshaped, in_mt_reshaped);
    } else {
      // 原有 ≤3D 逻辑
      if (!original_pad_pairs.empty()) {
        pad_op.SetPaddingInfo(static_cast<int>(original_pad_pairs.size()),
                              original_pad_pairs.data());
      } else {
        pad_op.SetPaddingInfo(0, nullptr);
      }
      std::vector<int64_t> in_nd_info, out_nd_info;
      in_mt.GetNdInfo(in_nd_info);
      out_mt.GetNdInfo(out_nd_info);
      if (enable_print) {
        LOG(ERROR) << "in_mt nd info size: " << in_nd_info.size();
        LOG(ERROR) << "out_mt nd info size: " << out_nd_info.size();
      }
      status = pad_op.Run(handle, out_mt, in_mt);
    }

    OP_REQUIRES(context, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Pad execution failed with status: ", static_cast<int>(status)));
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
REGISTER_MUSA_PAD_TYPE(Eigen::half);
REGISTER_MUSA_PAD_TYPE(bfloat16);
REGISTER_MUSA_PAD_TYPE(double);
REGISTER_MUSA_PAD_TYPE(uint8);

#undef REGISTER_MUSA_PAD_TYPE

}  // namespace musa
}  // namespace tensorflow
