#include <regex>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>
#include "../utils_op.h"

namespace tensorflow {
namespace musa {

class MusaStaticRegexFullMatchOp : public OpKernel {
 public:
  explicit MusaStaticRegexFullMatchOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    string pattern_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pattern", &pattern_str));

    try {
      regex_ = std::regex(pattern_str, std::regex_constants::optimize);
    } catch (const std::regex_error& e) {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Invalid regex pattern: ", e.what()));
    }
  }

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    std::stringstream ss;
    ss << "[MUSA Debug] Thread: " << std::this_thread::get_id()
              << " | Op: " << __FILE__
              << " | Method: " << __FUNCTION__;
    int input_num = ctx->num_inputs();
    for (int i = 0; i < input_num; ++i) {
        ss << " | Input " << i << ": " << ctx->input(i).shape().DebugString();
    }
    LOG(ERROR) << ss.str();
  }
  static bool sync_execute = std::getenv("MUSA_LAUNCH_BLOCKING") == "1";
  if (sync_execute) {
    musaStreamSynchronize(GetMusaStreamByCtx(ctx));
  }

    const Tensor& input_tensor = ctx->input(0);
    const auto& input_flat = input_tensor.flat<tstring>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<bool>();

    const int64 N = input_tensor.NumElements();

    for (int64 i = 0; i < N; ++i) {
      const std::string& s = input_flat(i);
      bool is_match = std::regex_match(s, regex_);
      output_flat(i) = is_match;
    }
  }

 private:
  std::regex regex_;
};

#define REGISTER_MUSA_KERNEL()                         \
  REGISTER_KERNEL_BUILDER(Name("StaticRegexFullMatch") \
                              .Device("MUSA")          \
                              .HostMemory("input")     \
                              .HostMemory("output"),   \
                          MusaStaticRegexFullMatchOp);

REGISTER_MUSA_KERNEL();

}  // namespace musa
}  // namespace tensorflow
