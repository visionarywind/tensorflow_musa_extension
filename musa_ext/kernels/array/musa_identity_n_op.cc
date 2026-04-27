#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {
class MusaIdentityNOp : public OpKernel {
 public:
  explicit MusaIdentityNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    std::stringstream ss;
    ss << "[MUSA Debug] Thread: " << std::this_thread::get_id()
              << " | Op: " << __FILE__
              << " | Method: " << __FUNCTION__;
    int input_num = context->num_inputs();
    for (int i = 0; i < input_num; ++i) {
        ss << " | Input " << i << ": " << context->input(i).shape().DebugString();
    }
    LOG(ERROR) << ss.str();
  }
  static bool sync_execute = std::getenv("MUSA_LAUNCH_BLOCKING") == "1";
  if (sync_execute) {
    musaStreamSynchronize(GetMusaStreamByCtx(context));
  }

    OpInputList input;
    OpOutputList output;

    OP_REQUIRES_OK(context, context->input_list("input", &input));
    OP_REQUIRES_OK(context, context->output_list("output", &output));
    OP_REQUIRES(context, input.size() == output.size(),
                errors::InvalidArgument("Input and output counts must match."));
    // Skip checking for TPU here and conduct `IdentityN` directly.
    for (int i = 0; i < input.size(); ++i) {
      output.set(i, input[i]);
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_KERNEL_BUILDER(Name("IdentityN").Device("MUSA"), MusaIdentityNOp);

}  // namespace musa
}  // namespace tensorflow
