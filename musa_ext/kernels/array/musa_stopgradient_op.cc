#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaStopGradientOp : public MusaOpKernel {
 public:
  explicit MusaStopGradientOp(OpKernelConstruction* context)
      : MusaOpKernel(context) {}

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

    const Tensor& input = ctx->input(0);

    if (IsRefType(ctx->input_dtype(0))) {
      ctx->forward_ref_input_to_ref_output(0, 0);
    } else {
      ctx->set_output(0, input);
    }
  }

  bool IsExpensive() override { return false; }
};

#define REGISTER_MUSA_STOP_GRADIENT(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("StopGradient").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaStopGradientOp<TYPE>)

REGISTER_MUSA_STOP_GRADIENT(float);
REGISTER_MUSA_STOP_GRADIENT(double);
REGISTER_MUSA_STOP_GRADIENT(Eigen::half);
REGISTER_MUSA_STOP_GRADIENT(int32);
REGISTER_MUSA_STOP_GRADIENT(int64);
REGISTER_MUSA_STOP_GRADIENT(bfloat16);

#undef REGISTER_MUSA_STOP_GRADIENT

}  // namespace musa
}  // namespace tensorflow
