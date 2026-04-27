#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaAbsOp : public MusaOpKernel {
 public:
  explicit MusaAbsOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Abs is element-wise - lightweight
  bool IsExpensive() override { return false; }

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

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mUnary unary_op;
    unary_op.SetMode(::musa::dnn::Unary::Mode::ABS);

    mTensor mt_input = CreateMTensor(input, format_);
    mTensor mt_output = CreateMTensor(*output, format_);

    auto status = unary_op.Run(handle, mt_output, mt_input);

    OP_REQUIRES(
        ctx, status == ::musa::dnn::Status::SUCCESS,
        errors::Internal("MUSA Abs execution failed. Status: ", (int)status));
  }
};

#define REGISTER_MUSA_ABS(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Abs").Device("MUSA").TypeConstraint<TYPE>("T"), MusaAbsOp<TYPE>)

REGISTER_MUSA_ABS(float);
REGISTER_MUSA_ABS(Eigen::half);
REGISTER_MUSA_ABS(bfloat16);
REGISTER_MUSA_ABS(double);
REGISTER_MUSA_ABS(int32);
REGISTER_MUSA_ABS(int64);

#undef REGISTER_MUSA_ABS

}  // namespace musa
}  // namespace tensorflow
