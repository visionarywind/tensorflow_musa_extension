#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaRoundOp : public MusaOpKernel {
 public:
  explicit MusaRoundOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    LOG(ERROR) << "[MUSA Debug] Thread: " << std::this_thread::get_id() 
              << " | Op: " << __FILE__ 
              << " | Method: " << __FUNCTION__;
  }

    const Tensor& input = ctx->input(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));

    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    mTensor t_input = CreateMTensor(input, format_);
    mTensor t_output = CreateMTensor(*output, format_);

    ::musa::dnn::Unary op;
    op.SetMode(::musa::dnn::Unary::Mode::ROUND);

    auto status = op.Run(handle, t_output, t_input);
    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Round execution failed."));
  }

};  // class MusaRoundOp


#define REGISTER_MUSA_ROUND(TYPE) \
  REGISTER_KERNEL_BUILDER(      \
      Name("Round").Device("MUSA").TypeConstraint<TYPE>("T"), MusaRoundOp<TYPE>);

REGISTER_MUSA_ROUND(float);
REGISTER_MUSA_ROUND(double);
REGISTER_MUSA_ROUND(Eigen::half);
REGISTER_MUSA_ROUND(bfloat16);
// REGISTER_MUSA_ROUND(int32);
// REGISTER_MUSA_ROUND(int64);

}  // namespace musa
}  // namespace tensorflow