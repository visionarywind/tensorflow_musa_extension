#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

class MusaLogicalNotOp : public MusaOpKernel {
 public:
  explicit MusaLogicalNotOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

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
    OP_REQUIRES(ctx, input.dtype() == DT_BOOL,
                errors::InvalidArgument("LogicalNot expects bool input, got ",
                                        DataTypeString(input.dtype())));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (input.NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    auto in_mt = CreateMTensor(input, format_);
    auto out_mt = CreateMTensor(*output, format_);

    // NOT(x) = XOR(x, True)
    Tensor true_tensor;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_BOOL, TensorShape({}), &true_tensor));
    true_tensor.scalar<bool>()() = true;
    auto true_mt = CreateMTensor(true_tensor, format_);

    ::musa::dnn::Binary op;
    MTOP_CHECK_OK(op.SetMode(::musa::dnn::Binary::Mode::LOGICAL_XOR),
                  "Set LOGICAL_XOR", ctx);
    MTOP_CHECK_OK_RUN(op.Run(handle, out_mt, in_mt, true_mt),
                      "LogicalNot via XOR Run", ctx);
  }
};

REGISTER_KERNEL_BUILDER(Name("LogicalNot").Device("MUSA"), MusaLogicalNotOp);

}  // namespace musa
}  // namespace tensorflow
