#include <mudnn.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {
namespace {

template <typename T>
class MusaZerosLikeOp : public MusaOpKernel {
 public:
  explicit MusaZerosLikeOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

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

    if (output->NumElements() == 0) return;

    auto& h = GetHandleByCtx(ctx);
    auto out_mt = CreateMTensor(*output);

    ::musa::dnn::Fill op;

    MTOP_CHECK_OK(op.SetValue(0.0), "Fill SetValue to 0", ctx);

    MTOP_CHECK_OK_RUN(op.Run(h, out_mt), "Fill Run for ZerosLike", ctx);
  }
};

#define REGISTER_MUSA_ZEROS_LIKE(type)                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("ZerosLike").Device(DEVICE_MTGPU).TypeConstraint<type>("T"), \
      MusaZerosLikeOp<type>);

REGISTER_MUSA_ZEROS_LIKE(float);
REGISTER_MUSA_ZEROS_LIKE(Eigen::half);
REGISTER_MUSA_ZEROS_LIKE(double);
REGISTER_MUSA_ZEROS_LIKE(int32);
REGISTER_MUSA_ZEROS_LIKE(int64);
REGISTER_MUSA_ZEROS_LIKE(bool);

#undef REGISTER_MUSA_ZEROS_LIKE

}  // namespace
}  // namespace musa
}  // namespace tensorflow
