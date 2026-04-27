#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/bcast.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaGreaterOp : public MusaOpKernel {
 public:
  explicit MusaGreaterOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  // Greater is element-wise - lightweight
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

    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);

    BCast bcast(BCast::FromShape(x.shape()), BCast::FromShape(y.shape()));
    OP_REQUIRES(ctx, bcast.IsValid(),
                errors::InvalidArgument("Incompatible shapes for Greater: ",
                                        x.shape().DebugString(), " and ",
                                        y.shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            0, BCast::ToShape(bcast.output_shape()), &output));

    if (output->NumElements() == 0) return;

    auto& handle = GetHandleByCtx(ctx);

    mTensor mt_x = CreateMTensor(x);
    mTensor mt_y = CreateMTensor(y);
    mTensor mt_out = CreateMTensor(*output);

    mBinary op;
    op.SetMode(::musa::dnn::Binary::Mode::GT);

    auto status = op.Run(handle, mt_out, mt_x, mt_y);

    OP_REQUIRES(ctx, status == ::musa::dnn::Status::SUCCESS,
                errors::Internal("MUSA Greater execution failed. Status: ",
                                 (int)status));
  }
};

#define REGISTER_MUSA_GREATER(TYPE)                             \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Greater").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaGreaterOp<TYPE>)

REGISTER_MUSA_GREATER(float);
REGISTER_MUSA_GREATER(Eigen::half);
REGISTER_MUSA_GREATER(int32);
REGISTER_MUSA_GREATER(int64);

#undef REGISTER_MUSA_GREATER

}  // namespace musa
}  // namespace tensorflow
