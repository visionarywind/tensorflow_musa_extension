#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "../utils_op.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
void MusaInvertPermutationKernelLauncher(const void* perm, void* inv_perm,
                                         int64_t n, musaStream_t stream);

template <typename T>
class MusaInvertPermutationOp : public MusaOpKernel {
 public:
  explicit MusaInvertPermutationOp(OpKernelConstruction* ctx)
      : MusaOpKernel(ctx) {}

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
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input.shape()),
                errors::InvalidArgument("input must be a vector"));
    const int64_t n = input.NumElements();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    if (n == 0) return;

    auto& handle = GetHandleByCtx(ctx);
    musaStream_t stream = reinterpret_cast<musaStream_t>(handle.GetStream());

    MusaInvertPermutationKernelLauncher<T>(
        input.tensor_data().data(),
        const_cast<char*>(output->tensor_data().data()), n, stream);
  }
};

#define REGISTER_MUSA_INVERT_PERMUTATION(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("InvertPermutation").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaInvertPermutationOp<TYPE>)

REGISTER_MUSA_INVERT_PERMUTATION(int32);
REGISTER_MUSA_INVERT_PERMUTATION(int64);

#undef REGISTER_MUSA_INVERT_PERMUTATION

}  // namespace musa
}  // namespace tensorflow
