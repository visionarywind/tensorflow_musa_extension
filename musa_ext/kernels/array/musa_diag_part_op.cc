#include "../utils_op.h"
#include "tensorflow/core/framework/bfloat16.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
void MusaDiagPartkernelLauncher(musaStream_t stream, uint64_t size, const T* in,
                                T* out);

template <typename T>
class MusaDiagPartOp : public MusaOpKernel {
  /*
    Implementation for DiagPart op, which extracts the diagonal part of a
  tensor.
    The shape of input should be like, considiering a tensor with a dim of 2k:
  [s1, s2, ..., sk, s1, s2, ..., sk]
    Then the output will be a tensor with shape [s1, s2, ..., sk]
  */
 public:
  explicit MusaDiagPartOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

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

    const Tensor& tensor = context->input(0);
    const int num_dims = tensor.dims();
    const int out_dims = num_dims / 2;
    OP_REQUIRES(
        context, 0 == num_dims % 2,
        errors::InvalidArgument("Input must have even number of dimensions"));

    TensorShape out_shape;
    for (int i = 0; i < out_dims; ++i) {
      OP_REQUIRES(
          context, tensor.dim_size(i) == tensor.dim_size(i + out_dims),
          errors::InvalidArgument("Invalid shape ",
                                  tensor.shape().DebugString(), ": dimensions ",
                                  i, " and ", i + out_dims, " do not match."));
      OP_REQUIRES_OK(context, out_shape.AddDimWithStatus(tensor.dim_size(i)));
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Get stream
    auto& handle = GetHandleByCtx(context);
    musaStream_t stream = (musaStream_t)handle.GetStream();

    // Launch kernel
    MusaDiagPartkernelLauncher<T>(stream, output->NumElements(),
                                  tensor.flat<T>().data(),
                                  output->flat<T>().data());
  }
};  // class MusaDiagPartOp

#define REGISTER_MUSA_DIAG_PART(TYPE)                            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DiagPart").Device("MUSA").TypeConstraint<TYPE>("T"), \
      MusaDiagPartOp<TYPE>)

REGISTER_MUSA_DIAG_PART(float);
REGISTER_MUSA_DIAG_PART(double);
REGISTER_MUSA_DIAG_PART(int32);
REGISTER_MUSA_DIAG_PART(int64);
REGISTER_MUSA_DIAG_PART(Eigen::half);
REGISTER_MUSA_DIAG_PART(bfloat16);

}  // namespace musa
}  // namespace tensorflow