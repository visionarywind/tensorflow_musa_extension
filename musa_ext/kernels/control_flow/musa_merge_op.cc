#include "../utils_op.h"
#include "mu/device/musa_device.h"
#include <thread>
#include <tensorflow/core/platform/logging.h>
#include <cstdlib>

namespace tensorflow {
namespace musa {

template <typename T>
class MusaMergeOp : public MusaOpKernel {
 public:
  explicit MusaMergeOp(OpKernelConstruction* context) : MusaOpKernel(context) {}

  void Compute(OpKernelContext* context) override {

  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    std::stringstream ss;
    ss << "[MUSA Debug] Thread: " << std::this_thread::get_id()
              << " | Op: " << __FILE__
              << " | Method: " << __FUNCTION__;
    // int input_num = context->num_inputs();
    // for (int i = 0; i < input_num; ++i) {
    //     ss << " | Input " << i << ": " << context->input(i).shape().DebugString();
    // }
    LOG(ERROR) << ss.str();
  }
  static bool sync_execute = std::getenv("MUSA_LAUNCH_BLOCKING") == "1";
  if (sync_execute) {
    musaStreamSynchronize(GetMusaStreamByCtx(context));
  }

    bool input_seen = false;
    for (int i = 0; i < context->num_inputs(); ++i) {
      if (context->has_input(i)) {
        if (input_seen) {
          LOG(WARNING) << "Merge op has more than one valid input. This "
                       << "indicates that the graph doesn't use merge op "
                       << "properly. Please check your graph. ";
          return;
        }
        input_seen = true;

        if (IsRefType(context->input_dtype(i))) {
          context->forward_ref_input_to_ref_output(i, 0);
        } else {
          context->set_output(0, context->input(i));
        }
        // The value_index output is typically used only in gradient
        // calculations, so we can avoid allocating in many inference workloads.
        if (context->output_required(1)) {
          Tensor* value_index = nullptr;
          OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                           &value_index));
          value_index->scalar<int32_t>()() = i;
        }
      }
    }
  }
};  // class MusaMergeOp

#define REGISTER_MUSA_MERGE(type)                         \
  REGISTER_KERNEL_BUILDER(Name("Merge")                   \
                              .Device(DEVICE_MTGPU)       \
                              .TypeConstraint<type>("T")  \
                              .HostMemory("value_index"), \
                          MusaMergeOp<type>);

REGISTER_MUSA_MERGE(float);
REGISTER_MUSA_MERGE(int32);
REGISTER_MUSA_MERGE(int64);
REGISTER_MUSA_MERGE(Eigen::half);
REGISTER_MUSA_MERGE(bfloat16);
REGISTER_MUSA_MERGE(double);
REGISTER_MUSA_MERGE(uint8);
REGISTER_MUSA_MERGE(bool);

}  // namespace musa
}  // namespace tensorflow
