#include "../utils_op.h"
#include "utils/logging.h"

namespace tensorflow {
namespace musa {
class MusaIdentityNOp : public MusaOpKernel {
 public:
  explicit MusaIdentityNOp(OpKernelConstruction* ctx) : MusaOpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {}
    MUSA_KERNEL_TIMING_GUARD(ctx);
};
}  // namespace musa
}  // namespace tensorflow