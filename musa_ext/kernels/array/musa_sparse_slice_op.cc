// musa_sparse_slice_op.cc

#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

using namespace tensorflow;

extern "C" {
// New Fused Launchers
void LaunchGenerateSparseSliceMaskAndCount1D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream);
void LaunchGenerateSparseSliceMaskAndCount2D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream);
void LaunchGenerateSparseSliceMaskAndCount3D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream);
void LaunchGenerateSparseSliceMaskAndCount4D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream);
void LaunchGenerateSparseSliceMaskAndCount5D(const int64_t* indices, const int64_t* start, const int64_t* size, int64_t N, bool* mask, int32_t* count, musaStream_t stream);

// Scan Pipeline
void LaunchFullScanPipeline(
    const int32_t* d_counts,
    int64_t* d_pos,
    int64_t* d_block_sums,
    int64_t* d_block_prefix_sums,
    int64_t N,
    int64_t block_size,
    int64_t num_blocks,
    musaStream_t stream);

void LaunchGetTotalCount(
    const int64_t* d_block_sums,
    int64_t* d_total_count,
    int64_t num_blocks,
    musaStream_t stream);

// Gather Launchers
void LaunchGatherSparseSliceElementsFloat1D(const int64_t* indices, const float* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat2D(const int64_t* indices, const float* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat3D(const int64_t* indices, const float* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat4D(const int64_t* indices, const float* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat5D(const int64_t* indices, const float* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, float* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsDouble1D(const int64_t* indices, const double* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble2D(const int64_t* indices, const double* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble3D(const int64_t* indices, const double* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble4D(const int64_t* indices, const double* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble5D(const int64_t* indices, const double* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, double* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt321D(const int64_t* indices, const int32_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt322D(const int64_t* indices, const int32_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt323D(const int64_t* indices, const int32_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt324D(const int64_t* indices, const int32_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt325D(const int64_t* indices, const int32_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int32_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt641D(const int64_t* indices, const int64_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt642D(const int64_t* indices, const int64_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt643D(const int64_t* indices, const int64_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt644D(const int64_t* indices, const int64_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt645D(const int64_t* indices, const int64_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int64_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsUInt81D(const int64_t* indices, const uint8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt82D(const int64_t* indices, const uint8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt83D(const int64_t* indices, const uint8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt84D(const int64_t* indices, const uint8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt85D(const int64_t* indices, const uint8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint8_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt81D(const int64_t* indices, const int8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt82D(const int64_t* indices, const int8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt83D(const int64_t* indices, const int8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt84D(const int64_t* indices, const int8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt85D(const int64_t* indices, const int8_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int8_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt161D(const int64_t* indices, const int16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt162D(const int64_t* indices, const int16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt163D(const int64_t* indices, const int16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt164D(const int64_t* indices, const int16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt165D(const int64_t* indices, const int16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, int16_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsUInt161D(const int64_t* indices, const uint16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt162D(const int64_t* indices, const uint16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt163D(const int64_t* indices, const uint16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt164D(const int64_t* indices, const uint16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt165D(const int64_t* indices, const uint16_t* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, uint16_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsBool1D(const int64_t* indices, const bool* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool2D(const int64_t* indices, const bool* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool3D(const int64_t* indices, const bool* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool4D(const int64_t* indices, const bool* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool5D(const int64_t* indices, const bool* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, bool* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsHalf1D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf2D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf3D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf4D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf5D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsBFloat161D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat162D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat163D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat164D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat165D(const int64_t* indices, const void* values, const int64_t* start, const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices, void* out_values, musaStream_t stream);
}

namespace tensorflow {
namespace musa {

template <typename T>
struct LauncherSelector;

#define DEFINE_LAUNCHER_SELECTOR(T, Prefix)                                   \
  template <>                                                                 \
  struct LauncherSelector<T> {                                                \
    static void Launch1D(const int64_t* indices, const T* values,             \
                         const int64_t* start, const bool* mask,              \
                         const int64_t* pos, int64_t N, int64_t* out_indices, \
                         T* out_values, musaStream_t stream) {                \
      Prefix##1D(indices, values, start, mask, pos, N, out_indices,           \
                 out_values, stream);                                         \
    }                                                                         \
    static void Launch2D(const int64_t* indices, const T* values,             \
                         const int64_t* start, const bool* mask,              \
                         const int64_t* pos, int64_t N, int64_t* out_indices, \
                         T* out_values, musaStream_t stream) {                \
      Prefix##2D(indices, values, start, mask, pos, N, out_indices,           \
                 out_values, stream);                                         \
    }                                                                         \
    static void Launch3D(const int64_t* indices, const T* values,             \
                         const int64_t* start, const bool* mask,              \
                         const int64_t* pos, int64_t N, int64_t* out_indices, \
                         T* out_values, musaStream_t stream) {                \
      Prefix##3D(indices, values, start, mask, pos, N, out_indices,           \
                 out_values, stream);                                         \
    }                                                                         \
    static void Launch4D(const int64_t* indices, const T* values,             \
                         const int64_t* start, const bool* mask,              \
                         const int64_t* pos, int64_t N, int64_t* out_indices, \
                         T* out_values, musaStream_t stream) {                \
      Prefix##4D(indices, values, start, mask, pos, N, out_indices,           \
                 out_values, stream);                                         \
    }                                                                         \
    static void Launch5D(const int64_t* indices, const T* values,             \
                         const int64_t* start, const bool* mask,              \
                         const int64_t* pos, int64_t N, int64_t* out_indices, \
                         T* out_values, musaStream_t stream) {                \
      Prefix##5D(indices, values, start, mask, pos, N, out_indices,           \
                 out_values, stream);                                         \
    }                                                                         \
  };

DEFINE_LAUNCHER_SELECTOR(float, LaunchGatherSparseSliceElementsFloat)
DEFINE_LAUNCHER_SELECTOR(double, LaunchGatherSparseSliceElementsDouble)
DEFINE_LAUNCHER_SELECTOR(int32_t, LaunchGatherSparseSliceElementsInt32)
DEFINE_LAUNCHER_SELECTOR(int64_t, LaunchGatherSparseSliceElementsInt64)
DEFINE_LAUNCHER_SELECTOR(uint8_t, LaunchGatherSparseSliceElementsUInt8)
DEFINE_LAUNCHER_SELECTOR(uint16_t, LaunchGatherSparseSliceElementsUInt16)
DEFINE_LAUNCHER_SELECTOR(int8_t, LaunchGatherSparseSliceElementsInt8)
DEFINE_LAUNCHER_SELECTOR(int16_t, LaunchGatherSparseSliceElementsInt16)
DEFINE_LAUNCHER_SELECTOR(bool, LaunchGatherSparseSliceElementsBool)

template <>
struct LauncherSelector<Eigen::half> {
  static void Launch1D(const int64_t* indices, const Eigen::half* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       Eigen::half* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsHalf1D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch2D(const int64_t* indices, const Eigen::half* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       Eigen::half* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsHalf2D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch3D(const int64_t* indices, const Eigen::half* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       Eigen::half* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsHalf3D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch4D(const int64_t* indices, const Eigen::half* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       Eigen::half* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsHalf4D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch5D(const int64_t* indices, const Eigen::half* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       Eigen::half* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsHalf5D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
};

template <>
struct LauncherSelector<bfloat16> {
  static void Launch1D(const int64_t* indices, const bfloat16* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       bfloat16* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsBFloat161D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch2D(const int64_t* indices, const bfloat16* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       bfloat16* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsBFloat162D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch3D(const int64_t* indices, const bfloat16* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       bfloat16* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsBFloat163D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch4D(const int64_t* indices, const bfloat16* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       bfloat16* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsBFloat164D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
  static void Launch5D(const int64_t* indices, const bfloat16* values,
                       const int64_t* start, const bool* mask,
                       const int64_t* pos, int64_t N, int64_t* out_indices,
                       bfloat16* out_values, musaStream_t stream) {
    LaunchGatherSparseSliceElementsBFloat165D(
        indices, reinterpret_cast<const void*>(values), start, mask, pos, N,
        out_indices, reinterpret_cast<void*>(out_values), stream);
  }
};

#undef DEFINE_LAUNCHER_SELECTOR

template <typename T>
class MusaSparseSliceOp : public OpKernel {
 public:
  explicit MusaSparseSliceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& dense_shape = context->input(2);
    const Tensor& start = context->input(3);
    const Tensor& size = context->input(4);

    const int64_t N = indices.dim_size(0);
    const int ndims = indices.dim_size(1);

    OP_REQUIRES(context, ndims >= 1 && ndims <= 5,
                errors::Unimplemented(
                    "MusaSparseSlice only supports 1-5D sparse tensors, got ",
                    ndims, "D"));
    OP_REQUIRES(context, values.dim_size(0) == N,
                errors::InvalidArgument("Values size must match indices size: ",
                                        values.dim_size(0), " vs ", N));
    OP_REQUIRES(
        context, dense_shape.dim_size(0) == ndims,
        errors::InvalidArgument("Dense shape size must match ndims: ",
                                dense_shape.dim_size(0), " vs ", ndims));
    OP_REQUIRES(context, start.dim_size(0) == ndims,
                errors::InvalidArgument("Start size must match ndims: ",
                                        start.dim_size(0), " vs ", ndims));
    OP_REQUIRES(context, size.dim_size(0) == ndims,
                errors::InvalidArgument("Size size must match ndims: ",
                                        size.dim_size(0), " vs ", ndims));

    if (N == 0) {
      Tensor* out_indices = nullptr;
      Tensor* out_values = nullptr;
      Tensor* out_dense_shape = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  0, TensorShape({0, ndims}), &out_indices));
      OP_REQUIRES_OK(
          context, context->allocate_output(1, TensorShape({0}), &out_values));
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({ndims}),
                                                       &out_dense_shape));
      for (int d = 0; d < ndims; ++d) {
        out_dense_shape->flat<int64_t>()(d) = size.flat<int64_t>()(d);
      }
      return;
    }

    musaStream_t raw_stream = GetMusaStreamByCtx(context);
    
    // --- Memory Allocation ---
    Tensor d_mask_tensor;
    Tensor d_count_tensor;
    Tensor d_pos_tensor;
    Tensor d_block_sums_tensor;
    Tensor d_block_prefix_sums_tensor;
    Tensor d_total_count_tensor;

    OP_REQUIRES_OK(context, context->allocate_temp(DT_BOOL, TensorShape({N}),
                                                   &d_mask_tensor));
    bool* d_mask = d_mask_tensor.flat<bool>().data();

    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({N}),
                                                   &d_count_tensor));
    int32_t* d_count = d_count_tensor.flat<int32_t>().data();

    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({N}),
                                                   &d_pos_tensor));
    int64_t* d_pos = d_pos_tensor.flat<int64_t>().data();

    const int64_t block_size = 256;
    const int64_t num_blocks = (N + block_size - 1) / block_size;

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_blocks}),
                                          &d_block_sums_tensor));
    int64_t* d_block_sums = d_block_sums_tensor.flat<int64_t>().data();

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_blocks}),
                                          &d_block_prefix_sums_tensor));
    int64_t* d_block_prefix_sums =
        d_block_prefix_sums_tensor.flat<int64_t>().data();
        
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({1}),
                                          &d_total_count_tensor));
    int64_t* d_total_count = d_total_count_tensor.flat<int64_t>().data();

    // --- Execution Pipeline ---

    // 1. Fused Mask Generation and Counting
    if (ndims == 1) {
      LaunchGenerateSparseSliceMaskAndCount1D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, d_count, raw_stream);
    } else if (ndims == 2) {
      LaunchGenerateSparseSliceMaskAndCount2D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, d_count, raw_stream);
    } else if (ndims == 3) {
      LaunchGenerateSparseSliceMaskAndCount3D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, d_count, raw_stream);
    } else if (ndims == 4) {
      LaunchGenerateSparseSliceMaskAndCount4D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, d_count, raw_stream);
    } else if (ndims == 5) {
      LaunchGenerateSparseSliceMaskAndCount5D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, d_count, raw_stream);
    }

    // 2. Parallel Exclusive Scan (Pos Calculation)
    LaunchFullScanPipeline(d_count, d_pos, d_block_sums, d_block_prefix_sums,
                           N, block_size, num_blocks, raw_stream);

    // 3. Calculate Total Count (on Device)
    LaunchGetTotalCount(d_block_sums, d_total_count, num_blocks, raw_stream);

    // 4. Copy Total Count to Host to allocate output
    int64_t out_N = 0;
    musaMemcpyAsync(&out_N, d_total_count, sizeof(int64_t),
                    musaMemcpyDeviceToHost, raw_stream);
    musaStreamSynchronize(raw_stream);

    // 5. Allocate Outputs
    Tensor* out_indices = nullptr;
    Tensor* out_values = nullptr;
    Tensor* out_dense_shape = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({out_N, ndims}), &out_indices));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({out_N}),
                                                     &out_values));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({ndims}),
                                                     &out_dense_shape));

    for (int d = 0; d < ndims; ++d) {
      out_dense_shape->flat<int64_t>()(d) = size.flat<int64_t>()(d);
    }

    // 6. Gather Elements (if any)
    if (out_N > 0) {
      if (ndims == 1) {
        LauncherSelector<T>::Launch1D(
            indices.flat<int64_t>().data(), values.flat<T>().data(),
            start.flat<int64_t>().data(), d_mask, d_pos, N,
            out_indices->flat<int64_t>().data(), out_values->flat<T>().data(),
            raw_stream);
      } else if (ndims == 2) {
        LauncherSelector<T>::Launch2D(
            indices.flat<int64_t>().data(), values.flat<T>().data(),
            start.flat<int64_t>().data(), d_mask, d_pos, N,
            out_indices->flat<int64_t>().data(), out_values->flat<T>().data(),
            raw_stream);
      } else if (ndims == 3) {
        LauncherSelector<T>::Launch3D(
            indices.flat<int64_t>().data(), values.flat<T>().data(),
            start.flat<int64_t>().data(), d_mask, d_pos, N,
            out_indices->flat<int64_t>().data(), out_values->flat<T>().data(),
            raw_stream);
      } else if (ndims == 4) {
        LauncherSelector<T>::Launch4D(
            indices.flat<int64_t>().data(), values.flat<T>().data(),
            start.flat<int64_t>().data(), d_mask, d_pos, N,
            out_indices->flat<int64_t>().data(), out_values->flat<T>().data(),
            raw_stream);
      } else if (ndims == 5) {
        LauncherSelector<T>::Launch5D(
            indices.flat<int64_t>().data(), values.flat<T>().data(),
            start.flat<int64_t>().data(), d_mask, d_pos, N,
            out_indices->flat<int64_t>().data(), out_values->flat<T>().data(),
            raw_stream);
      }
    }

    musaError_t err = musaGetLastError();
    OP_REQUIRES(context, err == musaSuccess,
                errors::Internal("MUSA SparseSlice kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

// ----------------------------------------------------------------------------
// 算子注册
// ----------------------------------------------------------------------------
#define REGISTER_MUSA_SPARSE_SLICE_KERNEL(T)                     \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SparseSlice").Device("MUSA").TypeConstraint<T>("T"), \
      MusaSparseSliceOp<T>);

REGISTER_MUSA_SPARSE_SLICE_KERNEL(float);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(double);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(int32_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(int64_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(uint8_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(int16_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(int8_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(uint16_t);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(bool);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(Eigen::half);
REGISTER_MUSA_SPARSE_SLICE_KERNEL(bfloat16);

#undef REGISTER_MUSA_SPARSE_SLICE_KERNEL
}  // namespace musa
}  // namespace tensorflow