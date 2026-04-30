#include <vector>

#include "../utils_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "utils/logging.h"

using namespace tensorflow;

// ----------------------------------------------------------------------------
// 核函数启动器声明（完整无省略）
// ----------------------------------------------------------------------------
extern "C" {
// 掩码生成启动器
void LaunchGenerateSparseSliceMask1D(const int64_t* indices,
                                     const int64_t* start, const int64_t* size,
                                     int64_t N, bool* mask,
                                     musaStream_t stream);
void LaunchGenerateSparseSliceMask2D(const int64_t* indices,
                                     const int64_t* start, const int64_t* size,
                                     int64_t N, bool* mask,
                                     musaStream_t stream);
void LaunchGenerateSparseSliceMask3D(const int64_t* indices,
                                     const int64_t* start, const int64_t* size,
                                     int64_t N, bool* mask,
                                     musaStream_t stream);
void LaunchGenerateSparseSliceMask4D(const int64_t* indices,
                                     const int64_t* start, const int64_t* size,
                                     int64_t N, bool* mask,
                                     musaStream_t stream);
void LaunchGenerateSparseSliceMask5D(const int64_t* indices,
                                     const int64_t* start, const int64_t* size,
                                     int64_t N, bool* mask,
                                     musaStream_t stream);

// 元素收集启动器 - 基础类型
void LaunchGatherSparseSliceElementsFloat1D(
    const int64_t* indices, const float* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat2D(
    const int64_t* indices, const float* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat3D(
    const int64_t* indices, const float* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat4D(
    const int64_t* indices, const float* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    float* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsFloat5D(
    const int64_t* indices, const float* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    float* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsDouble1D(
    const int64_t* indices, const double* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble2D(
    const int64_t* indices, const double* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble3D(
    const int64_t* indices, const double* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble4D(
    const int64_t* indices, const double* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    double* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsDouble5D(
    const int64_t* indices, const double* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    double* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt321D(
    const int64_t* indices, const int32_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt322D(
    const int64_t* indices, const int32_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt323D(
    const int64_t* indices, const int32_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt324D(
    const int64_t* indices, const int32_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int32_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt325D(
    const int64_t* indices, const int32_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int32_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsInt641D(
    const int64_t* indices, const int64_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt642D(
    const int64_t* indices, const int64_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt643D(
    const int64_t* indices, const int64_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt644D(
    const int64_t* indices, const int64_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int64_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt645D(
    const int64_t* indices, const int64_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int64_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsUInt81D(
    const int64_t* indices, const uint8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt82D(
    const int64_t* indices, const uint8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt83D(
    const int64_t* indices, const uint8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt84D(
    const int64_t* indices, const uint8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt85D(
    const int64_t* indices, const uint8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint8_t* out_values, musaStream_t stream);

    // 元素收集启动器 - Int8
void LaunchGatherSparseSliceElementsInt81D(
    const int64_t* indices, const int8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt82D(
    const int64_t* indices, const int8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt83D(
    const int64_t* indices, const int8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt84D(
    const int64_t* indices, const int8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int8_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt85D(
    const int64_t* indices, const int8_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int8_t* out_values, musaStream_t stream);

// 元素收集启动器 - Int16
void LaunchGatherSparseSliceElementsInt161D(
    const int64_t* indices, const int16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt162D(
    const int64_t* indices, const int16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt163D(
    const int64_t* indices, const int16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt164D(
    const int64_t* indices, const int16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsInt165D(
    const int64_t* indices, const int16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    int16_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsUInt161D(
    const int64_t* indices, const uint16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt162D(
    const int64_t* indices, const uint16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt163D(
    const int64_t* indices, const uint16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt164D(
    const int64_t* indices, const uint16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint16_t* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsUInt165D(
    const int64_t* indices, const uint16_t* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    uint16_t* out_values, musaStream_t stream);

void LaunchGatherSparseSliceElementsBool1D(
    const int64_t* indices, const bool* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool2D(
    const int64_t* indices, const bool* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool3D(
    const int64_t* indices, const bool* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool4D(
    const int64_t* indices, const bool* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    bool* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBool5D(
    const int64_t* indices, const bool* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    bool* out_values, musaStream_t stream);

// 元素收集启动器 - FP16
void LaunchGatherSparseSliceElementsHalf1D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf2D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf3D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf4D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsHalf5D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);

// 元素收集启动器 - BF16
void LaunchGatherSparseSliceElementsBFloat161D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat162D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat163D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat164D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);
void LaunchGatherSparseSliceElementsBFloat165D(
    const int64_t* indices, const void* values, const int64_t* start,
    const bool* mask, const int64_t* pos, int64_t N, int64_t* out_indices,
    void* out_values, musaStream_t stream);

// 辅助核函数启动器
void LaunchMaskToCount(const bool* mask, int32_t* count, int64_t N,
                       musaStream_t stream);
void LaunchBlockScan(const int32_t* count, int64_t* pos, int64_t* block_sums,
                     int64_t N, int64_t block_size, musaStream_t stream);
void LaunchAddBlockPrefixSum(int64_t* pos, const int64_t* block_prefix_sums,
                             int64_t N, int64_t block_size,
                             musaStream_t stream);
}

namespace tensorflow {
namespace musa {

// ----------------------------------------------------------------------------
// 类型分发辅助模板
// ----------------------------------------------------------------------------
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

// 基础类型
DEFINE_LAUNCHER_SELECTOR(float, LaunchGatherSparseSliceElementsFloat)
DEFINE_LAUNCHER_SELECTOR(double, LaunchGatherSparseSliceElementsDouble)
DEFINE_LAUNCHER_SELECTOR(int32_t, LaunchGatherSparseSliceElementsInt32)
DEFINE_LAUNCHER_SELECTOR(int64_t, LaunchGatherSparseSliceElementsInt64)
DEFINE_LAUNCHER_SELECTOR(uint8_t, LaunchGatherSparseSliceElementsUInt8)
DEFINE_LAUNCHER_SELECTOR(uint16_t, LaunchGatherSparseSliceElementsUInt16)
DEFINE_LAUNCHER_SELECTOR(int8_t, LaunchGatherSparseSliceElementsInt8)   // 新增
DEFINE_LAUNCHER_SELECTOR(int16_t, LaunchGatherSparseSliceElementsInt16) // 新增
DEFINE_LAUNCHER_SELECTOR(bool, LaunchGatherSparseSliceElementsBool)

// FP16 特化
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

// BF16 特化
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

// ----------------------------------------------------------------------------
// MusaSparseSliceOp 主类（无 Thrust，无 __global__ 核函数）
// ----------------------------------------------------------------------------
template <typename T>
class MusaSparseSliceOp : public OpKernel {
 public:
  explicit MusaSparseSliceOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 1. 获取输入张量
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& dense_shape = context->input(2);
    const Tensor& start = context->input(3);
    const Tensor& size = context->input(4);

    // 2. 解析输入维度
    const int64_t N = indices.dim_size(0);
    const int ndims = indices.dim_size(1);

    // 3. 基础校验
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

    // 空输入直接返回空输出
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

    // 4. 获取MUSA流
    musaStream_t raw_stream = GetMusaStreamByCtx(context);

    // 5. 分配设备端临时内存 (使用 allocate_temp 替代 musaMalloc)
    Tensor d_mask_tensor;
    Tensor d_count_tensor;
    Tensor d_pos_tensor;
    Tensor d_block_sums_tensor;
    Tensor d_block_prefix_sums_tensor;

    // 分配 mask: N * bool
    OP_REQUIRES_OK(context, context->allocate_temp(DT_BOOL, TensorShape({N}),
                                                   &d_mask_tensor));
    bool* d_mask = d_mask_tensor.flat<bool>().data();

    // 分配 count: N * int32
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, TensorShape({N}),
                                                   &d_count_tensor));
    int32_t* d_count = d_count_tensor.flat<int32_t>().data();

    // 分配 pos: N * int64
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({N}),
                                                   &d_pos_tensor));
    int64_t* d_pos = d_pos_tensor.flat<int64_t>().data();

    // 6. 调用掩码生成核函数
    if (ndims == 1) {
      LaunchGenerateSparseSliceMask1D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, raw_stream);
    } else if (ndims == 2) {
      LaunchGenerateSparseSliceMask2D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, raw_stream);
    } else if (ndims == 3) {
      LaunchGenerateSparseSliceMask3D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, raw_stream);
    } else if (ndims == 4) {
      LaunchGenerateSparseSliceMask4D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, raw_stream);
    } else if (ndims == 5) {
      LaunchGenerateSparseSliceMask5D(
          indices.flat<int64_t>().data(), start.flat<int64_t>().data(),
          size.flat<int64_t>().data(), N, d_mask, raw_stream);
    }

    // 7. 纯 MUSA 原生流程替代 Thrust
    const int64_t block_size = 1024;
    const int64_t num_blocks = (N + block_size - 1) / block_size;

    // Step 7.1: Mask 转 Count
    LaunchMaskToCount(d_mask, d_count, N, raw_stream);

    // Step 7.2: 分块扫描（第一阶段）
    // 分配 block_sums: num_blocks * int64
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_blocks}),
                                          &d_block_sums_tensor));
    int64_t* d_block_sums = d_block_sums_tensor.flat<int64_t>().data();

    LaunchBlockScan(d_count, d_pos, d_block_sums, N, block_size, raw_stream);

    // Step 7.3: 块总和扫描（简化版：拷回主机做）
    // 注意：这里仍然需要 Host 内存进行串行前缀和计算，因为数据量小且依赖性强
    std::vector<int64_t> h_block_sums(num_blocks);
    std::vector<int64_t> h_block_prefix_sums(num_blocks);

    // 同步以确保 d_block_sums 已计算完成
    musaStreamSynchronize(raw_stream);

    musaMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(int64_t),
               musaMemcpyDeviceToHost);

    h_block_prefix_sums[0] = 0;
    for (int64_t i = 1; i < num_blocks; ++i) {
      h_block_prefix_sums[i] = h_block_prefix_sums[i - 1] + h_block_sums[i - 1];
    }

    // Step 7.4: 拷回块前缀和并累加
    // 分配 block_prefix_sums: num_blocks * int64
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_blocks}),
                                          &d_block_prefix_sums_tensor));
    int64_t* d_block_prefix_sums =
        d_block_prefix_sums_tensor.flat<int64_t>().data();

    musaMemcpy(d_block_prefix_sums, h_block_prefix_sums.data(),
               num_blocks * sizeof(int64_t), musaMemcpyHostToDevice);

    LaunchAddBlockPrefixSum(d_pos, d_block_prefix_sums, N, block_size,
                            raw_stream);

    // Step 7.5: 计算 out_N
    // 同步以获取最终结果
    musaStreamSynchronize(raw_stream);

    int64_t out_N = 0;
    int32_t last_count = 0;
    int64_t last_pos = 0;

    // 如果 N > 0，则读取最后一个元素
    if (N > 0) {
      musaMemcpy(&last_count, d_count + N - 1, sizeof(int32_t),
                 musaMemcpyDeviceToHost);
      musaMemcpy(&last_pos, d_pos + N - 1, sizeof(int64_t),
                 musaMemcpyDeviceToHost);
      out_N = last_pos + last_count;
    }

    // 8. 分配输出张量
    Tensor* out_indices = nullptr;
    Tensor* out_values = nullptr;
    Tensor* out_dense_shape = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({out_N, ndims}), &out_indices));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({out_N}),
                                                     &out_values));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({ndims}),
                                                     &out_dense_shape));

    // 9. 拷贝size到out_dense_shape
    for (int d = 0; d < ndims; ++d) {
      out_dense_shape->flat<int64_t>()(d) = size.flat<int64_t>()(d);
    }

    // 10. 调用元素收集核函数
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

    // 11. 释放临时内存
    // 不需要手动 musaFree，Tensor 对象销毁时会自动处理或由 TF 运行时管理

    // 12. 检查错误
    musaError_t err = musaGetLastError();
    OP_REQUIRES(context, err == musaSuccess,
                errors::Internal("MUSA SparseSlice kernel launch failed: ",
                                 musaGetErrorString(err)));
  }
};

// ----------------------------------------------------------------------------
// 算子注册（完整无省略）
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