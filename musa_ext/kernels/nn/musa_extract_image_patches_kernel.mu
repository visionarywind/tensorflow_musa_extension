// High-Performance MUSA ExtractImagePatches Kernels
// Optimized for memory coalescing and NHWC tensor layout
// 完全对齐TensorFlow原生ExtractImagePatches算子逻辑，适配MUSA GPU架构
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

// MUSA运行时头文件，对应CUDA的cuda_runtime.h
#include <musa_runtime.h>
// 半精度类型支持
#include <musa_fp16.h>
#include <musa_bf16.h>
// 整数类型定义
#include <stdint.h>

// ============================================================================
// 核心优化：ExtractImagePatches 主核函数
// 设计思路：每个线程处理输出张量中的1个元素，保证全局内存合并访问，最大化显存带宽利用率
// 模板参数：T=输入输出数据类型，IndexT=索引类型（int32/int64）
// ============================================================================
template <typename T, typename IndexT>
__global__ void ExtractImagePatchesKernel(
    // __restrict__ 关键字：告诉编译器指针无内存别名，解锁最大程度的编译器优化
    const T* __restrict__ images,    // 输入图像张量，设备端指针
    T* __restrict__ patches,         // 输出patch张量，设备端指针
    // 输入张量维度信息
    const int64_t batch_size,        // 批次大小，输入第0维
    const int64_t in_h,              // 输入图像高度，输入第1维
    const int64_t in_w,              // 输入图像宽度，输入第2维
    const int64_t in_c,              // 输入图像通道数，输入第3维
    // 输出张量维度信息
    const int64_t out_h,             // 输出高度，滑动窗口后的H维度大小
    const int64_t out_w,             // 输出宽度，滑动窗口后的W维度大小
    // 滑动窗口核心参数
    const int64_t kH,                // 滑动窗口高度
    const int64_t kW,                // 滑动窗口宽度
    const int64_t stride_h,          // H维度滑动步长
    const int64_t stride_w,          // W维度滑动步长
    const int64_t rate_h,            // H维度膨胀率（空洞采样间隔）
    const int64_t rate_w,            // W维度膨胀率
    // Padding参数
    const int64_t pad_top,           // H维度顶部填充量
    const int64_t pad_left) {        // W维度左侧填充量

    // -------------------------- 1. 线程全局ID与越界保护 --------------------------
    // 计算每个patch展平后的深度：kH*kW*in_c，对应输出张量的最后一维
    const int64_t patch_depth = kH * kW * in_c;
    // 输出张量总元素数：batch * out_h * out_w * patch_depth
    const int64_t total_elements = batch_size * out_h * out_w * patch_depth;
    // 全局线程ID：block内线程ID + block编号*block大小
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界线程直接退出，避免非法内存访问
    if (tid >= total_elements) return;

    // -------------------------- 2. 全局ID分解为多维坐标 --------------------------
    // 输出张量布局：[batch, out_h, out_w, patch_depth]
    // 分解逻辑：从最低维到最高维，保证内存访问连续，符合NHWC的内存布局
    // 步骤1：分解出patch内的索引（对应输出最后一维）
    const int64_t patch_idx = tid % patch_depth;
    // 步骤2：分解出输出的x坐标（W维度）
    int64_t temp = tid / patch_depth;
    const int64_t out_x = temp % out_w;
    // 步骤3：分解出输出的y坐标（H维度）
    temp /= out_w;
    const int64_t out_y = temp % out_h;
    // 步骤4：分解出批次索引（batch维度）
    temp /= out_h;
    const int64_t batch_idx = temp;

    // -------------------------- 3. Patch内索引分解为核内坐标 --------------------------
    // TF原生展平顺序：ky(核y) → kx(核x) → c(通道)，必须严格对齐，否则输出顺序错误
    // 步骤1：分解出通道索引
    const int64_t c = patch_idx % in_c;
    // 步骤2：分解出核内x坐标
    int64_t temp_patch = patch_idx / in_c;
    const int64_t kx = temp_patch % kW;
    // 步骤3：分解出核内y坐标
    const int64_t ky = temp_patch / kW;

    // -------------------------- 4. 输入图像坐标映射 --------------------------
    // 计算当前patch在输入图像上的对应坐标，包含膨胀率和padding
    const int64_t in_y = out_y * stride_h - pad_top + ky * rate_h;
    const int64_t in_x = out_x * stride_w - pad_left + kx * rate_w;

    // -------------------------- 5. 边界检查与数据赋值 --------------------------
    // 检查坐标是否在输入图像范围内，超出范围填0，与TF原生SAME padding逻辑完全对齐
    const bool in_bounds = (in_y >= 0 && in_y < in_h) && (in_x >= 0 && in_x < in_w);
    
    // 计算输入张量的内存偏移：NHWC布局，内存顺序为 batch → H → W → C
    const int64_t src_offset = batch_idx * in_h * in_w * in_c 
                              + in_y * in_w * in_c 
                              + in_x * in_c 
                              + c;
    // 输出张量的内存偏移：tid就是连续的内存地址，因为我们按输出内存顺序分解tid
    const int64_t dst_offset = tid;

    // 赋值：范围内取输入值，范围外填0
    patches[dst_offset] = in_bounds ? images[src_offset] : T(0);
}

// ============================================================================
// 核函数启动器（Launcher）
// 作用：封装核函数启动逻辑，计算grid/block大小，对外暴露C接口，供OpKernel调用
// ============================================================================
extern "C" {

// 最优线程配置：MUSA GPU的SM架构最优线程数为256（32*8，刚好8个warp）
#define OPTIMAL_THREADS 256
// 向上取整计算grid大小，保证线程数覆盖所有元素
#define OPTIMAL_BLOCKS(count) (((count) + OPTIMAL_THREADS - 1) / OPTIMAL_THREADS)

// ----------------------------------------------------------------------------
// 启动器模板宏：批量生成不同数据类型+索引类型的启动器，减少代码冗余
// 模板参数：T=数据类型，IndexT=索引类型，Name=对外暴露的函数名
// ----------------------------------------------------------------------------
#define DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(T, IndexT, Name) \
  void Name( \
      const T* images, \
      T* patches, \
      int64_t batch_size, \
      int64_t in_h, \
      int64_t in_w, \
      int64_t in_c, \
      int64_t out_h, \
      int64_t out_w, \
      int64_t kH, \
      int64_t kW, \
      int64_t stride_h, \
      int64_t stride_w, \
      int64_t rate_h, \
      int64_t rate_w, \
      int64_t pad_top, \
      int64_t pad_left, \
      musaStream_t stream) { \
    const int64_t patch_depth = kH * kW * in_c; \
    const int64_t total_elements = batch_size * out_h * out_w * patch_depth; \
    if (total_elements == 0) return; \
    const int blocks = OPTIMAL_BLOCKS(total_elements); \
    ExtractImagePatchesKernel<T, IndexT><<<blocks, OPTIMAL_THREADS, 0, stream>>>( \
        images, patches, batch_size, in_h, in_w, in_c, out_h, out_w, \
        kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left); \
  }

// ----------------------------------------------------------------------------
// 基础数据类型启动器定义
// 覆盖TF原生支持的所有基础数值类型，int32/int64索引分别实现
// ----------------------------------------------------------------------------
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(float, int, LaunchExtractImagePatchesFloatInt32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(float, int64_t, LaunchExtractImagePatchesFloatInt64)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(double, int, LaunchExtractImagePatchesDoubleInt32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(double, int64_t, LaunchExtractImagePatchesDoubleInt64)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(int32_t, int, LaunchExtractImagePatchesInt32Int32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(int32_t, int64_t, LaunchExtractImagePatchesInt32Int64)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(int64_t, int, LaunchExtractImagePatchesInt64Int32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(int64_t, int64_t, LaunchExtractImagePatchesInt64Int64)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(uint8_t, int, LaunchExtractImagePatchesUInt8Int32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(uint8_t, int64_t, LaunchExtractImagePatchesUInt8Int64)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(bool, int, LaunchExtractImagePatchesBoolInt32)
DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER(bool, int64_t, LaunchExtractImagePatchesBoolInt64)

// ----------------------------------------------------------------------------
// FP16半精度启动器（Eigen::half）
// 单独实现：需要显式类型强转，适配MUSA的half类型
// ----------------------------------------------------------------------------
void LaunchExtractImagePatchesHalfInt32(
    const void* images,
    void* patches,
    int64_t batch_size,
    int64_t in_h,
    int64_t in_w,
    int64_t in_c,
    int64_t out_h,
    int64_t out_w,
    int64_t kH,
    int64_t kW,
    int64_t stride_h,
    int64_t stride_w,
    int64_t rate_h,
    int64_t rate_w,
    int64_t pad_top,
    int64_t pad_left,
    musaStream_t stream) {

    const int64_t patch_depth = kH * kW * in_c;
    const int64_t total_elements = batch_size * out_h * out_w * patch_depth;
    if (total_elements == 0) return;
    const int blocks = OPTIMAL_BLOCKS(total_elements);

    // 强转为MUSA原生half类型，与TF的Eigen::half内存布局完全兼容
    ExtractImagePatchesKernel<half, int><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(images),
        reinterpret_cast<half*>(patches),
        batch_size, in_h, in_w, in_c, out_h, out_w,
        kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left);
}

void LaunchExtractImagePatchesHalfInt64(
    const void* images,
    void* patches,
    int64_t batch_size,
    int64_t in_h,
    int64_t in_w,
    int64_t in_c,
    int64_t out_h,
    int64_t out_w,
    int64_t kH,
    int64_t kW,
    int64_t stride_h,
    int64_t stride_w,
    int64_t rate_h,
    int64_t rate_w,
    int64_t pad_top,
    int64_t pad_left,
    musaStream_t stream) {

    const int64_t patch_depth = kH * kW * in_c;
    const int64_t total_elements = batch_size * out_h * out_w * patch_depth;
    if (total_elements == 0) return;
    const int blocks = OPTIMAL_BLOCKS(total_elements);

    ExtractImagePatchesKernel<half, int64_t><<<blocks, OPTIMAL_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(images),
        reinterpret_cast<half*>(patches),
        batch_size, in_h, in_w, in_c, out_h, out_w,
        kH, kW, stride_h, stride_w, rate_h, rate_w, pad_top, pad_left);
}

// 清理宏定义，避免命名污染
#undef DEFINE_EXTRACT_IMAGE_PATCHES_LAUNCHER
#undef OPTIMAL_THREADS
#undef OPTIMAL_BLOCKS

}  // extern "C" 结束：对外暴露纯C接口，避免C++名称修饰