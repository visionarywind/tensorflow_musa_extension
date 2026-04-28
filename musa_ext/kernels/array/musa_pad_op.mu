// MUSA Pad Custom Kernel
//
// Copyright 2026 The TensorFlow MUSA Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0.

#include <musa_runtime.h>

#define MAX_DIM 4  // NHWC 4d support only

template <typename T>
__global__ void nd_pad_kernel(const T *__restrict__ in, T *__restrict__ out,
                              const int real_dim, const int64_t *in_dims,
                              const int64_t *out_dims,
                              const int64_t *pad_before,
                              const int64_t *pad_after, const T pad_val,
                              const int64_t total_out_elements) {
  int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= total_out_elements) {
    return;
  }

  int64_t in_stride[MAX_DIM] = {1};
  int64_t out_stride[MAX_DIM] = {1};
  for (int d = real_dim - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_dims[d + 1];
    out_stride[d] = out_stride[d + 1] * out_dims[d + 1];
  }

  int64_t remain = global_idx;
  int64_t in_idx = 0;
  bool is_inside = true;

  for (int d = 0; d < real_dim; ++d) {
    int64_t out_coord = remain / out_stride[d];
    remain = remain % out_stride[d];

    int64_t in_coord = out_coord - pad_before[d];
    if (in_coord < 0 || in_coord >= in_dims[d]) {
      is_inside = false;
      break;
    }

    in_idx += in_coord * in_stride[d];
  }

  if (is_inside) {
    out[global_idx] = in[in_idx];
  } else {
    out[global_idx] = pad_val;
  }
}

template __global__ void nd_pad_kernel<float>(
    const float *__restrict__ in, float *__restrict__ out, const int real_dim,
    const int64_t *in_dims, const int64_t *out_dims, const int64_t *pad_before,
    const int64_t *pad_after, const float pad_val,
    const int64_t total_out_elements);

template __global__ void nd_pad_kernel<double>(
    const double *__restrict__ in, double *__restrict__ out, const int real_dim,
    const int64_t *in_dims, const int64_t *out_dims, const int64_t *pad_before,
    const int64_t *pad_after, const double pad_val,
    const int64_t total_out_elements);

template __global__ void nd_pad_kernel<int32_t>(
    const int32_t *__restrict__ in, int32_t *__restrict__ out,
    const int real_dim, const int64_t *in_dims, const int64_t *out_dims,
    const int64_t *pad_before, const int64_t *pad_after, const int32_t pad_val,
    const int64_t total_out_elements);

template __global__ void nd_pad_kernel<int64_t>(
    const int64_t *__restrict__ in, int64_t *__restrict__ out,
    const int real_dim, const int64_t *in_dims, const int64_t *out_dims,
    const int64_t *pad_before, const int64_t *pad_after, const int64_t pad_val,
    const int64_t total_out_elements);

template <typename T>
void nd_pad_kernel_launcher(const T *input_data, T *output_data, const int dims,
                            const int64_t *in_dims, const int64_t *out_dims,
                            const int64_t *pad_before, const int64_t *pad_after,
                            const T pad_value, const int64_t total_out_elements,
                            const musaStream_t stream) {
  const int block_size = 256;
  int grid_size = (total_out_elements + block_size - 1) / block_size;

  nd_pad_kernel<T><<<grid_size, block_size, 0, stream>>>(
      input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after,
      pad_value, total_out_elements);
}

extern "C" {

// 为每种类型定义具体的启动器函数
// 注意：函数名需要唯一，通常加上类型后缀

void nd_pad_kernel_launcher_float(const float *input_data, float *output_data, 
                                  const int dims, const int64_t *in_dims, 
                                  const int64_t *out_dims, const int64_t *pad_before, 
                                  const int64_t *pad_after, const float pad_value, 
                                  const int64_t total_out_elements, const musaStream_t stream) {
    const int block_size = 256;
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size;
    nd_pad_kernel<float><<<grid_size, block_size, 0, stream>>>(
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, 
        pad_value, total_out_elements);
}

void nd_pad_kernel_launcher_double(const double *input_data, double *output_data, 
                                   const int dims, const int64_t *in_dims, 
                                   const int64_t *out_dims, const int64_t *pad_before, 
                                   const int64_t *pad_after, const double pad_value, 
                                   const int64_t total_out_elements, const musaStream_t stream) {
    const int block_size = 256;
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size;
    nd_pad_kernel<double><<<grid_size, block_size, 0, stream>>>(
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, 
        pad_value, total_out_elements);
}

void nd_pad_kernel_launcher_int32(const int32_t *input_data, int32_t *output_data, 
                                  const int dims, const int64_t *in_dims, 
                                  const int64_t *out_dims, const int64_t *pad_before, 
                                  const int64_t *pad_after, const int32_t pad_value, 
                                  const int64_t total_out_elements, const musaStream_t stream) {
    const int block_size = 256;
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size;
    nd_pad_kernel<int32_t><<<grid_size, block_size, 0, stream>>>(
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, 
        pad_value, total_out_elements);
}

void nd_pad_kernel_launcher_int64(const int64_t *input_data, int64_t *output_data, 
                                  const int dims, const int64_t *in_dims, 
                                  const int64_t *out_dims, const int64_t *pad_before, 
                                  const int64_t *pad_after, const int64_t pad_value, 
                                  const int64_t total_out_elements, const musaStream_t stream) {
    const int block_size = 256;
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size;
    nd_pad_kernel<int64_t><<<grid_size, block_size, 0, stream>>>(
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, 
        pad_value, total_out_elements);
}

void nd_pad_kernel_launcher_uint8(const uint8_t *input_data, uint8_t *output_data, 
                                  const int dims, const int64_t *in_dims, 
                                  const int64_t *out_dims, const int64_t *pad_before, 
                                  const int64_t *pad_after, const uint8_t pad_value, 
                                  const int64_t total_out_elements, const musaStream_t stream) {
    const int block_size = 256;
    int grid_size = (static_cast<int>(total_out_elements) + block_size - 1) / block_size;
    nd_pad_kernel<uint8_t><<<grid_size, block_size, 0, stream>>>(
        input_data, output_data, dims, in_dims, out_dims, pad_before, pad_after, 
        pad_value, total_out_elements);
}

} // extern "C"
