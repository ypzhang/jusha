#pragma once

#include <algorithm>
#include "cuda/cuda_config.h"


namespace jusha {

template <typename T> 
void d2d_memcpy_kernel(T *dst, const T *src, size_t _size) {
  size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  for (; gid < _size; gid++) {
    dst[gid] = src[gid];
  }
}


template <typename T>
void d2d_memcpy(T *dst, const T *src, size_t _size) 
{
  int block_size = jusha::cuda::cuda_blocksize;
  int blocks = std::min((_size+block_size-1)/block_size, (unsigned long)J_CUDA_MAX_BLOCKS);
  d2d_memcpy_kernel<<<blocks, block_size>>>(dst, src, _size);
}  




}
