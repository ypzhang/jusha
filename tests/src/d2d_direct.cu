template <typename T> 
__global__ void d2d_direct_kernel(T *dst, const T *src, size_t size) {
  for (size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; index += gridDim.x * blockDim.x) 
      dst[index] = src[index];
}

template <typename T>
void d2d_direct(T *dst, const T *src, size_t size) {
  int Blocks = GET_BLOCKS(size);
  d2d_direct_kernel<<<Blocks, jusha::cuda::JCKonst::cuda_blocksize>>>(dst, src, size);
}
