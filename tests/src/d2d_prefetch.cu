template <typename T> 
__global__ void d2d_prefetch_kernel(T *dst, const T * __restrict__ src, size_t size) {
  // size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  // size_t s_stride = gridDim.x * blockDim.x;
  // size_t stride = gridDim.x * blockDim.x *unroll;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  dst += index;
  src += index;
  //  T tmp[unroll];
  __shared__ T sh_tmp[JC_cuda_blocksize];
  T tmp;
  if (index < size){
    tmp = *src;  src+=stride;
  }
  for (; index < size; ) 
    {
      index += stride;
      sh_tmp[threadIdx.x] = tmp;
      //prefetch
      if (index < size) {
        tmp = *src;
        src+=stride;
      }
      *dst  = sh_tmp[threadIdx.x];
      dst += stride;

    }

    // for the remaining elements
  // for (; index < size;  index += s_stride)  {
  //   *dst = *src;
  //   dst+=s_stride;
  //   src+=s_stride;
  // }

}

template <typename T>
void d2d_prefetch(T *dst, const T *src, size_t size){
  const int BS = jusha::cuda::JCKonst::cuda_blocksize;
  int Blocks = std::min(jusha::cuda::JCKonst::cuda_max_blocks, (int)(size+BS-1/BS));
  d2d_prefetch_kernel<T><<<Blocks, BS>>>(dst, src, size);
}
