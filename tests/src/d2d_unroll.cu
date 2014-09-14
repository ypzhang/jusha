template <typename T, int unroll> 
__global__ void d2d_unroll_kernel(T *dst, const T * __restrict__ src, size_t size) {
  // size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  // size_t s_stride = gridDim.x * blockDim.x;
  // size_t stride = gridDim.x * blockDim.x *unroll;
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int s_stride = gridDim.x * blockDim.x;
  int stride = gridDim.x * blockDim.x *unroll;

  dst += index;
  src += index;
  //  T tmp[unroll];
  for (; index < size-stride;) 
    {
#if 1
      #pragma unroll 
      for (int iter = 0; iter != unroll; iter++) {
        *dst  = my_fetch_x<true>(index, src);
        index += s_stride;
        dst += s_stride;
        //        src += s_stride;
      }
      // #pragma unroll 
      // for (int iter = 0; iter != unroll; iter++) {
      //   tmp[iter] = *src;
      //   src += s_stride;
      // }

      // #pragma unroll 
      // for (int iter = 0; iter != unroll; iter++) {
      //   *dst = tmp[iter];
      //   dst += s_stride;
      // }
#else
      if (unroll >= 8) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        index += stride;
      }
      if (unroll >= 7) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      if (unroll >= 6) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      if (unroll >= 5) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }

      if (unroll >= 4) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      if (unroll >= 3) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      if (unroll >= 2) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      if (unroll >= 1) {
        *dst = *src;
        dst+=s_stride;
        src+=s_stride;
        //        dst[index] = src[index];
        //        index += stride;
      }
      index  += stride;
#endif

    }

    // for the remaining elements
  for (; index < size;  index += s_stride)  {
    *dst = *src;
    dst+=s_stride;
    src+=s_stride;
  }

}

template <typename T, int unroll>
void d2d_unroll(T *dst, const T *src, size_t size){
  int BS = jusha::cuda::JCKonst::cuda_blocksize;
  int Blocks = std::min(jusha::cuda::JCKonst::cuda_max_blocks, (int)(size+BS-1/BS));
  my_bind_x(src, size);  
  d2d_unroll_kernel<T, unroll><<<Blocks, BS>>>(dst, src, size);
  my_unbind_x(src);
}
