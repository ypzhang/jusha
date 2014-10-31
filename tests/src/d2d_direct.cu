template <typename T> 
__global__ void d2d_direct_kernel(T *  __restrict__  dst, const T * __restrict__ src, size_t size) {
  for (size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; 
       index += gridDim.x * blockDim.x) 
    {
      dst[index] = src[index];
    }
}

template <typename T>
void d2d_direct(T *dst, const T *src, size_t size){
  int BS = jusha::cuda::JCKonst::cuda_blocksize;
  int Blocks = std::min(jusha::cuda::JCKonst::cuda_max_blocks, (int)(size+BS-1/BS));
  //  printf("copying from %p to %p\n", src, dst);
  d2d_direct_kernel<<<Blocks, BS>>>(dst, src, size);
}
