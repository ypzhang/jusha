template <typename T> 
void d2d_direct_kernel(T *dst, cosnt T *src, size_t size) {
  for (size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; 
       index += gridDim.x * blockDim.x) 
    {
      dst[index] = src[index];
    }
}

template <typename T>
void d2d_direct(T *dst, const T *src, size_t size){
  int BS = jusha::cuda::cuda_blocksize;
  int Blocks = std::min(jusha::cuda::cuda_max_blocks, size+BS-1/BS);
  d2d_direct_kernel<<<Blocks, BS>>>(dst, src, size);
}
