static  texture<float4,1>  tex_x_float4;

inline void my_bind_x(const float4 * x, size_t size)
{
  size_t offset = size_t(-1);
  cudaBindTexture(&offset, tex_x_float4, x, size*sizeof(float4));
  jassert(offset == 0);
}

inline void my_unbind_x(const float4 * x)
{ cudaUnbindTexture(tex_x_float4);  }


template <bool UseCache>
__inline__ __device__ float4 my_fetch_x(const int& i, const float4 * __restrict__ x)
{
#ifndef DEBUG
  if (UseCache)
    return tex1Dfetch(tex_x_float4, i);
  else
    return x[i];
#else
  return x[i];
#endif

}


template <typename T> 
__global__ void d2d_texture_kernel(T *dst, const T *src, size_t size) {
  for (size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; 
       index += gridDim.x * blockDim.x) 
    {
      dst[index] = my_fetch_x<true>(index, src);
    }
}

template <typename T>
void d2d_texture(T *dst, const T *src, size_t size){
  int BS = jusha::cuda::JCKonst::cuda_blocksize;
  int Blocks = std::min(jusha::cuda::JCKonst::cuda_max_blocks, (int)(size+BS-1/BS));
  my_bind_x(src, size);
  d2d_texture_kernel<<<Blocks, BS>>>(dst, src, size);
  my_unbind_x(src);
}
