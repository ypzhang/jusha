// reduce primitives on device
// #include <cub/cub.h>
namespace jusha {
  namespace cuda {
  

    template <int block_size, typename T> 
    __inline__  __device__ T block_reduce(const T &val, volatile T *sh_mem) 
    {
      assert(block_size <= 1024);
      assert(block_size >= JC_cuda_warpsize);
      sh_mem[threadId.x] = val;
      __syncthreads();
      if (block_size >= 1024) {
        if (threadIdx.x < 512)  sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 512];
        __syncthreads();
      }

      if (block_size >= 512) {
        if (threadIdx.x < 256)  sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 256];
        __syncthreads();
      }

      if (block_size >= 256) {
        if (threadIdx.x < 128)  sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 128];
        __syncthreads();
      }

      if (block_size >= 128) {
        if (threadIdx.x < 64)  sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 64];
        __syncthreads();
      }

      if (block_size >= 64) {
        if (threadIdx.x < 32)  sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 32];
        __syncthreads();
      }

      if (threadIdx.x < 32) { 
        sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 16]; // __syncthreads();
        sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 8];
        sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 4];
        sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 2];
        sh_mem[threadIdx.x] += sh_mem[threadIdx.x + 1];
      }
      __syncthreads();
      return sh_mem[0];
    }   



  
  }
}
