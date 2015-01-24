#pragma once
#include <cuda_runtime_api.h>

namespace jusha {
  namespace cuda {
    /*! \brief defines are used in device code
     */
    #define JC_cuda_warpsize_shift  5
    #define JC_cuda_warpsize_mask   0x1F
    #define JC_cuda_blocksize       256
    #define JC_cuda_blocksize_shift 8
    #define JC_cuda_bs_mask         0xFF
    #define JC_cuda_bs_mask2        0xFFFFFF00
    #define JC_cuda_max_blocks      64
    #define JC_cuda_warpsize        32

    class JCKonst {
    public:
      static const int cuda_blocksize;
      static const int cuda_max_blocks;
      static const int cuda_warpsize;
      static const int cuda_warpsize_shift;
      static const int cuda_warpsize_mask;

    };

    __inline__ void get_cuda_property(cudaDeviceProp &property) {
      int gpu; cudaGetDevice(&gpu);
      cudaGetDeviceProperties(&property, gpu);
    }

  }
}


#define CAP_BLOCK_SIZE(block) (block > jusha::cuda::JCKonst::cuda_max_blocks ? jusha::cuda::JCKonst::cuda_max_blocks:block)
#define GET_BLOCKS(N) CAP_BLOCK_SIZE( (N + jusha::cuda::JCKonst::cuda_blocksize -1 )/jusha::cuda::JCKonst::cuda_blocksize)

// used in kernels 
#define kernel_get_1d_gid  blockIdx.x*blockDim.x + threadIdx.x
#define kernel_get_1d_stride blockDim.x * gridDim.x

 


