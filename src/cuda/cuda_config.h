#pragma once

namespace jusha {
  namespace cuda {
    /*! \brief defines are used in device code
     */
    #define JC_cuda_warpsize_shift 5
    #define JC_cuda_warpsize_mask  0x1F
    #define JC_cuda_blocksize      256
    #define JC_cuda_max_blocks     64
    #define JC_cuda_warpsize       32

    class JCKonst {
    public:
      static const int cuda_blocksize;
      static const int cuda_max_blocks;
      static const int cuda_warpsize;
      static const int cuda_warpsize_shift;
      static const int cuda_warpsize_mask;

    };



  }
}

#define CAP_BLOCK_SIZE(block) (block > jusha::cuda::JCKonst::cuda_max_blocks ? jusha::cuda::JCKonst::cuda_max_blocks:block)
#define GET_BLOCKS(N) CAP_BLOCK_SIZE( (N + jusha::cuda::JCKonst::cuda_blocksize -1 )/jusha::cuda::JCKonst::cuda_blocksize)
