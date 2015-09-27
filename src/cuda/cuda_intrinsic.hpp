#pragma once
#include "./cuda_config.h"

namespace jusha {
  namespace cuda {

    /*!*****************************************************************
     *                                 Reduction
     ******************************************************************/
    /* Warp level reduction, Only the first lane id gets the reduction */
    template <class T>
    __inline__ __device__
    T warpReduceSum(T val) {
      for (int offset = JC_cuda_warpsize/2; offset > 0; offset /= 2) 
        val += __shfl_down(val, offset);
      return val;
    }
    
    /* Warp level reduction, all lane id get the reduction */
    template <class T>
    __inline__ __device__
    T warpAllReduceSum(T val) {
      for (int mask = warpSize/2; mask > 0; mask /= 2) 
        val += __shfl_xor(val, mask);
      return val;
    }
    
    /* block level reduction */
    template <class T>
    __inline__ __device__
    T blockReduceSum(T val) {
      assert(blockDim.x <= (32*32));
      static __shared__ int shared[32]; // Shared mem for 32 partial sums
      int lane = threadIdx.x % warpSize;
      int wid = threadIdx.x / warpSize;
      
      val = warpReduceSum(val);     // Each warp performs partial reduction
      
      if (lane==0) shared[wid]=val; // Write reduced value to shared memory
      
      __syncthreads();              // Wait for all partial reductions
      
      //read from shared memory only if that warp existed
      val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
      
      if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
      
      return val;
    }


    /*!*****************************************************************
     *                                 Scan
     ******************************************************************/

  } // cuda
} // jusha
