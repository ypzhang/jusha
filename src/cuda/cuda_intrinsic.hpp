#pragma once
#include <cassert>
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
#if 0 // this does not work for some reason
      val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
      if (blockIdx.x == 0 && threadIdx.x < blockDim.x/warpSize)
        printf("************** reading shm reduce %f\n", val);
      if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
      return val;
#else
      // do it sequentially
      if (threadIdx.x == 0) {
        for (int i = 1; i < (blockDim.x + warpSize-1)/warpSize; i++)
          val += shared[i];
        shared[0] = val;
      }
      __syncthreads();
      if (threadIdx.x != 0)
        val = shared[0];
      // if (threadIdx.x == 0)
      //   printf("*** final val is %f.\n", shared[0]);
      return val;
#endif
    }


    /*!*****************************************************************
     *                                 Scan
     ******************************************************************/

  } // cuda
} // jusha
