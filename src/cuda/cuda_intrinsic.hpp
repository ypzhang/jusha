#pragma once
#include <cassert>
#include <cub/cub.cuh>

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
      static __shared__ T shared[32]; // Shared mem for 32 partial sums
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

    /*! Block level in-place scan over a range */
    template <class T, class Op, int BS, bool exclusive>
    __inline__ __device__
    void blockScan(T *start, T *end) {
      static __shared__ T scan_val[BS];
      assert(blockDim.x == BS);
      Op op;
      int N = end - start;
      T carry_out = T();
      T outval;
      for (int id = threadIdx.x;  id < (N + BS-1)/BS * BS; id+=BS) {
        T val;
        if (id < N)
          val = start[id];
        if (threadIdx.x == 0)  {
          val = op(val, carry_out);
        }
        scan_val[threadIdx.x] = val;
        __syncthreads();
        if (threadIdx.x >=  1)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  1]); __syncthreads();
        if (threadIdx.x >=  2)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  2]); __syncthreads();
        if (threadIdx.x >=  4)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  4]); __syncthreads();
        if (threadIdx.x >=  8)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x -  8]); __syncthreads();
        if (threadIdx.x >= 16)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 16]); __syncthreads();
        if (threadIdx.x >= 32)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 32]); __syncthreads();
        if (threadIdx.x >= 64)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 64]); __syncthreads();
        if (threadIdx.x >= 128)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 128]); __syncthreads();
        if (threadIdx.x >= 256)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 256]); __syncthreads();
        if (threadIdx.x >= 512)  scan_val[threadIdx.x] = op(scan_val[threadIdx.x], scan_val[threadIdx.x - 512]); __syncthreads();
        if (!exclusive)
          outval = scan_val[threadIdx.x];
        else {
          if (threadIdx.x == 0) {
            outval = carry_out;
          }
          else
            outval = scan_val[threadIdx.x-1];
        }
        carry_out = scan_val[BS-1];

        if (id < N) {
          start[id] = outval;
        }
        
      }
    }

    /*********** sort *
     */

  } // cuda
} // jusha
