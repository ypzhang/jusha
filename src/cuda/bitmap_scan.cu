#include <cstdio>
#include <cassert>
#include <thrust/device_vector.h>
#include "./bitmap_scan.h"
#include "./cuda_config.h"

// A good article about restrict keyword
// http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html

namespace jusha {
  namespace cuda {
    /*! 
      \brief Determine warp begin and end index from total work load N
      Both  warp_begin and warp_end are dividable by warp_size
     */
    __device__ void partition_N_by_warp(const int N
                                        , int warp_id
                                        , int total_warp
                                        , int &warp_begin
                                        , int &warp_end) 
    {
      size_t warps_to_do = (N+JC_cuda_warpsize-1)>>JC_cuda_warpsize_shift; 
      warp_begin = (((warp_id+0) * warps_to_do) / total_warp) << JC_cuda_warpsize_shift;
      warp_end   = (((warp_id+1) * warps_to_do) / total_warp) << JC_cuda_warpsize_shift;
    }

    __inline__ __device__ int warp_reduce(int val, volatile int *sh_mem)
    {
#if __CUDA_ARCH__ < 300
      sh_mem[threadIdx.x] = val;
      #pragma unroll
      for (int offset = JC_cuda_blocksize/2; offset > 0; offset /= 2)  {
        if (threadIdx.x < offset) sh_mem[threadIdx.x] += sh_mem[threadIdx.x + offset];
      }
      val = sh_mem[threadIdx.x >> JC_cuda_warpsize_shift << JC_cuda_warpsize_shift];
#else // skip shared memory for Kepler architecture
      #pragma unroll
      for (int offset = JC_cuda_warpsize/2; offset > 0; offset /= 2) 
        val += __shfl_down(val, offset);
#endif      
      return val;
    }

    template <typename T>
    __global__ void warp_reduction_k( const T * __restrict__ input
                                    , T * __restrict__ reduction_per_warp
                                    , const int N)
    {
#if __CUDA_ARCH__ < 300
      __shared__ int sh_sum[JC_cuda_blocksize];
#else
      int * sh_sum(0);   // a dummy pointer
#endif      
      int g_thread_id = threadIdx.x + blockIdx.x * blockDim.x;
      int warp_id = g_thread_id >> JC_cuda_warpsize_shift;
      int total_warp = gridDim.x * blockDim.x >> JC_cuda_warpsize_shift;
      int warp_begin, warp_end;
      int N_guard = (N+31)>>5;
      partition_N_by_warp(N_guard, warp_id, total_warp, warp_begin, warp_end);
      int lane_id = threadIdx.x & JC_cuda_warpsize_mask;
      if (lane_id == 0) printf("warp id %d begin %d end %d.\n", warp_id, 
                               warp_begin, warp_end);
      int thread_sum(0);

      for (int index = warp_begin; index  < warp_end; index += JC_cuda_warpsize) {
        if (index + lane_id < N_guard) {
          printf("reading val %d by warp_id %d lane_id %d guard %d.\n", 
                 input[index + lane_id], warp_id, lane_id, N_guard);
          thread_sum += __popc(input[index + lane_id]);
        }
      }

      // the last warp process the last (N - warp_end) elements
      // if (warp_id == (total_warp-1) & lane_id < (N - warp_end)) {
      //   printf("reading index %d.\n", warp_end + lane_id);
      //   thread_sum += __popc(input[warp_end + lane_id]);
      // }

      thread_sum = warp_reduce(thread_sum, sh_sum);

      if (lane_id == 0) {
        printf("writing warp id %d to %d.\n", warp_id, thread_sum);
        reduction_per_warp[warp_id] = thread_sum;
      }
    }

    template <int block_size, bool exclusive>
    __device__ int one_block_scan(int val, int &carry_out, volatile int *sh_mem) 
    {
      //#if __CUDA_ARCH__ < 300
      sh_mem[threadIdx.x] = val;
      __syncthreads();
      
      
      //#else//
      //#endif
    }

    /*!
     */
    template <typename T, int block_size, bool exclusive>
    __global__ void single_block_scan_k(const T * __restrict__ input
                                        , T * __restrict__ output
                                        , const int N)
    {
      assert(gridDim.x == 1);
#if __CUDA_ARCH__ < 300
      __shared__ int sh_sum[block_size];
#else
      __shared__ int sh_sum[block_size >> JC_cuda_warpsize_shift];
#endif
      int base_id = 0;
      int carry_out(0);
      int val;
      for (; base_id < N; base_id += block_size) {
        val = input[base_id + threadIdx.x];
        output[base_id + threadIdx.x] = one_block_scan<block_size, exclusive>(val, carry_out, sh_sum);
      }
      // for the remaining (N - base_id) elements
      if (base_id + threadIdx.x < N) 
        val = input[base_id + threadIdx.x];

      val = one_block_scan<block_size, exclusive>(val, carry_out, sh_sum);
      if (base_id + threadIdx.x < N) 
        output[base_id + threadIdx.x] = val;
        
    }


    template <bool exclusive> 
    __global__ void bitmap_scan(const int * __restrict__ warp_carry_out) 
    {
      int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> JC_cuda_warpsize_shift;
      int carryout = warp_carry_out[warp_id];      
    }
    
    void exclusive_bitmap_scan(thrust::device_ptr<unsigned int> in_begin, 
                               thrust::device_ptr<unsigned int> out_begin,
                               int N)
    {
      int BS = JCKonst::cuda_blocksize;
      int input_size = (N + sizeof(unsigned int) - 1)/sizeof(unsigned int);
      int nblocks = std::min(JCKonst::cuda_max_blocks, (input_size+BS-1)/BS);
      int nwarps = (BS * nblocks)/JC_cuda_warpsize; 
      // the intermediate buffer for warp reductions
      thrust::device_vector<unsigned int> warp_reduction_vec(nwarps);

      warp_reduction_k<<<nblocks, BS>>>(thrust::raw_pointer_cast(in_begin),
                                        thrust::raw_pointer_cast(&warp_reduction_vec[0]),

                                        N);

      // single_block_scan_k<unsigned int, 1024, true><<<1, 1024>>>(thrust::raw_pointer_cast(&warp_reduction_vec[0]),
      //                                                     thrust::raw_pointer_cast(&warp_reduction_vec[0]),
      //                                                     warp_reduction_vec.size());
      
    }
    
  }
}
