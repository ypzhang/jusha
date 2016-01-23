#pragma once

//#ifdef __GNUG__
//#endif
#include <cassert>
#include <thrust/tuple.h>
#include <cstdio>
#include <typeinfo>
#include <nvfunctional>
#include "kernel.hpp"
#include "policy.hpp"
//#include <thread>

namespace jusha {
  //template <int groupsize>

  /*!
   * The for_each kernel template for simple kernels that do not use shared memory
   * 
   */
  template <template <int, bool> class Policy, int groupsize, int need_sync, class Fn/*, class Policy*/, class... Args>
  static __global__ void for_each_kernel(int N, Args... args)
  {
    //    Policy policy;
    //    int stride = blockDim.x * gridDim.x;
    int max_id = blockDim.x * gridDim.x;
    int id = threadIdx.x+blockDim.x*blockIdx.x; 
    ForEach<Policy, groupsize, need_sync> fe(N, id, max_id);
    int stride = fe.get_stride();

    //    printf("here my_id %d max_id %d batches %d\n", my_id, max_id, m_batches);
    thrust::tuple<Args...> tuple (args...);
      int batches = fe.num_batches();

    Fn _method; 
    //    global_for_each(threadIdx.x, tuple);

    while (fe.not_last_batch()) {
        {
          _method(fe.get_id(), tuple);
        }
        fe.next_batch();
    }

    while (fe.not_done()) {
      if (fe.is_active(N))
        {
          //          if (blockIdx.x == 0 && threadIdx.x == 1)
          _method(fe.get_id(), tuple);
        }
      fe.next_batch();
    }
  }


  /*!
   * A template for each kernel class that does not use shared memorya
   */ 
  template <template<int, bool> class Policy, int group_size, bool need_sync>
  class ForEachKernel: public CudaKernel {
  public: 
    explicit ForEachKernel(int32_t _N, const std::string &tag): m_N(_N), CudaKernel(_N, tag) {
    }

    template <class Method, class... Args>
    void run(Args... args) {
      if (m_N == 0) return;
      int blocks = GET_BLOCKS(m_N, m_block_size);
      int BS = m_block_size; //jusha::cuda::JCKonst::cuda_blocksize;
      if (m_auto_tuning) {
        //must be equal to or above cuda 6.5 
        int min_gridsize, blocksize;
        cudaOccupancyMaxPotentialBlockSize(&min_gridsize, &blocksize, for_each_kernel<Policy, group_size, need_sync, Method, Args...>,
                                           0, m_N);
        //        gridsize = GET_BLOCKS(m_N, blocksize);
        //        gridsize = 120;
        // sometimes CUDA API returns 0, a possible bug in CUDA
        if (blocksize == 0) 
          blocksize = m_block_size;
        BS = blocksize;
        blocks = GET_BLOCKS(m_N, blocksize);
        blocks = std::min(blocks, 8 * min_gridsize);
        // printf("auto tuning kernel %s to use block size %d grid size %d, min_gridsize %d\n",
        //        get_tag().c_str(), blocks, blocksize, min_gridsize);

      }
      //      printf ("running kernel %s at gridsize %d blocksize %d.\n", get_tag().c_str(), blocks, BS);
      blocks = std::min(blocks, m_max_blocks);
      for_each_kernel<Policy, group_size, need_sync, Method, Args...><<<blocks, BS>>>(m_N, args...);
    }
      
    void set_N(int32_t _N) {
      m_N = _N;
    }
  private:
    int m_N{0};
    //    Fn m_method;
  };
}
