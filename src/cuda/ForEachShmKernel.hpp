#pragma once

#include <cassert>
#include <thrust/tuple.h>
#include <cstdio>
#include <typeinfo>
#include <nvfunctional>
#include "./kernel.hpp"
#include "./policy.hpp"

namespace jusha {
  template <template <int, bool> class Policy, int groupsize, int need_sync, class Fn, class Shared_T, int SharedSize, class... Args>
  static __global__ void for_each_shm_kernel(int N, Args... args)
  {
    Shared_T sh_item[SharedSize];
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
    if (threadIdx.x == 0 && blockIdx.x == 0)
      printf("calling post proc\n");
    _method.post_proc(fe.get_id(), tuple);
  }


  /*!
   * A template kernel class that uses shared memory
   */
  template <template<int, bool> class Policy, int group_size, bool need_sync>
  class ForEachShmKernel: public CudaKernel {
  public: 
    explicit ForEachShmKernel(int32_t _N, const std::string &tag): m_N(_N), CudaKernel(_N, tag) {
    }

    template <class Method, class Shared_T, int SharedSize, class... Args>
    void run(Args... args) {
      m_blocks = GET_BLOCKS(m_N, m_block_size);
      int BS = m_block_size; //jusha::cuda::JCKonst::cuda_blocksize;
#if 0
      if (m_auto_tuning) {
        //must be equal to or above cuda 6.5 
        int min_gridsize, blocksize;
        cudaOccupancyMaxPotentialBlockSize(&min_gridsize, &blocksize, for_each_shm_kernel<Policy, group_size, need_sync, Method, Args...>,
                                           0, m_N);
        BS = blocksize;
        m_blocks = GET_BLOCKS(m_N, blocksize);
        m_blocks = std::min(blocks, 8 * min_gridsize);
        printf("auto tuning kernel %s to use block size %d grid size %d, min_gridsize %d\n",
               get_tag().c_str(), blocks, blocksize, min_gridsize);

      }
      //      printf ("running kernel %s at gridsize %d blocksize %d.\n", get_tag().c_str(), blocks, BS);
#endif
      m_blocks = std::min(m_max_blocks, m_blocks);
      printf ("running kernel %s at gridsize %d blocksize %d.\n", get_tag().c_str(), m_blocks, BS);
      for_each_shm_kernel<Policy, group_size, need_sync, Method, Shared_T, SharedSize, Args...><<<m_blocks, BS>>>(m_N, args...);
    }
      
    void set_N(int32_t _N) {
      m_N = _N;
    }
  private:
    int m_N{0};
    //    Fn m_method;
  };

}
