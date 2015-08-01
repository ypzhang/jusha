#pragma once

//#ifdef __GNUG__
#include <cxxabi.h>
//#endif
#include <sstream>
#include <cassert>
#include <thrust/tuple.h>
#include <cstdio>
#include <typeinfo>
#include <nvfunctional>
#include "kernel.hpp"
#include "cuda/cuda_config.h"
//#include <thread>

namespace jusha {
  //template <int groupsize>
class ForEachPolicy {
// #ifndef __forceinline__ 
// #define __forceinline__ 
//#endif

public:
  virtual __device__ void group_sync() {}
  // __device__ int group_size() {
  //   return groupsize;
  // }
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const = 0;
};

  template <int groupsize/* = jusha::cuda::JC_cuda_blocksize*/, bool need_sync>
  class StridePolicy { // : public ForEachPolicy<groupsize> {
public:
    __device__ void init(const int & _N, const int &my_id, const int &max_id, const bool &_need_sync, int &n_batches) const {
    if (!_need_sync) {
      n_batches = (_N/max_id);
      n_batches += my_id < (_N - n_batches * max_id)? 1:0 ;
    } else {
      n_batches = (_N+max_id-1)/max_id;
    }
  }
    __device__ int stride() const {
      return blockDim.x * gridDim.x;
    }
    __device__  __forceinline__ void next_batch(int &batches, int &my_id) {
     --batches;
     my_id += stride();
    //    is_active = (my_id < N);
   }
    //    __device__ 
};

  template <int groupsize/* = jusha::cuda::JC_cuda_blocksize*/, bool need_sync>
  class BlockPolicy { // :public ForEachPolicy{
public:
  __device__ void init(const int & _N, const int &my_id, const int &max_id, const bool &_need_sync, int &n_batches) const {
    n_batches = num_batches(_N, my_id, max_id, _need_sync);
  }

    __device__ int stride() const {
      return groupsize;
    }

  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool _need_sync) const {
    assert((max_id % groupsize) == 0);
    int num_groups = max_id /groupsize;
    int group_id = my_id/groupsize;
    int total_batches = (_N+groupsize-1)/groupsize;
    int batch_per_group = (total_batches + num_groups - 1)/num_groups;
    int batch_start = batch_per_group * group_id;
    int batch_end = batch_start + batch_per_group;
    batch_end = batch_end > total_batches? batch_end : total_batches;
    int num_batch = batch_end - batch_start;
    if (!_need_sync)
      return num_batch;

    //    if (batch_end == total_batches)
    {
      int elem_in_last_batch = _N - batch_end * groupsize;
      int id_in_group = my_id % groupsize;
      return num_batch - (id_in_group > elem_in_last_batch? 1:0);
    }
    
  }

  __device__  __forceinline__ void next_batch(int &batches, int &my_id) {
    --batches;
    my_id += stride();
    //    is_active = (my_id < N);
    }

};

  template <template<int, bool> class Policy, int group_size, bool need_sync>
class ForEach {
public:
  __device__ ForEach(int32_t _N, int32_t my_id, int32_t max_id):
    m_id(my_id){
    
    //    Policy<group_size> policy;
    policy.init(_N, my_id, max_id, need_sync, m_batches);

    //    int lane_id = my_id % group_size;
  }

  __device__ __forceinline__ int num_batches() const {
    //    Policy<group_size
    return m_batches;
  }
  
  __device__ __forceinline__ bool not_done() const {
    return m_batches > 0;
  }

  __device__ __forceinline__ bool not_last_batch() const {
    return m_batches > 1;
  }

  __device__ __forceinline__ bool is_active(int N) {
    return (m_id < N);
  }

  __device__ __forceinline__ int get_id() const {
    return m_id;
  }

    __device__ __forceinline__ int get_stride() const {
      return policy.stride();
    }
  __device__ __forceinline__  void next_batch() {
    policy.next_batch(m_batches, m_id);
    //    Policy<group_size> policy;
    //   --m_batches;
    // m_id += m_stride;
    // m_is_active = (m_id < N);

  //    policy.next_batch(m_batches, m_id, m_stride, m_is_active, N);
   }
  
private:
  Policy<group_size, need_sync> policy;
  int m_id = 0;
  //  int m_group_size = 0;
  int m_batches = 0;
  //  bool m_is_active = false;
};

  template <template <int, bool> class Policy, class Fn/*, class Policy*/, class... Args>
  static __global__ void for_each_kernel(int N, Args... args)
  {
    //    Policy policy;
    //    int stride = blockDim.x * gridDim.x;
    int max_id = blockDim.x * gridDim.x;
    int id = threadIdx.x+blockDim.x*blockIdx.x; 
    ForEach<Policy, 256, false> fe(N, id, max_id);
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


  template <template<int, bool> class Policy, int group_size, bool need_sync>
  class ForEachKernel: public CudaKernel {
  public: 
    explicit ForEachKernel(int32_t _N): N(_N) {
      //      m_method = method;
      std::stringstream tag_stream ;
#ifdef __GNUG__
      int status;
      char * demangled = abi::__cxa_demangle(typeid(*this).name(),0,0,&status);
      //      m_tag = std::string(demangled);
      tag_stream << demangled;
      free(demangled);
#else
      tag_stream <<       typeid(this).name();
#endif
      tag_stream << ":Dim_" << _N;
      set_tag(tag_stream.str());
    }

    template <class Method, class... Args>
    void run(Args... args) {
      printf ("running kernel %s.\n", get_tag().c_str());
      int blocks = GET_BLOCKS(N);
      int BS = jusha::cuda::JCKonst::cuda_blocksize;
      for_each_kernel<Policy, Method, Args...><<<blocks, BS>>>(N, args...);
    }
      
    void set_N(int32_t _N) {
      N = _N;
    }
  private:
    int N{0};
    //    Fn m_method;
  };
}
