#pragma once
#include <cassert>
#include <tuple>
#include <cstdio>
#include <nvfunctional>
#include "cuda/cuda_config.h"
//#include <thread>

namespace jusha {
  
template <int groupsize>
class ForEachPolicy {
public:
  virtual __device__ void group_sync() {}
  __device__ int group_size() {
    return groupsize;
  }
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const = 0;
};

template <int groupsize>
class StridePolicy: public ForEachPolicy<groupsize> {
public:
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const {
    if (!need_sync) {
      return _N/max_id + (_N%max_id)<my_id? 1:0 ;
    } else {
      return (_N+max_id-1)/max_id;
    }
  }
};

template <int groupsize>
class BlockPolicy: public ForEachPolicy<groupsize> {
public:
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const {
    assert((max_id % groupsize) == 0);
    int num_groups = max_id /groupsize;
    int group_id = my_id/groupsize;
    int total_batches = (_N+groupsize-1)/groupsize;
    int batch_per_group = (total_batches + num_groups - 1)/num_groups;
    int batch_start = batch_per_group * group_id;
    int batch_end = batch_start + batch_per_group;
    batch_end = batch_end > total_batches? batch_end : total_batches;
    int num_batches = batch_end - batch_start;
    if (!need_sync)
      return num_batches;

    //    if (batch_end == total_batches)
    {
      int elem_in_last_batch = _N - batch_end * groupsize;
      int id_in_group = my_id % groupsize;
      return num_batches - (id_in_group > elem_in_last_batch? 1:0);
    }
    
  }
};

template <template<int> class Policy, int group_size, bool need_sync>
class ForEach {
public:
  __device__ ForEach(int32_t _N, int32_t my_id, int32_t max_id):
    N(_N), m_id(my_id), m_max_id (max_id), m_group_size(group_size) {

    Policy<group_size> policy;
    m_batches = policy.num_batches(N, my_id, max_id, need_sync);
    //    int lane_id = my_id % group_size;
  }
  
  __device__ bool is_active() const {
    return m_is_active;
  }

  __device__ void execute_wrapper()
  {
    
  }

  __device__ int num_batches() const {
    //    Policy<group_size
    return m_batches;
  }
  
  __device__ void next_batch() const {
    
  }
  
private:
  int N = 0;
  int m_id = 0;
  int m_max_id = 0;
  int m_group_size = 0;
  int m_batches = 0;
  bool m_is_active = false;
};


  class Each {
  public:
    __host__ __device__ Each(int _N):N(_N){}
  private:
    int N;
  };

  __global__ void dummy_kernel() {
    printf("in dummy kernel");
  }

  __global__ void wrapper_kernel(Each foreach) {
    dummy_kernel<<<1, 20>>>();
  }
  
  class KernelWrapper {
  public:
    void run(int N) {
      Each each(N);
      wrapper_kernel<<<1,1>>>(each);
    }
    
  };

  template <typename T>
  __device__ void for_each_recursive(T value)
  {
    printf("inside value for_each last\n");
  }

  template <typename T, class... Args>
  __device__ void for_each_recursive(T value, Args... args)
  {
    printf("inside value for_each\n");
    for_each_recursive(args...);
  }

  __device__ void test(int gid) {
    printf("int test gid %d.\n", gid);
  }

  template <class Fn, size_t tuple_size, class... Args>
  __global__ void for_each_kernel(int N, Args... args)
  {
    ForEach<StridePolicy,  256, false> fe(N, threadIdx.x+blockDim.x*blockIdx.x, blockDim.x * gridDim.x);

    //    printf("here my_id %d max_id %d batches %d\n", my_id, max_id, m_batches);
    std::tuple<Args...> tuple (args...);
    //    Fn method;
    //    std::tuple_element<0, std::tuple<Args...>> tuple_0;
    //    std::tuple_element<4, std::tuple<Args...>> tuple_4;
    // printf("first  %d.\n", std::get<0>(tuple));
    // printf("second %p.\n", std::get<1>(tuple));
    // printf("third %p.\n", std::get<2>(tuple));
    //for_each_recursive(args...);
    auto lambda = [](int gid) {
      printf("gid %d.\n", gid);
    };


    //    nvstd::function<decltype(method)> _method; // = method;
//    nvstd::function<void(int)> _method = method; // = method;
//    nvstd::function<void(int)> _method = Fn(); // = method;

#if 1
    int batches = fe.num_batches();
    //    printf("here. batches %d method %p\n", batches, _method);
    Fn _method; 
    _method(threadIdx.x, tuple);
    //    global_for_each(threadIdx.x, tuple);
    while (batches--) {
      if (fe.is_active())
        printf("I am here\n");
      printf("I am here\n");
      //      method(threadIdx.x);

      fe.next_batch();
    }
#endif
    //    int num_batches = fe.
  }


  template <template<int> class Policy, class Fn, int group_size, bool need_sync>
  class ForEachKernel {
  public: 
    explicit ForEachKernel(int32_t _N, Fn method): N(_N) {
      //      m_method = method;
    }

    template <class Method, class... Args>
    void run(Args... args) {
      int blocks = GET_BLOCKS(N);
      blocks = CAP_BLOCK_SIZE(blocks);
      int BS = 1; //jusha::cuda::JCKonst::cuda_blocksize;
      printf("calling generic kernel\n");
      //      std::tuple_size<std::tuple<Args...>> tuple_size;
    
      for_each_kernel<Method, std::tuple_size<std::tuple<Args...>>::value, Args...><<<blocks, BS/*, tuple_size::value*/>>>(N, args...);
      //      cudaDeviceReset();
    }

    virtual __device__ void do_1() {}
  private:
    int N{0};
    //    Fn m_method;
  };
}
