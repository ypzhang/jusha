#include <catch.hpp>
#include "utility.h"

#include "cuda/cuda_config.h"
#include "cuda/array.h"
#include "cuda/ForEachShmKernel.hpp"
#include "cuda/for_each.hpp"
#include "cuda/cuda_intrinsic.hpp"

using namespace jusha;

template <class T>
struct ShReduce {
  T *sh_ptr;
};

template <class T>
class reduce_run_nv: public nvstd::function<void(T)> {
public:
  __device__ reduce_run_nv() {
    m_reduce = {};
  }
  __device__ void operator()(int gid, thrust::tuple<const T*, T*> &tuple)  {
    m_reduce += (thrust::get<0>(tuple))[gid];
    //    atomicAdd(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }

  __device__ void post_proc(int gid, thrust::tuple<const T*, T*> &tuple)  {
    m_reduce = jusha::cuda::blockReduceSum(m_reduce);
    if (threadIdx.x == 0)
      m_reduce += (thrust::get<1>(tuple))[blockIdx.x] = m_reduce;

    //    m_reduce += (thrust::get<0>(tuple))[gid];
    //    atomicAdd(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }

  //  __device__ ~reduce_run_nv() {
    // m_reduce = jusha::cuda::blockReduceSum(m_reduce);
    // if (threadIdx.x == 0)
    //   printf("my reduce is %d.\n", m_reduce);
  //  }

private:
  T m_reduce;
};



TEST_CASE( "ForEachShmReduce", "[sum]" ) {
  int n = 2000;
  JVector<int> sum(n);
  thrust::fill(sum.gbegin(), sum.gend(), 1);
  //  ForEachKernel<StridePolicy, 256, false> fe(300);
  //  AtomicAdd kernel(300);
  //  AtomicAdd/*<decltype(atomic_run)>*/ kernel(/*atomic_run,*/ n1);
  ForEachShmKernel<BlockPolicy, JC_cuda_warpsize, false> kernel(n, "Reduction");
  kernel.set_block_size(1024);
  kernel.set_max_block(1024);
  JVector<int> inter_sum(1024);
  constexpr int shared_bsize = sizeof(int)*1024/32;
  kernel.run<reduce_run_nv<int>, int, shared_bsize, const int *, int *>(sum.getReadOnlyGpuPtr(), inter_sum.getGpuPtr());



  // int sum_now = sum[0];
  // REQUIRE(sum_now == n1*add_per_thread);

  // kernel.set_N(257);
  // kernel.run<atomic_run_nv<int>, int *, int >(sum.getGpuPtr(), 12);
  // REQUIRE(sum[0] == (sum_now + 257*12));

  check_cuda_error("atomic", __FILE__, __LINE__);
}


