#include <catch.hpp>
#include "utility.h"

#include "cuda/cuda_config.h"
#include "cuda/array.h"

using namespace jusha;

template <class Tuple>
static __device__ void global_for_each(int gid, Tuple &tuple) {
  //   printf("first  %d.\n", std::get<0>(tuple));
  //   printf("second %p.\n", std::get<1>(tuple));
  //   printf("third %p.\n", std::get<2>(tuple));

  // printf("gid %d.\n", gid);
}

#include "cuda/for_each.hpp"


__global__ void kernel(int N)
{
  ForEach<StridePolicy, 128, false> for_each(N, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
  
}

//template <class Tuple>
class atomic_run_nv: public nvstd::function<void(int)> {
public:
__device__ void operator()(int gid, std::tuple<int, int *> &tuple) const {
  // printf("first  %d.\n", std::get<0>(tuple));
  // printf("second %p.\n", std::get<1>(tuple));
  atomicAdd(std::get<1>(tuple), std::get<0>(tuple));
}
};

__device__ void atomic_run(int gid) {
    printf("gid %d.\n", gid);
}


//template <class Fn>
class AtomicAdd: public ForEachKernel<StridePolicy, JC_cuda_blocksize, false> 
{
public:
  explicit AtomicAdd(int N): ForEachKernel<StridePolicy, JC_cuda_blocksize, false>(N){
  }
};

TEST_CASE( "ForEach", "[sum]" ) {
  JVector<int> sum(1);
  sum.zero();
  //  ForEachKernel<StridePolicy, 256, false> fe(300);
  //  AtomicAdd kernel(300);
  int n1 = 3;
  int add_per_thread = 2;
  //  AtomicAdd/*<decltype(atomic_run)>*/ kernel(/*atomic_run,*/ n1);
  ForEachKernel<StridePolicy, JC_cuda_blocksize, false> kernel(n1);
//atomic_run_nv nv_run;
//  printf("running atomic add kernel\n");
  kernel.run<atomic_run_nv, int, int *>(add_per_thread, sum.getGpuPtr());
  int sum_now = sum[0];
  REQUIRE(sum_now == n1*add_per_thread);

  kernel.set_N(257);
  kernel.run<atomic_run_nv, int, int *>(12, sum.getGpuPtr());
  REQUIRE(sum[0] == (sum_now + 257*12));

  //  kernel.run(sum.getGpuPtr(), 2, sum.getReadOnlyPtr());
  //  kernel.run();
  //  generic_kernel<<<1,1>>>(sum.getGpuPtr());
  //  fe.run(sum.getGpuPtr());
  check_cuda_error("atomic", __FILE__, __LINE__);
}

#if 0
TEST_CASE( "ForEach2", "[wrapper]" ) {
  JVector<int> sum(1);
  sum.zero();
  //  ForEachKernel<StridePolicy, 256, false> fe(300);
  //  AtomicAdd kernel(300);
  KernelWrapper kernel;
  kernel.run(3);
  //  kernel.run(sum.getGpuPtr());
  //  kernel.run();
  //  generic_kernel<<<1,1>>>(sum.getGpuPtr());
  //  fe.run(sum.getGpuPtr());
}
#endif
