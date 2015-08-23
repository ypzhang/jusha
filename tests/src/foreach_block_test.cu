#include <catch.hpp>
#include "utility.h"

#include "cuda/cuda_config.h"
#include "cuda/array.h"

#include "cuda/for_each.hpp"

using namespace jusha;

template <class T>
class atomic_run_nv_block: public nvstd::function<void(T)> {
public:
  __device__ void operator()(int gid, thrust::tuple<T*, T> &tuple) const {
    // if (gid == 32)
    //   printf("here gid %d threadIdx.x %d, blockIdx.x %d.\n", gid, threadIdx.x, blockIdx.x);
    atomicAdd(thrust::get<0>(tuple) + gid, thrust::get<1>(tuple));
  }
};


#if 1
TEST_CASE( "ForEachBlock", "[sum]" ) {
  int  n1 = 5120129;//1200000;
  JVector<int> sum(n1), sum_ref(n1);
  sum.zero();
  for (int i = 0; i != n1; i++)
    sum_ref[i] = 1;

  //  ForEachKernel<StridePolicy, 256, false> fe(300);
  //  AtomicAdd kernel(300);

  int add_per_thread = 1;
  //  AtomicAdd/*<decltype(atomic_run)>*/ kernel(/*atomic_run,*/ n1);
  ForEachKernel<BlockPolicy, JC_cuda_warpsize, false> kernel(n1, "AtomicBlockUt");
  kernel.disable_auto_tuning();
  //  int sum_now = 0;
//atomic_run_n nv_run;
//  printf("running atomic add kernel\n");
#if 0
  kernel.run<atomic_run_nv_block<int>, int *, int>(sum.getGpuPtr(), add_per_thread);
  sum_now = sum[0];
  REQUIRE(sum_now == n1*add_per_thread);
#endif

#if 0
  kernel.set_N(257);
  kernel.run<atomic_run_nv_block<int>, int *, int >(sum.getGpuPtr(), add_per_thread);
  REQUIRE(sum[0] == (sum_now + 257*add_per_thread));
#endif  
//  sum.zero();
//  n1 = 5120124;//1200000;

  kernel.set_N(n1);
  //  sum_now = sum[0];
  kernel.run<atomic_run_nv_block<int>, int *, int >(sum.getGpuPtr(), add_per_thread);

  REQUIRE(sum.isEqualTo(sum_ref));
  // for (int i = 0; i != n1; i++ )
  //   {
  //     if (sum[i] != 1) {
  //       printf("wrong sum @ %d value is %d\n", i, sum[i]);
  //       break;
  //     }
          //    }
//  REQUIRE(sum[0] == (sum_now + n1*add_per_thread));

  //  kernel.run(sum.getGpuPtr(), 2, sum.getReadOnlyPtr());
  //  kernel.run();
  //  generic_kernel<<<1,1>>>(sum.getGpuPtr());
  //  fe.run(sum.getGpuPtr());
  check_cuda_error("atomic", __FILE__, __LINE__);
}
#endif


