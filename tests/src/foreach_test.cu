#include <catch.hpp>
#include "utility.h"
#include "cuda/for_each.hpp"
#include "cuda/cuda_config.h"
#include "cuda/array.h"

using namespace jusha;

__global__ void kernel(int N)
{
  ForEach<StridePolicy, 128, false> for_each(N, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
  
}

class AtomicAdd: public ForEachKernel<StridePolicy, JC_cuda_blocksize, false> 
{
public:
  explicit AtomicAdd(int _N): ForEachKernel<StridePolicy, JC_cuda_blocksize, false>(_N){
  }
  
  virtual __device__ void do_1
() {}  

};

TEST_CASE( "ForEach", "[sum]" ) {
  JVector<int> sum(1);
  sum.zero();
  //  ForEachKernel<StridePolicy, 256, false> fe(300);
  //  AtomicAdd kernel(300);
  AtomicAdd kernel(3);
  kernel.run(sum.getGpuPtr());
  //  kernel.run();
  //  generic_kernel<<<1,1>>>(sum.getGpuPtr());
  //  fe.run(sum.getGpuPtr());
}

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
