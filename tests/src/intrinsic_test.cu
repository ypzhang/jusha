#include <catch.hpp>
#include "utility.h"

#include "cuda/cuda_intrinsic.hpp"
#include "cuda/array.h"

using namespace jusha;

__global__ void sum_kernel(float *val, int size)
{
  float _sum = 0.0;
  for (int i = threadIdx.x ; i < size; i+=blockDim.x)
    _sum += val[i];
  _sum = jusha::cuda::blockReduceSum(_sum);
  if (threadIdx.x == 0)
    val[0] = _sum;
}

TEST_CASE( "BlockSum", "[sum]" ) {
  cuda::MirroredArray<float> to_sum(480);
  float *sum_ptr = to_sum.getPtr();
  for (auto i = 0; i != to_sum.size(); i++)
    sum_ptr[i] = (float)i;

  sum_kernel<<<1, 512>>>(to_sum.getGpuPtr(), to_sum.size());
  printf("BLock sum is %f\n", to_sum.getElementAt(0));
}
