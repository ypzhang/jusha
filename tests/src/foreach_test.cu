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


TEST_CASE( "ForEach", "[sum]" ) {
  JVector<int> sum(1);
  sum.zero();
  
}
