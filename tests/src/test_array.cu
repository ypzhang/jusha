#include <thrust/transform.h>

#include "catch.hpp"
#include "cuda/array.h"
#include "cuda/primitive.h"

TEST_CASE( "Simple", "[Array]" ) {
  int test_size = 1 << 23;
  jusha::cuda::MirroredArray<int> vector(test_size);
  REQUIRE( vector.size() == test_size );
  vector.resize(test_size+1);
  REQUIRE( vector.size() == (test_size +1));
  REQUIRE( vector.getGpuPtr() );
}


TEST_CASE( "Algorithm", "[bitmap scan]" ) {
  int test_size = 20;
  jusha::cuda::MirroredArray<int> vector(test_size);
  thrust::fill(vector.begin(), vector.end(), 0xffffffff);
  
}


