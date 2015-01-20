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
  thrust::fill(vector.gbegin(), vector.gend(), 0xffffffff);
  
}

TEST_CASE( "VecMultiply", "[simple]" ) {
  int test_size = 20;
  JVector<int> v0(test_size), v1(test_size);
  v0.sequence(0);
  v1.sequence(0);
  JVector<int> v0v1(test_size);
  v0v1.zero();
  jusha::multiply(v0, v1, v0v1);

  for (int i = 0; i != test_size; i++)
    REQUIRE(v0v1[i] == (i*i));

  
}



