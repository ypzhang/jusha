#include "catch.hpp"
#include "cuda/array.h"

TEST_CASE( "Simple", "[Array]" ) {
  int test_size = 1 << 23;
  jusha::cuda::MirroredArray<int> vector(test_size);
  REQUIRE( vector.size() == test_size );
  
}


