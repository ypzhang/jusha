#include "sparse/distri_obj.h"

#include <catch.hpp>
#include "utility.h"
#include "cuda/heap_allocator.h"
#include <list>

using namespace jusha;

namespace {
TEST_CASE( "DistirObj", "[obj]" ) {
  DistriObj distri;
  REQUIRE(distri.get_partition_size() == 1);
  std::vector<int> partition(5, 1);
  int rank = 2;
  distri.set_partition(partition, rank);
  
  REQUIRE(distri.get_partition_size() == 5);
  REQUIRE(distri.get_my_size() == 1);
  REQUIRE(distri.get_my_start() == rank);  
}

}
