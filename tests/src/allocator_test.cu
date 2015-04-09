#include <catch.hpp>
#include "utility.h"
#include "cuda/heap_allocator.h"


using namespace jusha;


TEST_CASE( "MemoryManagement", "[mm]" ) {


}

TEST_CASE( "BinIndex", "[simple]" ) {
  int bin_id;
  size_t bin_bsize;
  bin_index(MIN_BLOCK_BSIZE, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 0);
  REQUIRE(bin_bsize == MIN_BLOCK_BSIZE);

  bin_index(MIN_BLOCK_BSIZE<<1, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 1);
  REQUIRE(bin_bsize == (MIN_BLOCK_BSIZE<<1));

  bin_index((MIN_BLOCK_BSIZE<<1)+4, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 2);
  REQUIRE(bin_bsize == (MIN_BLOCK_BSIZE<<2));

  bin_index(MAX_BLOCK_BSIZE, bin_id, bin_bsize) ;
  REQUIRE(bin_id == (NUM_BINS-1));
  REQUIRE(bin_bsize == MAX_BLOCK_BSIZE);

  bin_index(MAX_BLOCK_BSIZE-20, bin_id, bin_bsize) ;
  REQUIRE(bin_id == (NUM_BINS-1));
  REQUIRE(bin_bsize == MAX_BLOCK_BSIZE);

}
