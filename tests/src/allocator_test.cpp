#include <catch.hpp>
#include "utility.h"
#include "cuda/heap_allocator.h"
#include <list>

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

TEST_CASE( "HeapManager", "[simple]" ) {
  HeapAllocator hm;
  int *ptr = (int*)hm.allocate(MIN_BLOCK_BSIZE);
}

TEST_CASE( "HeapSubBin", "[simple]" ) {
  std::list<std::unique_ptr<SubBin>> bins;
  bins.emplace_back(std::unique_ptr<SubBin>(new SubBin()));
  bins.back()->init(20, SUB_BIN_SIZE);
  size_t index;
  for (int i = 0; i != SUB_BIN_SIZE; i++) {
    REQUIRE(bins.back()->insert(index));
    REQUIRE(index == i);
  }
  REQUIRE(bins.back()->insert(index)== false);
  for (size_t i = 0; i != SUB_BIN_SIZE; i++) {
    bins.back()->remove(i);
  }
}

TEST_CASE( "HeapBin", "[simple]" ) {
#if 0
  std::list<std::unique_ptr<SubBin<SUB_BIN_SIZE>>> bins;
  bins.emplace_back(std::unique_ptr<SubBin<SUB_BIN_SIZE>>(new SubBin<SUB_BIN_SIZE>()));
  bins.back()->init(20);
  int index;
  for (int i = 0; i != SUB_BIN_SIZE; i++) {
    bins.back()->insert(index);
    REQUIRE(index == i);
  }
  bins.back()->insert(index);
  REQUIRE(index==-1);
  for (int i = 0; i != SUB_BIN_SIZE; i++) {
    bins.back()->remove(i);
  }
#endif
}
