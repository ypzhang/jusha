#include <catch.hpp>
#include "utility.h"
#include "cuda/heap_allocator.h"
#include <list>

using namespace jusha;

namespace jusha {
  extern size_t g_sub_bin_size;
  extern size_t g_min_block_shift;
  extern size_t g_min_block_bsize;
  extern size_t g_max_block_shift;
  extern size_t g_max_block_bsize;
  extern size_t g_num_bins;

}

namespace {
TEST_CASE( "MemoryManagement", "[mm]" ) {


}

TEST_CASE( "BinIndex", "[simple]" ) {
  int bin_id;
  size_t bin_bsize;
  bin_index(g_min_block_bsize, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 0);
  REQUIRE(bin_bsize == g_min_block_bsize);

  bin_index(g_min_block_bsize<<1, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 1);
  REQUIRE(bin_bsize == (g_min_block_bsize<<1));

  bin_index((g_min_block_bsize<<1)+4, bin_id, bin_bsize) ;
  REQUIRE(bin_id == 2);
  REQUIRE(bin_bsize == (g_min_block_bsize<<2));

  bin_index(g_max_block_bsize, bin_id, bin_bsize) ;
  REQUIRE(bin_id == (g_num_bins-1));
  REQUIRE(bin_bsize == g_max_block_bsize);

  bin_index(g_max_block_bsize-20, bin_id, bin_bsize) ;
  REQUIRE(bin_id == (g_num_bins-1));
  REQUIRE(bin_bsize == g_max_block_bsize);

}

TEST_CASE( "HeapManager", "[simple]" ) {
  size_t free, total;
  int device_id = 0;
  cudaError_t error = cudaSetDevice(device_id);
  if (error != cudaSuccess) 
    printf("cudasetdevice return error.\n");

  struct cudaDeviceProp prop;
  error = cudaGetDeviceProperties(&prop, device_id);
  if (error != cudaSuccess) 
    printf("cudagetdeviceproperties return error %s.\n", cudaGetErrorString(error));
  else
    printf("cuda device %d %s compute capability %d.%d.\n", device_id, prop.name,
	   prop.major, prop.minor);
  
  error = cudaMemGetInfo(&free, &total);
  if (error != cudaSuccess) 
    printf("cudamemgetinfo return error %s.\n", cudaGetErrorString(error));
  
  printf("testing heap manager total %ld, free %ld\n", total, free);
  HeapAllocator hm;
  int *ptr = (int*)hm.allocate(g_min_block_bsize);
  hm.deallocate(ptr, g_min_block_bsize);
}

TEST_CASE( "HeapManager::Random", "[random]" ) {
  HeapAllocator hm;
  std::vector<std::pair<float *, size_t>> ptrs;
  for (int i = 0; i != 10; i++) {
    // get size in range [1, 1<<25]
    size_t size = rand() % (1<<24) + 1;  
    float *ptr = (float*)hm.allocate(size);
    ptrs.push_back(std::make_pair(ptr, size));
  }
  for (int i = 0; i != 10; i++) {
    // get size in range [1, 1<<25]
    hm.deallocate(ptrs[i].first, ptrs[i].second);
  }
  
  ptrs.clear();
  for (int i = 0; i != 20; i++) {
    size_t big_size = g_max_block_bsize*2 + rand() % 20000;
    float *ptr = (float*)hm.allocate(big_size);
    ptrs.push_back(std::make_pair(ptr, big_size));
  }
  for (size_t i = 0; i != ptrs.size(); i++) {
    // get size in range [1, 1<<25]
    hm.deallocate(ptrs[i].first, ptrs[i].second);
  }

}

TEST_CASE( "HeapSubBin", "[simple]" ) {
  SubBin bin;
  bin.init(20, g_sub_bin_size);
  size_t index;
  REQUIRE(bin.is_empty());
  assert(bin.insert(21) == nullptr);
  std::vector<float *> ptrs(g_sub_bin_size);
  for (size_t i = 0; i != g_sub_bin_size; i++) {
    ptrs[i] = (float*)bin.insert(19);
    REQUIRE(ptrs[i] != nullptr);
    //    REQUIRE(index == i);
    REQUIRE(bin.count_used() == (i+1));
  }

  REQUIRE(bin.is_full());
  //  REQUIRE(bin.insert(index)== false);
  for (size_t i = 0; i != g_sub_bin_size; i++) {
    REQUIRE(bin.remove(ptrs[i]));
  }
  REQUIRE(bin.is_empty());
}

TEST_CASE( "HeapBin", "[simple]" ) {
  Bin bin;
  int block_size = (2<<10);
  bin.init(block_size);
  std::vector<float *> ptrs;
  int num_sub_bins= 2;
  for (size_t i = 0; i != num_sub_bins*g_sub_bin_size; i++) {
    ptrs.push_back((float*)bin.allocate(19));
    REQUIRE(ptrs[i] != nullptr);
  }
  REQUIRE(bin.num_sub_bins() == num_sub_bins);
  for (auto p = ptrs.begin(); p != ptrs.end(); p++)
    REQUIRE(bin.free(*p, 19));
}

  
#if 0
  std::list<std::unique_ptr<SubBin<g_sub_bin_size>>> bins;
  bins.emplace_back(std::unique_ptr<SubBin<g_sub_bin_size>>(new SubBin<g_sub_bin_size>()));
  bins.back()->init(20);
  int index;
  for (int i = 0; i != g_sub_bin_size; i++) {
    bins.back()->insert(index);
    REQUIRE(index == i);
  }
  bins.back()->insert(index);
  REQUIRE(index==-1);
  for (int i = 0; i != g_sub_bin_size; i++) {
    bins.back()->remove(i);
  }
}
#endif

}
