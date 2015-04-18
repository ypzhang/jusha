#pragma once
#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>
#include <list>
#include <cuda_runtime_api.h>
#include "cuda/allocator.h"

namespace jusha {
  // create a bin for each size ranging [2^MIN_BLOCK_SHIFT, 2^MAX_BLOCK_SHIFT], each bin will grow by SUB_BIN_SIZE each time runs out of entries
  // for memory request larger than 2^MAX_BLOCK_SHIFT, track them one by one (SUB_BIN_SIZE is 1), in the last bin
#define DEBUG_HEAP_ALLOC
extern  size_t g_current_alloc_bsize;

  template <class T>
  struct cuda_heap_deleter {
    void operator () (T *ptr) const {
#ifdef DEBUG_HEAP_ALLOC
      float mega = (float)g_current_alloc_bsize/1000000.0;
      printf("cudaFreeing %p current memory usage %f MB\n", ptr, mega);
#endif      
      cudaFree((void *)ptr);
    }
  };

  class SubBin {
  public:
#ifdef DEBUG
    ~SubBin() {
      g_current_alloc_bsize -= m_bytes_this_sub_bin;
      assert(is_empty());
    }
#endif

    bool init(size_t block_size, int BIN_SIZE);
    void *insert(const size_t &request_size);
    bool remove(void *ptr); 

    bool is_full() const {
      return (m_used_count == m_used.size());
    }
    
    bool is_empty() const {
      return (m_used_count == 0);
    }
    
    size_t count_used() const {
      return m_used_count;
    }
    
    char *get_ptr() { return m_base.get(); }

    size_t bin_size() const { return m_used.size(); }
  private:
    bool is_in_range(void *ptr) {
      if (get_ptr() == NULL)
        return false;
      if (ptr >= get_ptr() && ptr < get_ptr() + m_block_size * bin_size())
        return true;
      return false;
    }

    size_t m_block_size  = 0;
    std::unique_ptr<char, cuda_heap_deleter<char>> m_base;
    std::vector<bool> m_used;
    size_t m_used_count = 0;
    size_t m_bytes_this_sub_bin = 0;

  };


  class Bin {
  public:
    void init(size_t block_size, bool fixing_size=true) {
      m_block_size = block_size;
      m_fixing_block_size = fixing_size;
    }

    void *allocate(const size_t &request_size);
    bool free(void *ptr, const size_t &request_size);

    // debug purpose
    size_t num_sub_bins() const { return m_bins.size(); }
  private:
    bool grow_sub_bin(const size_t &request_size);

    bool m_fixing_block_size = true;
    size_t m_block_size = 0;
    std::list<std::unique_ptr<SubBin>> m_bins;
  };

  // returns the nearest bin index according the required bytesize
  void bin_index(const size_t bsize, int &bin_id, size_t &bin_bsize);

  class HeapAllocator : public GpuAllocator {
public:
    HeapAllocator();
#ifdef DEBUG_HEAP_ALLOC
    ~HeapAllocator() {
      float mega = (float)m_max_alloc_size/1000000.0;
      printf("HeapAllocator maximal allocated size %f\n", mega);
  }

#endif

    void *allocate(size_t bsize);

    void deallocate(void *ptr, size_t bsize);
private:
    std::vector<Bin> m_bins;

    size_t m_alloc_size = 0;
    size_t m_max_alloc_size = 0;
  };

}
