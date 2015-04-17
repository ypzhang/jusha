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
#define MIN_BLOCK_SHIFT 8  // always aligned to 256 byte
#define MAX_BLOCK_SHIFT 27
#define MIN_BLOCK_BSIZE (1<<8)
#define MAX_BLOCK_BSIZE (1<<27)  // 128MB
#define SUB_BIN_SIZE 8 

#define NUM_BINS (MAX_BLOCK_SHIFT-MIN_BLOCK_SHIFT + 1)

#define DEBUG_HEAP_ALLOC

  template <class T>
  struct cuda_heap_deleter {
    void operator () (T *ptr) const {
#ifdef DEBUG_HEAP_ALLOC
      printf("cudaFreeing %p\n", ptr);
#endif      
      cudaFree((void *)ptr);
    }
  };

  class SubBin {
  public:
#ifdef DEBUG
    ~SubBin() {
      assert(is_empty());
    }
#endif
    bool init(size_t block_size, int BIN_SIZE) {
      m_block_size = block_size;
      size_t bytes_this_sub_bin = block_size * BIN_SIZE;
      char *base = 0;
      cudaMalloc((void**)(&base), bytes_this_sub_bin);
      if (base == 0) return false; // out of memory
#ifdef DEBUG_HEAP_ALLOC
      printf("cudaAllocing %p\n", base);
#endif      
      m_base.reset(base);
      m_used.resize(BIN_SIZE);
      m_used.assign(m_used.size(), false);
      m_used_count = 0;
      return true;
    }

    void *insert() {
      if (is_full()) {
        return nullptr;
      }
      size_t index = 0;
      for (; index != m_used.size();index++) {
        if (!m_used[index]) {
          m_used[index] = true;
          m_used_count++;
          break;
        }
      }
      return (void *)(get_ptr() + index*m_block_size);
    }

    bool remove(void *ptr) {
      if (is_in_range(ptr)) {
        size_t index = ((char*)ptr - get_ptr())/m_block_size;
        assert(m_used.size() > index);
        assert(m_used[index]);
        m_used[index] = false;
        m_used_count--;
        return true;
      }
      return false;
    }


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
      if (m_base.get() == NULL)
        return false;
      if (ptr >= m_base.get() && ptr < m_base.get() + m_block_size * bin_size())
        return true;
      return false;
    }

    size_t m_block_size  = 0;
    std::unique_ptr<char, cuda_heap_deleter<char>> m_base;
    std::vector<bool> m_used;
    size_t m_used_count = 0;
    };

  class Bin {
  public:
    void init(size_t block_size, bool fixing_size=false) {
      m_block_size = block_size;
      m_fixing_block_size = fixing_size;
    }

    bool insert(const size_t &request_size);

  private:
    bool m_fixing_block_size = false;
    size_t m_block_size = 0;
    std::list<std::unique_ptr<SubBin>> m_bins;
  };

  class BinMeta {
  public:
    //    BinMeta(){}
    void set_bin_bsize(size_t bsize) {
      block_size_this_bin = bsize; 
    }
    bool not_initialized() const { return inited==false; }

    void initialize(size_t block_size) {
      block_size_this_bin = block_size;
      // determin num of entries
// #ifdef DEBUG_HEAP_ALLOC
//       printf("cuda malloc return poitner %p.\n", base);
// #endif
      inited = true;
    }
    void *allocate(size_t bsize, size_t bin_bsize, size_t max_bsize)
    {
      if (not_initialized()) {
        initialize(bin_bsize);
      }
      return NULL;
    }

  private:
    
    void grow_sub_bin()
    {
      
    }

    size_t block_size_this_bin = 0;
    bool inited{false};
    bool full{false};
//    std::list<char *> m_pointers;

    int num_sub_bins{0};
    int num_total_entries{0}; // must be less than MAX_ENTRIES_PER_BIN
    int num_used_entries{0}; 
    int last_empty_index{0};
    // counters for debugging/profiling
    int max_used_counter{0}; //
    int overflow_counter{0}; // tracks how many times it is full and new allocation needs to go to upper level
  };

  // returns the nearest bin index according the required bytesize
    void bin_index(const size_t bsize, int &bin_id, size_t &bin_bsize);

  class HeapAllocator : public GpuAllocator {
public:
     HeapAllocator() 
  {
    size_t free;
    cudaMemGetInfo(&free, &m_gpu_capacity_byte);
#ifdef DEBUG_HEAP_ALLOC
    printf("gpu capacity %ld\n", m_gpu_capacity_byte);
#endif
  }

  void *allocate(size_t bsize)
  {
    return allocate_common(bsize);
  }

     void *allocate_common(size_t bsize)
    {
      int bin_id;
      size_t bin_bytesize;
      bin_index(bsize, bin_id, bin_bytesize);
      assert(bin_id < NUM_BINS);
      BinMeta &bin = m_bins[bin_id];
      bin.allocate(bsize, bin_bytesize, m_gpu_capacity_byte);
      return NULL;
    }

  void deallocate(void *ptr)
  {
    deallocate_common(ptr);
  }



private:
   void deallocate_common(void *) 
  {
  }

  size_t m_gpu_capacity_byte;
  BinMeta m_bins[NUM_BINS];
  
  };

  }
