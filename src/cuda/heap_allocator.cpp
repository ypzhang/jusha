#include <algorithm>
#include "heap_allocator.h"

namespace jusha {
  size_t g_sub_bin_size = 4;

  size_t g_min_block_shift = 8;
  size_t g_min_block_bsize = (1<<8);
  size_t g_max_block_shift = 27;
  size_t g_max_block_bsize = (1<<27);
  size_t g_num_bins = 27-8+1;

  size_t g_current_alloc_bsize = 0;
  /*************************************************************/
  bool SubBin::init(size_t block_size, int BIN_SIZE) {
    m_block_size = block_size;
    m_bytes_this_sub_bin = block_size * BIN_SIZE;
    
    char *base = 0;
    cudaMalloc((void**)(&base), m_bytes_this_sub_bin);
    if (base == 0) return false; // out of memory
#ifdef DEBUG_HEAP_ALLOC
    g_current_alloc_bsize += m_bytes_this_sub_bin;
    float mega = (float)g_current_alloc_bsize/1000000.0;
    printf("cudaAllocing %p block_size %ld, bin size %ld global size %f MB\n", base, block_size, m_bytes_this_sub_bin, mega);
#endif      
    m_base.reset(base);
    m_used.resize(BIN_SIZE);
    m_used.assign(m_used.size(), false);
    m_used_count = 0;
    return true;
  }

  void *SubBin::insert(const size_t &request_size) {
    if (m_block_size < request_size) 
      return nullptr;
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

  bool SubBin::remove(void *ptr) {
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

  /*************************************************************/
  bool Bin::grow_sub_bin(const size_t &request_size) {
    SubBin *bin = new SubBin();
    if (!m_fixing_block_size) {
      if (!bin->init(request_size, 1))  {
        delete bin;
        return false;
      }
    }else {
      assert(request_size <= m_block_size);
      if (!bin->init(m_block_size, g_sub_bin_size))  {
        delete bin;
        return false;
      }
    }
    m_bins.emplace_back(std::unique_ptr<SubBin>(bin));
    return true;
  }
  
  void *Bin::allocate(const size_t &request_size) {
    void *ptr = nullptr;
    for (auto i = m_bins.begin(); i != m_bins.end(); i++) {
      ptr = i->get()->insert(request_size);
      if (ptr != nullptr) return ptr;
    }
    if (m_fixing_block_size) 
      assert(request_size <= m_block_size);
    if (!grow_sub_bin(request_size)) 
      return nullptr;
    return m_bins.back()->insert(request_size);
  }

  bool Bin::free(void *ptr, const size_t &request_size) {
    for (auto i = m_bins.begin(); i != m_bins.end(); i++) {
      if (i->get()->remove(ptr)) {
        if (i->get()->is_empty()) {
          m_bins.erase(i);
        }
        return true;
      }
    }
    return false;
  }

  /*************************************************************/
  HeapAllocator::HeapAllocator() {
    for (size_t bsize = g_min_block_bsize; bsize < g_max_block_bsize; bsize<<=1)
      {
        m_bins.push_back(Bin());
        m_bins.back().init(bsize, true);
      }
    m_bins.push_back(Bin());
    // the last bin has variable size, for big buffers
    m_bins.back().init(0, false);
    assert(m_bins.size() == g_num_bins);
  }

  void *HeapAllocator::allocate(size_t bsize)
  {
    int bin_id(0);
    size_t bin_bsize(0);
    bin_index(bsize, bin_id, bin_bsize);
    assert(bin_id < (int)m_bins.size());
    void *ptr =  m_bins[bin_id].allocate(bsize);
    if (ptr == nullptr) {
      fprintf(stderr, "Error!!! Running out of GPU memory!!!!! Do defragmentating (TODO).\n");
      abort();
    }
#ifdef DEBUG_HEAP_ALLOC
    printf("heap allocator alloc return %p for size %ld\n", ptr, bsize);
#endif
    m_alloc_size += bsize;
    m_max_alloc_size = (std::max)(m_max_alloc_size, m_alloc_size);
    return ptr;
  }
  
  void HeapAllocator::deallocate(void *ptr, size_t bsize)
  {
    int bin_id(0);
    size_t bin_bsize(0);
    bin_index(bsize, bin_id, bin_bsize);
    assert(bin_id < (int)m_bins.size());
    m_bins[bin_id].free(ptr, bsize);
    m_alloc_size -= bsize;
  }
  /*************************************************************/
   void bin_index(const size_t bsize, int &bin_id, size_t &bin_bsize) 
  {
    if (bsize <= g_min_block_bsize) {
      bin_id = 0;
      bin_bsize = g_min_block_bsize;
      return;
    }
    if (bsize > g_max_block_bsize) {
      //      fprintf(stderr, "Requesting memory size %ld exceeding maximal block bsize %ld!\n", bsize, g_max_block_bsize);
      bin_id = g_num_bins-1;
      bin_bsize = 0;
      return;
    }
    size_t bit = g_max_block_bsize;
    bin_id = g_num_bins-1; 
    while ((bit & bsize) == 0) {
      bit >>= 1;
      --bin_id;
    }
    bin_bsize = bit;
    if (bsize & (bit-1)){
      bin_bsize <<= 1;
      bin_id++;
    }
  }

}
