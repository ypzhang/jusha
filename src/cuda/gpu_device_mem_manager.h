#pragma once

namespace jusha {
#define MIN_BLOCK_BSIZE 8
#define MAX_BLOCK_BSIZE (8<<30)  // 8G

#define NUM_BINS (MAX_BLOCK_SIZE/MIN_BLOCK_SIZE + 1)
#define MAX_ENTRIY_RECORD_PER_BIN (4096)



  class BinMeta {
  public:
    BinMeta(size_t block_size) : block_size_this_bin(block_size) {}
    
    bool not_initialized() const { return inited==false; }
    void init(size_t gpu_capacity_bsize) {
      
    }
  private:
    size_t block_size_this_bin(0);
    bool inited(false);
    bool full(false);
    char *base = NULL; //nullptr);
    char *upper = NULL; //(nullptr);
    int num_total_entries(0); // must be less than MAX_ENTRIES_PER_BIN
    int num_used_entries(0); 
    int last_empty_index(0);
    unsigned int entry_used_bit_flag[MAX_ENTRIES_PER_BIN/sizeof(int)];
    // counters for debugging/profiling
    int max_used_counter(0); //
    int overflow_counter(0); // tracks how many times it is full and new allocation needs to go to upper level
  };

  // returns the nearest bin index according the required bytesize
  __host__ __device__ int bin_index(size_t bsize) {
    
  }

class DeviceMemManager {
public:
  __host__ __device__ DeviceManager(int device_id) 
  {}

  __host__ __devivce__ void *allocate(size_t bsize)
  {
    int bin_id = bin_index(bsize);
    assert(bin_id < NUM_BINS);
    BinMeta &bin = m_bins[bin_id];
    if (bin.not_init()) {
      bin.iniitalize();
    }
  }

  __host__ __device__ void deallocate(void *) 
  {
  }
  

private:
  BinMeta m_bins[NUM_BINS];
  
};
