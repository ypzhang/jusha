#include "heap_allocator.h"

namespace jusha {
   void bin_index(const size_t bsize, int &bin_id, size_t &bin_bsize) 
  {
    if (bsize <= MIN_BLOCK_BSIZE) {
      bin_id = 0;
      bin_bsize = MIN_BLOCK_BSIZE;
      return;
    }
    if (bsize > MAX_BLOCK_BSIZE) {
      //      fprintf(stderr, "Requesting memory size %ld exceeding maximal block bsize %ld!\n", bsize, MAX_BLOCK_BSIZE);
      bin_id = NUM_BINS-1;
      bin_bsize = 0;
      return;
    }
    size_t bit = MAX_BLOCK_BSIZE;
    bin_id = NUM_BINS-1; 
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
