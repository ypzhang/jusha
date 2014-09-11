#ifndef JUSHA_CUDA_BITMAP_SCAN_H
#define JUSHA_CUDA_BITMAP_SCAN_H

#include <thrust/device_ptr.h>

namespace jusha {
  namespace cuda {
    void exclusive_bitmap_scan(thrust::device_ptr<unsigned int> in_begin, 
                               thrust::device_ptr<unsigned int> out_begin,
                               int N);

    void inclusive_bitmap_scan(thrust::device_ptr<unsigned int> in_begin, 
                               thrust::device_ptr<unsigned int> out_begin,
                               int N);
  }
}

#endif
