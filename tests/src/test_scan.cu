#include <sstream>
#include <cstdio>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include "timer.h"
#include "cuda/utility.h"
#include "cuda/cuda_config.h"
#include "cuda/bitmap_scan.h"


void test_scan_cuda(const char *case_name, int _size) {
  thrust::device_vector<uint> d_src_vec((_size+31)/32);
  thrust::device_vector<uint> d_dst_vec(_size);

  thrust::fill(d_src_vec.begin(), d_src_vec.end(), 0xFFFFFFFF);

  std::stringstream sstm;
  sstm << case_name << "_" << _size;
  jusha::cuda_event_start(sstm.str().c_str());

  jusha::cuda::exclusive_bitmap_scan(d_src_vec.data(), d_dst_vec.data(), _size);

  jusha::cuda_event_stop(sstm.str().c_str());
}

int main () {

#if 0
  size_t test_size = (2<<18);
  const int num_runs = 1;
  for (; test_size < (2<<26); test_size<<=1) 
    {
      std::cout << "test size " << test_size << std::endl;
      int runs = num_runs;
      while (runs--)
        test_d2d_cuda<char>(test_d2d_cudamemcpy, "CudaMemcpy_char", test_size);

      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<int>(test_d2d_cudamemcpy, "CudaMemcpy_int", test_size);

      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<char>(test_d2d_direct, "DirectKernel_char", test_size);

      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<int>(test_d2d_direct, "DirectKernel_int", test_size);
    }
#endif
  test_scan_cuda("size_100", 100);
  jusha::cuda_event_print() ;
  return 0;
}
