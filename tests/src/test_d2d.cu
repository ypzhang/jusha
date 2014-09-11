#include <sstream>
#include <cstdio>
#include <algorithm>

#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include "timer.h"
#include "utility.h"
#include "cuda/utility.h"
#include "cuda/cuda_config.h"

//#include "cuda/d2d_copy.h"

// include different kernel implementations
#include "cuda/cuda_types.h"
#include "./cudamemcpy.cu"
#include "./d2d_direct.cu"
#include "cuda/test/util.h"

enum test_d2d_kernel_type {
  test_d2d_cudamemcpy,
  test_d2d_direct,
  test_d2d_opt
};

template <typename T>
void test_d2d_cuda(test_d2d_kernel_type type, const char *case_name, size_t _size) {
  thrust::device_vector<T> d_src_vec(_size);
  thrust::device_vector<T>  d_dst_vec(_size);

  thrust::sequence(d_src_vec.begin(), d_src_vec.end(), T());
  thrust::sequence(d_dst_vec.begin(), d_dst_vec.end(), 1);

  std::stringstream sstm;
  sstm << case_name << "_" << _size*sizeof(T);
  jusha::cuda_event_start(sstm.str().c_str());
  
  switch (type) {
  case test_d2d_cudamemcpy:
    d2d_cudamemcpy<T>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  case test_d2d_direct:
    d2d_direct(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  default :
   ;
  }
  jusha::cuda_event_stop(sstm.str().c_str());

  bool equal = is_equal(d_src_vec.begin(), d_src_vec.end(), d_dst_vec.begin());
  jusha::jassert(is_equal);
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__  // printf("here 3.\n");
  // thrust::generate(d_dst_vec.begin(), d_dst_vec.end(), rand);
  // printf("here 4.\n");
  // std::stringstream sstm;
  // sstm << case_name << " " << _size;
  // jusha::cuda_event_start(sstm.str().c_str());
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__);
  // printf("here 4.\n");
  
}

int main () {
  size_t test_size = (2<<19);
  const int num_runs = 11;
  for (; test_size < (2<<27); test_size<<=1) 
    {
      std::cout << "test size " << test_size << std::endl;
      int runs = num_runs;
      while (runs--)
        test_d2d_cuda<char>(test_d2d_cudamemcpy, "CudaMemcpy_char", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<int>(test_d2d_cudamemcpy, "CudaMemcpy_int", test_size);

      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<jusha::cuda::Float3>(test_d2d_cudamemcpy, "CudaMemcpy_float3", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<double>(test_d2d_cudamemcpy, "CudaMemcpy_double", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<char>(test_d2d_direct, "DirectKernel_char", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<int>(test_d2d_direct, "DirectKernel_int", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<double>(test_d2d_direct, "DirectKernel_double", test_size);
      
    }
  jusha::cuda_event_print() ;
  return 0;
}
