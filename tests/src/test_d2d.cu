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
#include "./d2d_texture.cu"
#include "./cudamemcpy.cu"
#include "./d2d_direct.cu"
#include "./d2d_unroll.cu"
#include "./d2d_prefetch.cu"

#include "cuda/test/util.h"

enum test_d2d_kernel_type {
  test_d2d_cudamemcpy,
  test_d2d_direct,
  test_d2d_unroll,
  test_d2d_prefetch,
  test_d2d_texture,
  test_d2d_opt
};

#define NOVERIFY

template <typename T>
void test_d2d_cuda(test_d2d_kernel_type type, const char *case_name, size_t byte_size) {
  size_t _size = byte_size/sizeof(T);
  thrust::device_vector<T> d_src_vec(_size);
  thrust::device_vector<T>  d_dst_vec(_size);

#ifndef NOVERIFY
  thrust::sequence(d_src_vec.begin(), d_src_vec.end(), T());
  thrust::sequence(d_dst_vec.begin(), d_dst_vec.end(), 20);
#endif
// #ifndef NOVERIFY 
// {
//   bool equal = jusha::cuda::is_equal(d_src_vec.data(), d_src_vec.data(), d_dst_vec.data());
//   jassert(equal);
// }
// #endif

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
  // case test_d2d_unroll:
  //   d2d_unroll<T, 2>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
  //   break;
  case test_d2d_prefetch:
    d2d_prefetch<T>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  default :
   ;
  }
  jusha::cuda_event_stop(sstm.str().c_str());

#ifndef NOVERIFY
  bool equal = jusha::cuda::is_equal(d_src_vec.data(), d_src_vec.data(), d_dst_vec.data());
  jassert(equal);
#endif
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__  // printf("here 3.\n");
  // thrust::generate(d_dst_vec.begin(), d_dst_vec.end(), rand);
  // printf("here 4.\n");
  // std::stringstream sstm;
  // sstm << case_name << " " << _size;
  // jusha::cuda_event_start(sstm.str().c_str());
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__);
  // printf("here 4.\n");
  
}

template <typename T>
void test_d2d_texture_cuda(test_d2d_kernel_type type, const char *case_name, size_t byte_size) {
  size_t _size = byte_size/sizeof(T);
  thrust::device_vector<T> d_src_vec(_size);
  thrust::device_vector<T>  d_dst_vec(_size);

#ifndef NOVERIFY
  thrust::sequence(d_src_vec.begin(), d_src_vec.end(), T());
  thrust::sequence(d_dst_vec.begin(), d_dst_vec.end(), 20);
#endif
// #ifndef NOVERIFY 
// {
//   bool equal = jusha::cuda::is_equal(d_src_vec.data(), d_src_vec.data(), d_dst_vec.data());
//   jassert(equal);
// }
// #endif

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
  case test_d2d_unroll:
    d2d_unroll<T, 4>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  case test_d2d_prefetch:
    d2d_prefetch<T>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  case test_d2d_texture:
    d2d_texture<T>(thrust::raw_pointer_cast(d_dst_vec.data()), thrust::raw_pointer_cast(d_src_vec.data()), _size);
    break;
  default :
   ;
  }
  jusha::cuda_event_stop(sstm.str().c_str());

#ifndef NOVERIFY
  bool equal = jusha::cuda::is_equal(d_src_vec.data(), d_src_vec.data(), d_dst_vec.data());
  jassert(equal);
#endif
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__  // printf("here 3.\n");
  // thrust::generate(d_dst_vec.begin(), d_dst_vec.end(), rand);
  // printf("here 4.\n");
  // std::stringstream sstm;
  // sstm << case_name << " " << _size;
  // jusha::cuda_event_start(sstm.str().c_str());
  // jusha::check_cuda_error("after sequence", __FILE__, __LINE__);
  // printf("here 4.\n");
  
}

int main (int argc, char **argv) {
  size_t test_size = (2<<26);
  size_t test_max_size = (2<<27);
  if (argc == 3) {
    int min_shift, max_shift;
    min_shift = atoi(argv[1]);
    max_shift = atoi(argv[2]);
    test_size = 2<<min_shift;
    test_max_size = 2<<max_shift;
  }
  //  size_t test_size = (2<<19);
  const int num_runs = 101;
  for (; test_size < test_max_size; test_size<<=1) 
    {
      std::cout << "test size " << test_size << std::endl;
      int runs = num_runs;
      while (runs--)
        test_d2d_cuda<char>(test_d2d_cudamemcpy, "CudaMemcpy_1char", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<short>(test_d2d_cudamemcpy, "CudaMemcpy_2short", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<int>(test_d2d_cudamemcpy, "CudaMemcpy_3int", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<double>(test_d2d_cudamemcpy, "CudaMemcpy_4double", test_size);

#ifdef NOVERIFY
      runs = num_runs;
      while (runs--)
        test_d2d_cuda<float3>(test_d2d_cudamemcpy, "CudaMemcpy_5float3", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<float4>(test_d2d_cudamemcpy, "CudaMemcpy_6float4", test_size);
#endif


      runs = num_runs;
      while (runs--)
        test_d2d_cuda<char>(test_d2d_direct, "DirectKernel_1char", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<short>(test_d2d_direct, "DirectKernel_2short", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<int>(test_d2d_direct, "DirectKernel_3int", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<double>(test_d2d_direct, "DirectKernel_4double", test_size);

#ifdef NOVERIFY
      runs = num_runs;
      while (runs--)
        test_d2d_cuda<float3>(test_d2d_direct, "DirectKernel_5float3", test_size);

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<float4>(test_d2d_direct, "DirectKernel_6float4", test_size);

#endif


      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<int>(test_d2d_unroll, "UnrollKernel_3int", test_size);

#ifdef NOVERIFY

      runs = num_runs;
      while (runs--)
        test_d2d_texture_cuda<float4>(test_d2d_unroll, "Unroll2Kernel_6float4", test_size);

      // runs = num_runs;
      // while (runs--)
      //   test_d2d_cuda<double>(test_d2d_unroll, "Unroll2Kernel_4double", test_size);
#endif

      runs = num_runs;
      while (runs--)
        test_d2d_cuda<double>(test_d2d_prefetch, "PrefetchKernel_4double", test_size);


      runs = num_runs;
      while (runs--)
        test_d2d_cuda<float4>(test_d2d_prefetch, "PrefetchKernel_6float4", test_size);


      runs = num_runs;
      while (runs--)
        test_d2d_texture_cuda<float4>(test_d2d_texture, "TextureKernel_6float4", test_size);


    }
  jusha::cuda_event_print(true) ;
  return 0;
}
