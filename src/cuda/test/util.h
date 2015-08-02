#ifndef JUSHA_CUDA_TEST_UTIL_H
#define JUSHA_CUDA_TEST_UTIL_H

#include <thrust/device_ptr.h>

namespace jusha {
  namespace cuda {

    template <class T> __global__
    void raise_non_equal_flag(const T *lhs, const T *rhs, int *output, int N) {
      int tid = blockDim.x * blockIdx.x + threadIdx.x;
      const int grid_size = gridDim.x * blockDim.x;
      for (; tid < N; tid+=grid_size) {
        if (lhs[tid] != rhs[tid]) *output = 1;
      }
    }

    template <typename T>
    bool is_equal(thrust::device_ptr<T> lhs_first,
                  thrust::device_ptr<T> lhs_last,
                  thrust::device_ptr<T> rhs_first,
                  cudaStream_t stream = 0)
    {
      thrust::device_vector<int> equal(1, 0);
      //      thrust::transform_if(first, last, second_first, output, srt_ns::cuda::is_not_equal_functor<T>());
      // wish had a "reduce" iterator for the output
      
      int N = lhs_last - lhs_first;
      int blocks = GET_BLOCKS(N, jusha::cuda::JCKonst::cuda_blocksize);

      if (blocks > 0)
        raise_non_equal_flag<<<blocks, JCKonst::cuda_blocksize, 0, stream>>>(lhs_first.get(), rhs_first.get(), equal.data().get(), N);
      thrust::device_vector<int> equal_host = equal;
      return (equal_host[0] == 0);
    }

  }
}

#endif
