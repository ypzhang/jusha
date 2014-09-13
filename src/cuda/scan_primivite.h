#ifndef JUSHA_CUDA_SCAN_PRIMITIVE_H
#define JUSHA_CUDA_SCAN_PRIMITIVE_H

#include "cuda/cuda_config.h"
#include "cuda/primitive.h"
#include <cstdio>
#include <thrust/device_vector.h>
#include "cuda/device/global_barrier.h"

namespace jusha {
  namespace cuda {
    enum ScanType{
      kexclusive,
      kinclusive
    };

    __global__ void barrier(volatile int *counter)
    {
      if (blockIdx.x ==0 && threadIdx.x == 0) printf("max thread %d block %d counter %d. \n", 
                                                     blockDim.x, gridDim.x, *counter);

      device::global_barrier(counter);
    }

    template <ScanType type, typename T>
    class ScanPrimitive: public Primitive {
    public:
      ScanPrimitive():m_counter(1) {
        thrust::fill(m_counter.begin(), m_counter.end(), 0);
      }
      virtual void run();
      void scan(const T *in_begin, const T *in_end, T *output);
    private:
      //
      thrust::device_vector<int> m_counter;
    };
    
    template <ScanType type, typename T>
    void ScanPrimitive<type, T>::run() {
      cudaDeviceProp property;
      jusha::cuda::get_cuda_property(property);
      int BS = property.maxThreadsPerBlock;
      int blocks = property.multiProcessorCount;
      barrier<<<blocks, BS>>>(thrust::raw_pointer_cast(m_counter.data()));
    }

    template <ScanType type, typename T>
    void ScanPrimitive<type, T>::scan(const T *in_begin, const T *in_end, T *output)
    {
      
    }

  }


}

#endif
