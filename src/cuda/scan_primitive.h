#ifndef JUSHA_CUDA_SCAN_PRIMITIVE_H
#define JUSHA_CUDA_SCAN_PRIMITIVE_H

#include "cuda/cuda_config.h"
#include "cuda/primitive.h"
#include <cstdio>
#include <cassert>
#include <thrust/device_vector.h>
#include "cuda/device/global_barrier.cuh"
#include "cuda/device/reduce.cuh"
#include "cuda/device/scan.cuh"


namespace jusha {
  namespace cuda {
    enum ScanType{
      kexclusive,
      kinclusive
    };

    __global__ void barrier(volatile int *counter)
    {
      //      if (blockIdx.x ==0 && threadIdx.x == 0) printf("max thread %d block %d counter %d. \n", 
      //                                                     blockDim.x, gridDim.x, *counter);
      for (int i = 0; i != 10000; i++)  
        {
        device::global_barrier(counter);
        //        if (blockIdx.x ==0 && threadIdx.x == 0) printf("count %d.\n", *counter);
        device::global_barrier(counter);
        //             if (blockIdx.x ==0 && threadIdx.x == 0) printf("count %d.\n", *counter);

        }
    }

    template <typename T, int bsize, bool exclusive>
    __global__  void scan_primitive(const T *__restrict__ input, T * __restrict__ output, size_t N, volatile T *mailbox,int *counter)
    {
      int elem_per_batch = bsize * gridDim.x * 4;
      //      int batches = (N + elem_per_batch - 1)/elem_per_batch;
      int batches = 0;
      
      if (blockIdx.x == 0 && threadIdx.x == 0)
        *mailbox = T();
      //      __shared__ volatile T sh_storage[bsize];
      __shared__ volatile T sh_mem[bsize];
      __shared__ T carry_out;
      int index = bsize * blockIdx.x + threadIdx.x;
      for (int batch = 0; batch != batches; batch++, index += elem_per_batch) {
        // load a batch
        T elem;
        if (index < N) {
          elem = input[index];
        }

        // T sum = block_reduce<bsize, T>(elem, sh_mem);

         device::g_barrier_phase1(counter);
        // if (threadIdx.x == 0) {
        //   carry_out = (*mailbox);
        //   (*mailbox) = carry_out + elem;
        // }
         device::g_barrier_phase2(counter);

        // block_scan<bsize, T>(elem, carry_out,sh_mem);

        if (index < N)
          output[index] = sh_mem[threadIdx.x];

      }
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
      get_gpu_property(property); 
      int BS = property.maxThreadsPerBlock;
      int blocks = property.multiProcessorCount;
      barrier<<<blocks, BS>>>(thrust::raw_pointer_cast(m_counter.data()));
    }

    template <ScanType type, typename T>
    void ScanPrimitive<type, T>::scan(const T *in_begin, const T *in_end, T *output)
    {
      cudaDeviceProp property;
      get_gpu_property(property); 
      
      size_t N = in_end - in_begin;
      if (!N)  return;

      thrust::device_vector<T> mailbox(1);
      //      thrust::fill(mailbox.begin(), mailbox.end(), T());
      int BS = property.maxThreadsPerBlock;
      int blocks = (N + BS - 1)/BS; 
      blocks = std::min(blocks, property.multiProcessorCount);
      
      if (BS == 1024)
        scan_primitive<T, 1024, type==kexclusive><<<blocks, BS>>>(in_begin, output, N, thrust::raw_pointer_cast(mailbox.data()),thrust::raw_pointer_cast(m_counter.data()));
      else if (BS == 512)
        scan_primitive<T, 512, type==kexclusive><<<blocks, BS>>>(in_begin, output, N, thrust::raw_pointer_cast(mailbox.data()),thrust::raw_pointer_cast(m_counter.data()));
      else
        assert(0);
      
      
    }

  }


}

#endif
