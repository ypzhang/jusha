#pragma once

#include <cuda.h>
#include <iostream>
#include <fstream>
#include <string>
//#include "CudppPlanFactory.h"

#include <thrust/device_ptr.h>
//#include "SegmentArrayKernel.h"
#include <curand.h>
#include "../utility.h"
#include "./external_lib_wrapper.h"
#include "./heap_manager.h"
//#define USE_SHARED_PTR 1

#ifdef USE_SHARED_PTR
#include "boost/shared_ptr.hpp"
#endif
//extern cudaDeviceProp gDevProp;


namespace jusha {
  extern HeapManager gHeapManager;
  namespace cuda {  
    #define MAX_PRINT_SIZE 32

    template <class T>
      class MirroredArray{
    public:
      explicit MirroredArray(int size = 0):
      mSize(size),
        mCapacity(size),
        hostBase(),
        dvceBase(),
        isCpuValid(false),
        isGpuValid(false),
        gpuAllocated(false),
        cpuAllocated(false)
          {
          }
      
      ~MirroredArray() {
        if (dvceBase)
          gHeapManager.NeFree(GPU_HEAP, dvceBase);
        if (hostBase)
          gHeapManager.NeFree(CPU_HEAP, hostBase);

      }
      
      // Copy Constructor
      MirroredArray(const MirroredArray<T> &rhs) 
        {
          copy(rhs);
        }
      
      /* Init from raw pointers
       */
      void init(const T *ptr, size_t _size) {
        resize(_size);
        T *g_ptr = getGpuPtr();
        cudaMemcpy(g_ptr, ptr, sizeof(T) * _size, cudaMemcpyDefault);
      }
      // dangerous, used at your own risk!
      // does not check size consistency
      void setGpuPtr(T *ptr, bool needToFree=false)
      {
#if USE_SHARED_PTR
        if (!needToFree)
          {
            boost::shared_ptr<T> newDvceBase((T*)ptr, EmptyDeviceDeleter);
            dvceBase = newDvceBase;
          }
        else
          {
            boost::shared_ptr<T> newDvceBase((T*)ptr, GpuDeviceDeleter);
            dvceBase = newDvceBase;
          }
#else
        if (dvceBase)
          gHeapManager.NeFree(GPU_HEAP, dvceBase);
        dvceBase = ptr;
#endif
        isCpuValid = false;
        isGpuValid = true;
        gpuAllocated = true;
      }
      
      void setPtr(T *ptr)
      {
#if USE_SHARED_PTR
        hostBase.reset(ptr);
#else
        if (hostBase)
          gHeapManager.NeFree(CPU_HEAP, hostBase);
        hostBase = ptr;
#endif
        isGpuValid = false;
        isCpuValid = true;
        cpuAllocated = true;
      }
      
      MirroredArray<T> &operator=(const MirroredArray<T> &rhs)
        {
          copy(rhs);
          return *this;
        }
      
      
      // deep copy
      void clone(MirroredArray<T> &dst) const
      {
        dst.resize(size());
        if (isGpuValid)
          {
            cudaMemcpy(dst.getGpuPtr(), getReadOnlyGpuPtr(), sizeof(T)*mSize, cudaMemcpyDeviceToDevice);
          }
        if (isCpuValid)
          {
            memcpy(dst.getPtr(), getReadOnlyPtr(), sizeof(T)*mSize);
          }
        
        dst.isCpuValid = isCpuValid;
        dst.isGpuValid = isGpuValid;
        dst.gpuAllocated = gpuAllocated;
        dst.cpuAllocated = cpuAllocated;
      }
      
      /* swap info between two arrays */
      void swap(MirroredArray<T> &rhs)
      {
        MirroredArray<T> temp(rhs);
        //    temp.clone(rhs);
        rhs = *this;
        *this = temp;
        //    temp.hostBase = 0;
        //    temp.dvceBase = 0;
        temp.cpuAllocated = false;
        temp.gpuAllocated = false;
      }
      
      int size() const
      {
        return mSize;
      }

      void resize(int size) 
      {
#ifdef _DEBUG_
        std::cout << "new size " << size << " old size " << mSize << std::endl;
#endif
        if (size <= mCapacity)
          {
            mSize = size;
          }
        else // need to reallocate
          {
#if USE_SHARED_PTR 
            if (gpuAllocated)
              {
                boost::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(size*sizeof(T)), GpuDeviceDeleter);
                if (isGpuValid)
                  {
                    cudaError_t error = cudaMemcpy(newDvceBase.get(), dvceBase.get(), mSize*sizeof(T), cudaMemcpyDeviceToDevice);
                    //            std::cout << "memcpy d2d size:" << mSize*sizeof(T)  << std::endl;
                    assert(error == cudaSuccess);
                  }
                dvceBase = newDvceBase;
              }
            if (cpuAllocated)
              {
                boost::shared_ptr<T> newHostBase((T*)GpuHostAllocator(size*sizeof(T)), GpuHostDeleter);
                if (isCpuValid)
                  memcpy(newHostBase.get(), hostBase.get(), mSize*sizeof(T));
                hostBase = newHostBase;            
              }
            mSize = size;
            mCapacity = size;
#else
            T *newDvceBase(0);
            T *newHostBase(0);
            if (gpuAllocated)
              {
                // cutilSafeCall(cudaMalloc((void**) &newDvceBase, size * sizeof(T)));
                gHeapManager.NeMalloc(GPU_HEAP, (void**)&newDvceBase, size * sizeof(T));
                assert(newDvceBase);
                // TODO memcpy 
              }
            if (cpuAllocated)
              {
                gHeapManager.NeMalloc(CPU_HEAP, (void**)&newHostBase, size*sizeof(T));
                //            newHostBase = (T*)malloc(size * sizeof(T));
                assert(newHostBase);
              }
            if (isCpuValid)
              {
                memcpy(newHostBase, hostBase, mSize*sizeof(T));
              }
            if (isGpuValid)
              {
                cudaError_t error = cudaMemcpy(newDvceBase, dvceBase, mSize*sizeof(T), cudaMemcpyDeviceToDevice);
                //            std::cout << "memcpy d2d size:" << mSize*sizeof(T)  << std::endl;
                assert(error == cudaSuccess);
              }
            if (hostBase)
              gHeapManager.NeFree(CPU_HEAP, hostBase);
            //          free(hostBase);
            if (dvceBase)
              {
                gHeapManager.NeFree(GPU_HEAP, dvceBase);
                //            cutilSafeCall(cudaFree(dvceBase));
              }
#ifdef _DEBUG_
            std::cout << "free at resize:" << std::hex << dvceBase << std::endl;
#endif
            hostBase = newHostBase;
            dvceBase = newDvceBase;
            mSize = size;
            mCapacity = size;
#endif
          }
      }

      void zero()
      {
        enableGpuWrite();
        cudaMemset((void *)dvceBase, 0, sizeof(T)*mSize);
      }


      const T *getReadOnlyPtr() const
      {
        enableCpuRead();
        return hostBase;
      }

      T *getPtr()
      {
        enableCpuWrite();
        return hostBase;
      }

      const T *getReadOnlyGpuPtr() const
      {
        enableGpuRead();
        return dvceBase;
      }

      T *getGpuPtr()
      {
        enableGpuWrite();
        return dvceBase;
      }
  
      T &operator[](int index)
        {
          assert(index < mSize);
          T *host = getPtr();
          return host[index];
        }

      const T &operator[](int index) const
      {
        assert(index < mSize);
        const T *host = getReadOnlyPtr();
        return host[index];
      }

      //friend 
      //MirroredArray<T> &operator-(const MirroredArray<T> &lhs, const MirroredArray<T> &rhs);

      /* only dma what's needed, instead of the whole array */
      const T getElementAt(const int index) const
      {
        assert(index < mSize);
        assert(isCpuValid || isGpuValid);
        if (isCpuValid)
          //          return hostBase.get()[index];
          return hostBase[index];
        T ele; 
        allocateCpuIfNecessary();
        //        cudaError_t error = cudaMemcpy(&ele, dvceBase.get()+index, sizeof(T),cudaMemcpyDeviceToHost); 
        cudaError_t error = cudaMemcpy(&ele, dvceBase+index, sizeof(T),cudaMemcpyDeviceToHost); 
        //    std::cout << "memcpy d2h size:" << sizeof(T)  << std::endl;
        assert(error == cudaSuccess);
        return ele;
      }

      void setElementAt(T &value, const int index)
      {
        assert(index < mSize);
        assert(isCpuValid || isGpuValid);
        if (isCpuValid)
          //          hostBase.get()[index] = value;
          hostBase[index] = value;
        if (isGpuValid)
          {
            //            cudaError_t error = cudaMemcpy(dvceBase.get()+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
            cudaError_t error = cudaMemcpy(dvceBase+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
            assert(error == cudaSuccess);
          }
      }

      void randomize() 
      {
        RandomWrapper<CURAND_RNG_PSEUDO_MTGP32, T> rng;
        rng.apply(getGpuPtr(), mSize);
      }

      // use sequence in thrust
      void sequence(int dir)
      {
        T *ptr = getPtr();
        if (dir == 0) //ascending
          {
            for (int i = 0; i != mSize; i++)
              *ptr++ = (T) i;
          }
        else // descending
          {
            for (int i = 0; i != mSize; i++)
              *ptr++ = (T) (mSize-i);
          }
      }

      // for DEBUG purpose
      void print(const char *header=0, int print_size = MAX_PRINT_SIZE) const
      {
        const T *ptr = getReadOnlyPtr();
        int size = mSize > print_size? print_size: mSize;
        if (header)
          std::cout << header << std::endl;
        for (int i = 0; i != size; i++)
          std::cout << " " <<  ptr[i] ; 

        std::cout << std::endl;
      }
  
      /* math functions on mirrored array */
      /*  void reduce(CudppPlanFactory *factory, MirroredArray<T> &total, uint op)
          {
          // todo: test a threshold to determine whether do on CPU or GPU
          // currently do it on GPUs always
    
          }*/

      void saveToFile(const char *filename) const
      {
        //      assert(0);
        std::ofstream file;
        file.open(filename);
        assert(file);
        file << "size is " << size() << "\n";
        //      int size = printSize < size()? printSize:size();
        const T * ptr = getReadOnlyPtr();
        for (int i = 0; i != size(); i++)
          file << ptr[i] << "(" << i << ")" << " ";
        file.close();
      }


      bool isSubsetOf(const MirroredArray<T> &super)
      {
        const T *myBase = getReadOnlyPtr();
        const T *superBase = super.getReadOnlyPtr();
        size_t mySize = size();
        size_t superSize = super.size();
        if (mySize > superSize)
          return false;
    
        for (int i = 0; i != mySize; i++)
          {
            bool found = false;
            for (int j = 0; j != superSize; j++)
              {
                if (superBase[j] == myBase[i])
                  {
                    //                std::cout << "found " << myBase[i];
                    found = true;
                
                  }
              }
            if (!found)
              {
                std::cout << "not finding " << myBase[i] << " in super array.\n";
                return false;
              }
          }
        return true;
      }

      bool isAllZero() const
      {
        const T *buffer = getReadOnlyPtr();
        bool allzero = true;
        for (int i = 0; i < mSize; i++)
          {
            if (buffer[i] != 0)
              {
                std::cout << "the " << i << "th value " << buffer[i] << " is not zero " << std::endl;
                allzero = false;
                break;
              }
          }
        return allzero;
      }

      bool isEqualTo(const MirroredArray<T> &rhs) const
      {
        if (rhs.size() != size()) return false;
        const T *buffer = getReadOnlyPtr();
        const T *buffer2 = rhs.getReadOnlyPtr();
        bool equal = true;
        for (int i = 0; i < mSize; i++)
          {
            if (buffer[i] != buffer2[i])
              {
                equal = false;
                break;
              }
          }
        return equal;
      }


      bool isFSorted(int begin, int end) const
      {
        const T *buffer = getReadOnlyPtr();
        bool sorted = true;
        if (begin == -1) begin = 0;
        if (end == -1) end = mSize;
        for (int i = begin; i < end-1; i++)
          {
            if (buffer[i] > buffer[i+1])
              {
                std::cout << "the " << i << "th value " << buffer[i] << " is bigger than " << buffer[i+1] << std::endl;
                sorted = false;
                break;
              }
          }
        return sorted;
      }


    inline typename thrust::device_ptr<T> begin()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()); 
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> end()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()+mSize);
    }

    /*! \brief Return the iterator to the first element in the srt::vector */
    inline typename thrust::device_ptr<T> begin() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getReadOnlyGpuPtr()));
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> end() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    {
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getReadOnlyGpuPtr()+mSize));
    }


    private:
      inline void allocateCpuIfNecessary()  const
      {
        if (!cpuAllocated && mSize)
          {
#if USE_SHARED_PTR
            boost::shared_ptr<T> newHostBase((T*)GpuHostAllocator(mSize*sizeof(T)), GpuHostDeleter);
            hostBase = newHostBase;
#else
            gHeapManager.NeMalloc(CPU_HEAP, (void**)&hostBase, mSize*sizeof(T));
            assert(hostBase);
#endif
            //        hostBase = (T *)malloc(mSize * sizeof(T));

            cpuAllocated = true;
          }
      }

      inline void allocateGpuIfNecessary() const
      {
        if (!gpuAllocated && mSize)
          {
            //        cutilSafeCall(cudaMalloc((void**) &dvceBase, mSize * sizeof(T)));
#if USE_SHARED_PTR
            boost::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(mSize*sizeof(T)), GpuDeviceDeleter);
            assert(newDvceBase != 0);
            dvceBase = newDvceBase;
#else
            gHeapManager.NeMalloc(GPU_HEAP, (void**)&dvceBase, mSize * sizeof(T));
            assert(dvceBase);
#endif
            gpuAllocated = true;
          }
      }

      inline void enableGpuRead() const
      {
        if (!isGpuValid)
          {
            allocateGpuIfNecessary();
            fromHostToDvceIfNecessary();
            isGpuValid = true;
          }
      }

      inline void enableGpuWrite() const
      {
        if (!isGpuValid)
          {
            allocateGpuIfNecessary();
            fromHostToDvceIfNecessary();
            isCpuValid = false;
            isGpuValid = true;
          }
      }

      inline void enableCpuRead() const
      {
        if (!isCpuValid)
          {
            allocateCpuIfNecessary();
            fromDvceToHostIfNecessary();
            isCpuValid = true;
          }
      }

      inline void enableCpuWrite() const
      {
        if (!isCpuValid)
          {
            allocateCpuIfNecessary();
            fromDvceToHostIfNecessary();
            isCpuValid = true;
            isGpuValid = false;
          }
      }
  
      inline void fromHostToDvceIfNecessary() const
      {
        if (isCpuValid && !isGpuValid)
          {
#ifdef _DEBUG_
            std::cout << "sync mirror array from host 0x" << std::hex << hostBase.get() << " to device 0x" << dvceBase.get() << " size(" << mSize << "); \n";
#endif
            //            cudaError_t error = cudaMemcpy(dvceBase.get(), hostBase.get(), mSize* sizeof(T), cudaMemcpyHostToDevice);
            cudaError_t error = cudaMemcpy(dvceBase, hostBase, mSize* sizeof(T), cudaMemcpyHostToDevice);
            //        std::cout << "memcpy h2d size:" << mSize*sizeof(T)  << std::endl;
            assert(error == cudaSuccess);
          }
      }

      inline void fromDvceToHostIfNecessary() const
      {
        if (isGpuValid && !isCpuValid)
          {
            //            check_cuda_error("before memcpy", __FILE__, __LINE__);
#ifdef _DEBUG_
            std::cout << "sync mirror array from device 0x" << std::hex << dvceBase.get() << " to host 0x" << hostBase.get() << " size(" << mSize << "); \n";
            /* assert(gHeapManager.find(CPU_HEAP, hostBase.get()) >= (mSize * (int)sizeof(T))); */
            /* assert(gHeapManager.find(GPU_HEAP, dvceBase.get()) >= (mSize * (int)sizeof(T))); */
            assert(gHeapManager.find(CPU_HEAP, hostBase) >= (mSize * (int)sizeof(T)));
            assert(gHeapManager.find(GPU_HEAP, dvceBase) >= (mSize * (int)sizeof(T)));
#endif
            //            cudaError_t error = cudaMemcpy(hostBase.get(), dvceBase.get(), mSize * sizeof(T),cudaMemcpyDeviceToHost);
            cudaError_t error = cudaMemcpy(hostBase, dvceBase, mSize * sizeof(T),cudaMemcpyDeviceToHost);
            //        std::cout << "memcpy d2h size:" << mSize*sizeof(T)  << std::endl;
            assert(error == cudaSuccess);
          }
      }

      size_t mSize;
      int mCapacity;
#if USE_SHARED_PTR
      mutable boost::shared_ptr<T> hostBase;
      mutable boost::shared_ptr<T> dvceBase;
#else
      mutable T *hostBase;
      mutable T *dvceBase;
#endif
      mutable bool isCpuValid;
      mutable bool isGpuValid;
      mutable bool gpuAllocated;
      mutable bool cpuAllocated;

      void copy(const MirroredArray<T> &rhs)
      {
        mSize = rhs.mSize;
        mCapacity = rhs.mCapacity;
        hostBase = rhs.hostBase;
        dvceBase = rhs.dvceBase;
        isCpuValid = rhs.isCpuValid;
        isGpuValid = rhs.isGpuValid;
        gpuAllocated = rhs.gpuAllocated;
        cpuAllocated = rhs.cpuAllocated;
      }

      static curandGenerator_t curandGen;
  
    };

  } // cuda


  // aliasing C++11 feature
  /* template <typename T> */
  /*   using JVector = cuda::MirroredArray<T>; */
  #define JVector jusha::cuda::MirroredArray
} // jusha

/*
template <class T>
__global__
void  arrayMinusKernel(T *dst, const T * lhs, const T *rhs, int size)
{
  GET_GID
  OUTER_FOR 
  {
    dst[curId] = lhs[curId] - rhs[curId];
  }
  
}
*/
/*template <class T>
MirroredArray<T> &operator-(const MirroredArray<T> &lhs, const MirroredArray<T> &rhs)
{
  MirroredArray<T> result(lhs.size());
  assert(lhs.size() == rhs.size());
  int size = lhs.size();
  cudaDeviceProp *devProp = &gDevProp;  
  arrayMinusKernel<<<KERNEL_SETUP(size)>>>(result.getGpuPtr(), lhs.getReadOnlyGpuPtr(), rhs.getReadOnlyGpuPtr(), size);
  return result;
  }*/
