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
#include <cstddef>
//#include <memory>

//extern cudaDeviceProp gDevProp;


namespace jusha {
  extern HeapManager gHeapManager;
  namespace cuda {  
    #define MAX_PRINT_SIZE 32

    template <typename T>
    void fill(T *begin, T *end, const T & val);

    template <typename T>
    void fill(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, const T&val);
    
    enum class ArrayType
    { 
      CPU_ARRAY = 0,
        GPU_ARRAY = 1
        } ;

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
    cpuAllocated(false),
	isGpuArray(false),
    gpuNeedToFree(true),
    cpuNeedToFree(true)
      {
      }

    // explicit MirroredArray(bool gpuArray):
    //     mSize(0),
    //     mCapacity(0),
    //     hostBase(),
    //     dvceBase(),
    //     isCpuValid(false),
    //     isGpuValid(false),
    //     gpuAllocated(false),
    //     cpuAllocated(false),
    //   isGpuArray(gpuArray? true:  false)
    //   {
        
    //   }      

      // explicit MirroredArray(ArrayType t):
      //   mSize(0),
      //   mCapacity(0),
      //   hostBase(),
      //   dvceBase(),
      //   isCpuValid(false),
      //   isGpuValid(false),
      //   gpuAllocated(false),
      //   cpuAllocated(false),
      // isGpuArray( t == ArrayType::GPU_ARRAY? true:  false)
      // {
        
      // }      

      ~MirroredArray() {
        //        printf("inside destructor size %ld %p.\n", mSize, dvceBase);
        destroy();
      }
      
      void setGpuArray() {
        isGpuArray = true;
      }

      void setCpuArray() {
        isGpuArray = false;
      }

      bool IsGpuArray() const {
        return isGpuArray;
      }
      
      void destroy() {
        if (dvceBase && gpuNeedToFree) {
#ifdef USE_SHARED_PTR
          gHeapManager.NeFree(GPU_HEAP, dvceBase.get());
#else
          gHeapManager.NeFree(GPU_HEAP, dvceBase, size()*sizeof(T));
#endif
          dvceBase = NULL;
        }
        if (hostBase && mCapacity > 0 && cpuNeedToFree) {
          gHeapManager.NeFree(CPU_HEAP, hostBase, size()*sizeof(T));
          hostBase = NULL;	  
        }
        init_state();
      }
      // Copy Constructor
      MirroredArray(const MirroredArray<T> &rhs) 
      {
        init_state();
        //          printf("in copy constructore %d %d %d %d.\n", isGpuValid, gpuAllocated, isCpuValid, cpuAllocated);          
        deep_copy(rhs);
      }

      explicit MirroredArray(const std::vector<T> &rhs)
      {
        init_state();
        resize(rhs.size());
        cudaMemcpy(getOverwriteGpuPtr(), rhs.data(), size()*sizeof(T), cudaMemcpyDefault);
      }

      void operator=(const std::vector<T> &rhs)
      {
        //        init_state();
        resize(rhs.size());
        cudaMemcpy(getOverwriteGpuPtr(), rhs.data(), size()*sizeof(T), cudaMemcpyDefault);
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
      void setGpuPtr(T *ptr,  int _size, bool needToFree=false)
      {
#if USE_SHARED_PTR
        if (!needToFree)
          {
            std::shared_ptr<T> newDvceBase((T*)ptr, EmptyDeviceDeleter);
            dvceBase = newDvceBase;
          }
        else
          {
            std::shared_ptr<T> newDvceBase((T*)ptr, GpuDeviceDeleter);
            dvceBase = newDvceBase;
          }
#else
        if (dvceBase && gpuNeedToFree)
          gHeapManager.NeFree(GPU_HEAP, dvceBase, sizeof(T)*mSize);
        dvceBase = ptr;
#endif
	//        isCpuValid = false;
        mSize = _size;
        isGpuValid = true;
        gpuNeedToFree = needToFree;
        gpuAllocated = true;
      }
      
      void setPtr(T *ptr, int _size)
      {
#if USE_SHARED_PTR
        hostBase.reset(ptr);
#else
        if (hostBase && mCapacity >= 0 && cpuNeedToFree)
          gHeapManager.NeFree(CPU_HEAP, hostBase, sizeof(T)*mSize);
        hostBase = ptr;
#endif
        mSize = _size;
        isCpuValid = true;
        cpuAllocated = true;
        cpuNeedToFree = false;
	//        mCapacity = -1; // to disable calling free	
      }
      
      MirroredArray<T> &operator=(const MirroredArray<T> &rhs)
        {
          deep_copy(rhs);
          return *this;
        }


      // deep copy from
      void deep_copy(const MirroredArray<T> &src) 
      {
        resize(src.size());
        //        printf("deep copy src gpuvalid %d my gpuvalid %d gpu alloc %d.\n", src.isGpuValid, isGpuValid, gpuAllocated);
        if (src.isGpuValid)
          {
            if (src.size())
              cudaMemcpy(getGpuPtr(), src.getReadOnlyGpuPtr(), sizeof(T)*mSize, cudaMemcpyDeviceToDevice);
            //            printf("deep copy src gpuvalid %d my gpuvalid %d %p size %zd.\n", src.isGpuValid, isGpuValid,dvceBase, src.size());
          }
        else if (src.isCpuValid)
          {
            if (src.size())
              memcpy(getPtr(), src.getReadOnlyPtr(), sizeof(T)*mSize);
          }
        isGpuArray = src.isGpuArray;
      }

      bool GpuHasLatest() const {
        return isGpuValid;
      }

      bool CpuHasLatest() const {
        return isCpuValid;
      }
      
      // deep copy to
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

      void alias(const MirroredArray<T> & dst) {
        shallow_copy(dst);
        mCapacity = -1; // to disable calling free
      }
      
      void clear()
      {
        resize(0);
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
      
      /*! A clean version of resize.
        It does not copy the old data, 
        nor does it initialize the data 
      */
      void clean_resize(int64_t _size) {
        if (mCapacity >= _size || _size == 0)  {
          mSize = _size;
          isGpuValid = false;
          isCpuValid = false;
          return;
        }
        if (_size > 0)  {
          // depending on previous state
          if (gpuAllocated) {
            if (dvceBase)
              gHeapManager.NeFree(GPU_HEAP, dvceBase, mCapacity*sizeof(T));
            gHeapManager.NeMalloc(GPU_HEAP, (void**)&dvceBase, _size * sizeof(T));
            isGpuValid = false;
          }
          if (cpuAllocated) {
            if (hostBase)
              gHeapManager.NeFree(CPU_HEAP, hostBase, mCapacity*sizeof(T));
            gHeapManager.NeMalloc(CPU_HEAP, (void**)&hostBase, _size * sizeof(T));
            isCpuValid = false;
          }
          mSize = _size;
          mCapacity = _size;
        }
      }

      void resize(int64_t _size) 
      {
#ifdef _DEBUG_
        std::cout << "new size " << _size << " old size " << mSize << std::endl;
#endif
        if (_size <= mCapacity)
          {
            // free memory if resize to zero
            if (_size == 0 && mSize > 0) {
              destroy();
            }
            mSize = _size;
          }
        else // need to reallocate
          {
#if USE_SHARED_PTR 
            if (gpuAllocated)
              {
                std::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(_size*sizeof(T)), GpuDeviceDeleter);
                if (isGpuValid)
                  {
                    cudaError_t error = cudaMemcpy(newDvcebase, dvcebase, mSize*sizeof(T), cudaMemcpyDeviceToDevice);
                    //            std::cout << "memcpy d2d size:" << mSize*sizeof(T)  << std::endl;
                    assert(error == cudaSuccess);
                  }
                dvceBase = newDvceBase;
              }
            if (cpuAllocated)
              {
                std::shared_ptr<T> newHostBase((T*)GpuHostAllocator(_size*sizeof(T)), GpuHostDeleter);
                if (isCpuValid)
                  memcpy(newHostBase, hostBase, mSize*sizeof(T));
                hostBase = newHostBase;            
              }
            mSize = _size;
            mCapacity = _size;
#else
            T *newDvceBase(0);
            T *newHostBase(0);
	    if (!gpuAllocated && !cpuAllocated) {
	      if (isGpuArray) {
		gHeapManager.NeMalloc(GPU_HEAP, (void**)&newDvceBase, _size * sizeof(T));
                assert(newDvceBase);
	      } else {
		gHeapManager.NeMalloc(CPU_HEAP, (void**)&newHostBase, _size*sizeof(T));
                //            newHostBase = (T*)malloc(size * sizeof(T));
                assert(newHostBase);
	      }
	    }
            if (gpuAllocated)
              {
                // cutilSafeCall(cudaMalloc((void**) &newDvceBase, size * _sizeof(T)));
                gHeapManager.NeMalloc(GPU_HEAP, (void**)&newDvceBase, _size * sizeof(T));
                assert(newDvceBase);
                // TODO memcpy 
              }
            if (cpuAllocated)
              {
                gHeapManager.NeMalloc(CPU_HEAP, (void**)&newHostBase, _size*sizeof(T));
                //            newHostBase = (T*)malloc(size * sizeof(T));
                assert(newHostBase);
              }
            if (isCpuValid && cpuAllocated)
              {
                memcpy(newHostBase, hostBase, mSize*sizeof(T));
              }
            if (isGpuValid && gpuAllocated)
              {
                cudaError_t error = cudaMemcpy(newDvceBase, dvceBase, mSize*sizeof(T), cudaMemcpyDeviceToDevice);
                jassert(error == cudaSuccess);
              }
            if (hostBase && mCapacity > 0 && cpuNeedToFree)
              gHeapManager.NeFree(CPU_HEAP, hostBase, sizeof(T)*mSize);
            //          free(hostBase);
            if (dvceBase && gpuNeedToFree)
              {
                gHeapManager.NeFree(GPU_HEAP, dvceBase, sizeof(T)*mSize);
                //            cutilSafeCall(cudaFree(dvceBase));
              }
#ifdef _DEBUG_
            std::cout << "free at resize:" << std::hex << dvceBase << std::endl;
#endif
            hostBase = newHostBase;
            dvceBase = newDvceBase;
	    // if (hostBase)
	    //   std::fill(hostBase+mSize, hostBase + _size, T());
	    // if (dvceBase)
	    //   jusha::cuda::fill(dvceBase + mSize, dvceBase + _size, T());
            mSize = _size;
            gpuAllocated = dvceBase == 0? false: true;
            cpuAllocated = hostBase == 0? false: true;
            mCapacity = _size;
#endif
          }
      }

      void zero()
      {
        if (isGpuArray) {
          cudaMemset((void *)getOverwriteGpuPtr(), 0, sizeof(T)*mSize);
          check_cuda_error("after cudaMemset", __FILE__, __LINE__);
        } else {
          memset((void *)getOverwritePtr(), 0, sizeof(T)*mSize);
        }
      }

      /*! return the pointer without changing the internal state */
      T *getRawPtr() {
        allocateCpuIfNecessary();
        return hostBase;
      }

      /*! return the gpu pointer without changing the internal state */
      T *getRawGpuPtr() {
        allocateGpuIfNecessary();
        return dvceBase;
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
        //        printf("before enable gpu write  %p size %zd, %d %d\n", dvceBase, size(), isGpuValid, gpuAllocated);        
        enableGpuWrite();
        //        printf("returning %p size %zd, %d %d\n", dvceBase, size(), isGpuValid, gpuAllocated);
        return dvceBase;
      }
  
      T *getOverwritePtr() {
        allocateCpuIfNecessary();
        isCpuValid = true;
        isGpuValid = false;
        return hostBase;
      }

      T *getOverwriteGpuPtr() {
        allocateGpuIfNecessary();
        isCpuValid = false;
        isGpuValid = true;
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
        //        printf("cpu valid %d gpu valid %d.\n", isCpuValid, isGpuValid);
        assert(isCpuValid || isGpuValid);
        if (isCpuValid)
          //          return hostBase[index];
          return hostBase[index];
        T ele; 
        allocateCpuIfNecessary();
        //        cudaError_t error = cudaMemcpy(&ele, dvcebase+index, sizeof(T),cudaMemcpyDeviceToHost); 
	//        printf("calling cudamemcpy \n");
        cudaError_t error = cudaMemcpy(&ele, dvceBase+index, sizeof(T),cudaMemcpyDeviceToHost); 
        //    std::cout << "memcpy d2h size:" << sizeof(T)  << std::endl;
        jassert(error == cudaSuccess);
        return ele;
      }

      void setElementAt(T &value, const int index)
      {
        jassert(index < mSize);
        jassert(isCpuValid || isGpuValid);
        if (isCpuValid)
          //          hostBase[index] = value;
          hostBase[index] = value;
        if (isGpuValid)
          {
            //            cudaError_t error = cudaMemcpy(dvcebase+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
            cudaError_t error = cudaMemcpy(dvceBase+index, &value, sizeof(T), cudaMemcpyHostToDevice); 
            jassert(error == cudaSuccess);
          }
      }

      void randomize() 
      {
        RandomWrapper<CURAND_RNG_PSEUDO_MTGP32, T> rng;
        rng.apply(getOverwriteGpuPtr(), mSize);
      }

      // scale the array
      void scale(const T &ratio);
      
      // set the array to the same value
      void fill(const T &val) {
        if (isGpuArray) {
          //      if (true) {
          jusha::cuda::fill(owbegin(), owend(), val);
          check_cuda_error("array fill", __FILE__, __LINE__);
        } else {
          std::fill(getOverwritePtr(), getOverwritePtr()+size(), val);
        }
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

      void invalidateGpu() {
        isGpuValid = false;
      }


      void invalidateCpu() {
         isCpuValid = false;
      }


    inline typename thrust::device_ptr<T> gbegin()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()); 
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> gend()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    { 
      enableGpuWrite();
      return thrust::device_ptr<T>(getGpuPtr()+mSize);
    }

    inline typename thrust::device_ptr<T> owbegin()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      return thrust::device_ptr<T>(getOverwriteGpuPtr()); 
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> owend()
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    { 
      return thrust::device_ptr<T>(getOverwriteGpuPtr()+mSize);
    }


    /*! \brief Return the iterator to the first element in the srt::vector */
    inline typename thrust::device_ptr<T> gbegin() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer)); }
    { 
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getReadOnlyGpuPtr()));
    }

    /*! \brief Return the last iterator (the first invalid iterator) in the srt::vector */
    inline typename thrust::device_ptr<T> gend() const
    //{ return thrust::retag<srt_thrust_tag>(thrust::device_ptr<T>(data_pointer+m_size)); }
    {
      enableGpuRead();
      return thrust::device_ptr<T>(const_cast<T*>(getReadOnlyGpuPtr()+mSize));
    }

      /*! explicitly sync to GPU buffer */
      void syncToGpu() const {
        //	assert(!(isGpuValid && !isCpuValid));
	allocateGpuIfNecessary();
	fromHostToDvce();
	isGpuValid = true;
      }

      /*! explicitly sync to CPU buffer */
      void syncToCpu() const {
        //	assert(!(isCpuValid && !isGpuValid));
	allocateCpuIfNecessary();
	fromDvceToHost();	
	isCpuValid = true;
      }
    
      inline void enableGpuRead() const
      {
        allocateGpuIfNecessary();
        if (!isGpuValid)
          {
            fromHostToDvceIfNecessary();
            setGpuAvailable();
          }
      }

      inline void enableGpuWrite() const
      {
        allocateGpuIfNecessary();
        if (!isGpuValid)
	  fromHostToDvceIfNecessary();
	
	isCpuValid = false;
	isGpuValid = true;
      }

      inline void enableCpuRead() const
      {
	allocateCpuIfNecessary();
        if (!isCpuValid)
          {
            fromDvceToHostIfNecessary();
            isCpuValid = true;
          }
      }

      inline void enableCpuWrite() const
      {
        allocateCpuIfNecessary();
        if (!isCpuValid)
	  fromDvceToHostIfNecessary();

	isCpuValid = true;
	isGpuValid = false;
      }
  
      void setGpuAvailable() const {
        isGpuValid = true;
      }


    private:
      void init_state() {
        mSize = 0;
        mCapacity = 0;
        //        hostBase.reset();
        //        dvceBase.reset();
        hostBase = 0;//nullptr;
        dvceBase = 0; //nullptr;
        isCpuValid = false;
        isGpuValid = false;
        gpuAllocated = false;
        cpuAllocated = false;
        isGpuArray = false;
      }
      
      inline void allocateCpuIfNecessary()  const
      {
        if (!cpuAllocated && mSize)
          {
#if USE_SHARED_PTR
            std::shared_ptr<T> newHostBase((T*)GpuHostAllocator(mSize*sizeof(T)), GpuHostDeleter);
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
            std::shared_ptr<T> newDvceBase((T*)GpuDeviceAllocator(mSize*sizeof(T)), GpuDeviceDeleter);
            assert(newDvcebase != 0);
            dvceBase = newDvceBase;
#else
            gHeapManager.NeMalloc(GPU_HEAP, (void**)&dvceBase, mSize * sizeof(T));
            assert(dvceBase);
#endif
            gpuAllocated = true;
          }
      }


      inline void fromHostToDvce() const {
        if (mSize) {
          jassert(hostBase);
          jassert(dvceBase);
          cudaError_t error = cudaMemcpy(dvceBase, hostBase, mSize* sizeof(T), cudaMemcpyHostToDevice);
          //        std::cout << "memcpy h2d size:" << mSize*sizeof(T)  << std::endl;
          jassert(error == cudaSuccess);
        }
	
      }
      
      inline void fromHostToDvceIfNecessary() const
      {
        if (isCpuValid && !isGpuValid)
          {
#ifdef _DEBUG_
            std::cout << "sync mirror array from host 0x" << std::hex << hostBase << " to device 0x" << dvcebase << " size(" << mSize << "); \n";
#endif
            fromHostToDvce();
	    //            cudaError_t error = cudaMemcpy(dvcebase, hostBase, mSize* sizeof(T), cudaMemcpyHostToDevice);
          }
      }

      inline void fromDvceToHost() const
      {
        if (mSize){
          jassert(hostBase);
          jassert(dvceBase);
          cudaError_t error = cudaMemcpy(hostBase, dvceBase, mSize * sizeof(T),cudaMemcpyDeviceToHost);
          jassert(error == cudaSuccess);
	}
      }
      
      inline void fromDvceToHostIfNecessary() const
      {
        if (isGpuValid && !isCpuValid)
          {
            //            check_cuda_error("before memcpy", __FILE__, __LINE__);
            if (size()) {
              jassert(dvceBase);
              jassert(hostBase);
            }
#ifdef _DEBUG_
            std::cout << "sync mirror array from device 0x" << std::hex << dvceBase << " to host 0x" << hostBase << " size(" << mSize << "); \n";
            /* assert(gHeapManager.find(CPU_HEAP, hostBase) >= (mSize * (int)sizeof(T))); */
            /* assert(gHeapManager.find(GPU_HEAP, dvcebase) >= (mSize * (int)sizeof(T))); */
            assert(gHeapManager.find(CPU_HEAP, hostBase) >= (mSize * (int)sizeof(T)));
            assert(gHeapManager.find(GPU_HEAP, dvceBase) >= (mSize * (int)sizeof(T)));
#endif
            //            cudaError_t error = cudaMemcpy(hostBase, dvcebase, mSize * sizeof(T),cudaMemcpyDeviceToHost);
	    fromDvceToHost();
          }
      }

      int64_t mSize;
      int mCapacity;
#if USE_SHARED_PTR
      std::shared_ptr<T> hostBase;
      std::shared_ptr<T> dvceBase;
#else
      mutable T *hostBase = 0;
      mutable T *dvceBase = 0;
#endif
      mutable bool isCpuValid;
      mutable bool isGpuValid;
      mutable bool gpuAllocated;
      mutable bool cpuAllocated;
      mutable bool isGpuArray;
      mutable bool gpuNeedToFree = true;
      mutable bool cpuNeedToFree = true;      

      void shallow_copy(const MirroredArray<T> &rhs)
      {
        mSize = rhs.mSize;
        mCapacity = rhs.mCapacity;
        hostBase = rhs.hostBase;
        dvceBase = rhs.dvceBase;
        isCpuValid = rhs.isCpuValid;
        isGpuValid = rhs.isGpuValid;
        gpuAllocated = rhs.gpuAllocated;
        cpuAllocated = rhs.cpuAllocated;
        if (isGpuValid)
          assert(gpuAllocated);
        if (isCpuValid)
          assert(cpuAllocated);
      }

      static curandGenerator_t curandGen;
  
    };


    template <typename T, int BATCH>
    struct BatchInit {
      T *ptrs[BATCH];
      size_t sizes[BATCH];
      T vals[BATCH];
    };

    template <typename T, int BATCH>
    void batch_fill_wrapper(int num_arrays, const BatchInit<T, BATCH> &init, cudaStream_t stream);

    /*! Help class to initialize multiple vectors at the same time
     *  
     */
    template <class T, int BATCH>
    class BatchInitializer {
    public:
      
      void push_back(MirroredArray<T> *array, T val) {
        m_arrays.push_back(array);
        m_vals.push_back(val);
        assert(m_arrays.size() < BATCH);
      }
      void init(cudaStream_t stream = 0) {
        BatchInit<T, BATCH> init;
        if (m_arrays.size() > BATCH)
          std::cerr << "Number of arrays " << m_arrays.size() << 
            " exceeding template BATCH " << BATCH << ", please increase BATCH." << std::endl;
        for (int i = 0; i != m_arrays.size(); i++) {
          init.ptrs[i] = m_arrays[i]->getOverwriteGpuPtr();
          init.sizes[i] = m_arrays[i]->size();
          init.vals[i] = m_vals[i];
        }
        batch_fill_wrapper<T, BATCH>(m_arrays.size(), init, stream);
      }


    private:
      std::vector<MirroredArray<T> *> m_arrays;
      std::vector<T> m_vals;
    };
  } // cuda


  // aliasing C++11 feature
  /*  template <typename T> 
      using JVector = cuda::MirroredArray<T>;*/
  #define JVector jusha::cuda::MirroredArray

  /* array operations */

  // y = x0 * x1
  template <class T>
    void multiply(const JVector<T> &x0, const JVector<T> &x1, JVector<T> &y) ;

  // norm 
  template <class T>
    T norm(const JVector<T> &vec);

  template <class T>
  void addConst(JVector<T> &vec, T val);
  
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
