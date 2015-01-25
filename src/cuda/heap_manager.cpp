
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

#include "heap_manager.h"
#include "utility.h"
using namespace std;

namespace jusha {
  HeapManager gHeapManager;

  HeapManager::~HeapManager()
  {
    if (mCpuMemoryTracker.size() > 0)
      std::cout << "memory leak for cpu heap!!!" << std::endl;
    if (mGpuMemoryTracker.size() > 0)
      std::cout << "memory leak for Gpu heap!!!" << std::endl;
#ifdef _DEBUG
    std::cout << "Maximal GPU usage : " << (float)maxGpuUsage/1000000 << "M bytes" << std::endl;
#endif
  }

  void HeapManager::NeMalloc(Memory_Type type, void **addr, int size)
  {
    if (type == CPU_HEAP)
      {
        *addr = (void *)malloc(size);
#ifdef _DEBUG
        mCpuMemoryTracker.insert( pair<void *, int>(*addr, size));
        curCpuUsage += size;
        maxCpuUsage = maxCpuUsage > curCpuUsage? maxCpuUsage: curCpuUsage;
#endif      
      }
    else if (type == GPU_HEAP)
      {
	size_t free, total;
	cudaMemGetInfo(&free, &total);
        check_cuda_error("cudaMemGetInfo", __FILE__, __LINE__);	
        cudaMalloc(addr, size);
	if (size && (*addr == 0))  {
	  printf("allocating memory size %d failed, total %ld, free %ld\n", size, total, free);
	}
        check_cuda_error("cudaMalloc", __FILE__, __LINE__);
#ifdef _DEBUG
        mGpuMemoryTracker.insert( pair<void *, int>(*addr, size));
        curGpuUsage += size;
        maxGpuUsage = maxGpuUsage > curGpuUsage? maxGpuUsage: curGpuUsage;
#endif
      }
    else
      assert(0);

  }

  int HeapManager::find(Memory_Type type, void *addr)
  {
#ifdef _DEBUG
    if (type == CPU_HEAP)
      {
        std::map<void *, int>::iterator it;
        it = mCpuMemoryTracker.find(addr);
        if (it != mCpuMemoryTracker.end())
          {
            return (*it).second;
          }
      }
    else if(type == GPU_HEAP)
      {
        std::map<void *, int>::iterator it;
        it = mGpuMemoryTracker.find(addr);
        if (it != mGpuMemoryTracker.end())
          {
            return (*it).second;
          }
      }
    return 0;
#else
    return 0;
#endif
  }

  void HeapManager::NeFree(Memory_Type type, void *addr)
  {
    if (type == CPU_HEAP)
      {
#ifdef _DEBUG
        std::map<void *, int>::iterator it;
        it = mCpuMemoryTracker.find(addr);
        assert(it != mCpuMemoryTracker.end());
        mCpuMemoryTracker.erase(addr);
        curCpuUsage -= (*it).second;
#endif
        free(addr);

      }
    else if (type == GPU_HEAP)
      {
#ifdef _DEBUG
        std::map<void *, int>::iterator it;
        it = mGpuMemoryTracker.find(addr);
        assert(it != mGpuMemoryTracker.end());
        mGpuMemoryTracker.erase(addr);
        curGpuUsage -= (*it).second;
#endif
        cudaFree(addr);
      }
  }


  void *GpuHostAllocator(int size)
  {
    void *hostBase(0);

    gHeapManager.NeMalloc(CPU_HEAP, (void**)&hostBase, size);
    //  std::cout << " allocating host " << hostBase << std::endl;
    return hostBase;
  }

  void *GpuDeviceAllocator(int size)
  {
    void *dvceBase(0);
    gHeapManager.NeMalloc(GPU_HEAP, (void**)&dvceBase, size);
    //  std::cout << " allocating device " << dvceBase << std::endl;
    return dvceBase;
  }

  void GpuHostDeleter(void *ptr)
  {
    //  std::cout << " releasing host " << ptr << std::endl;
    gHeapManager.NeFree(CPU_HEAP, ptr);
  }

  void EmptyDeviceDeleter(void *ptr)
  {
    //  std::cout << " releasing device " << ptr << std::endl;
  }

  void GpuDeviceDeleter(void *ptr)
  {
    //  std::cout << " releasing device " << ptr << std::endl;
    gHeapManager.NeFree(GPU_HEAP, ptr);
  }

}
