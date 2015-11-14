#include <cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include "heap_manager.h"
#include "heap_allocator.h"



#include "utility.h"
using namespace std;

#define USE_CUDA_ALLOCATOR
//#undef USE_CUDA_ALLOCATOR
namespace jusha {
  HeapManager gHeapManager;
  int HeapManager::max_device_ids = 32;

  HeapManager::HeapManager() {
    mGpuHeapAllocators.resize(HeapManager::max_device_ids, nullptr);
  }

  HeapManager::~HeapManager()
  {
    if (mCpuMemoryTracker.size() > 0)
      std::cout << "memory leak for cpu heap!!!" << std::endl;
    if (mGpuMemoryTracker.size() > 0) {
      std::cout << "memory leak for Gpu heap!!!" << std::endl;
      for (auto i = mGpuMemoryTracker.begin(); i != mGpuMemoryTracker.end(); i++) {
        printf("Memory %p size %d were not properly freed.\n", i->first, i->second);
      }
    }
#ifdef _DEBUG
    std::cout << "Maximal GPU usage : " << (float)maxGpuUsage/1000000 << "M bytes" << std::endl;
#endif

    for (auto i = mGpuHeapAllocators.begin(); i != mGpuHeapAllocators.end(); i++) {
      if (*i != nullptr)
        delete *i;
    }
  }

  void HeapManager::NeMalloc(Memory_Type type, void **addr, const size_t &size)
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
        // init gpu allocator if not exist
#ifdef USE_CUDA_ALLOCATOR
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        check_cuda_error("cudaMemGetInfo", __FILE__, __LINE__);	
        cudaMalloc(addr, size);
#ifdef _DEBUG
        count++;
        printf("alocating GPU %p for size %ld count = %d\n", *addr, size, count);
#endif
        if (size && (*addr == 0))  {
          printf("allocating memory size %ld failed, total %ld, free %ld\n", size, total, free);
        }
        check_cuda_error("cudaMalloc", __FILE__, __LINE__);
#else
        HeapAllocator *allocator = get_gpu_allocator();
        assert(allocator);
        *addr = allocator->allocate(size);
#endif
        if (addr == 0) {
          size_t free, total;
          cudaMemGetInfo(&free, &total);
          fprintf(stderr, "Failed to allocate memory size %f Kbytes, free memory %f Kbytes, total %f Kbytes.\n",
                  float(size)/1000., float(free)/1000., float(total)/1000.);
        }
#ifdef _DEBUG
        
        mGpuMemoryTracker.insert( pair<void *, int>(*addr, size));
        curGpuUsage += size;
        maxGpuUsage = maxGpuUsage > curGpuUsage? maxGpuUsage: curGpuUsage;
#endif
      }
    else
      assert(0);
  }

  HeapAllocator *HeapManager::get_gpu_allocator()
  {
    int device(-1);
    cudaGetDevice(&device);
    assert(device >= 0);
    assert(device <= HeapManager::max_device_ids);
    if (mGpuHeapAllocators[device] == nullptr)
      mGpuHeapAllocators[device] = new HeapAllocator();
    return mGpuHeapAllocators[device];
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

  void HeapManager::NeFree(Memory_Type type, void *addr, const size_t &size)
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
        printf("free memory %p\n", addr);
        std::map<void *, int>::iterator it;
        it = mGpuMemoryTracker.find(addr);
        assert(it != mGpuMemoryTracker.end());
        mGpuMemoryTracker.erase(addr);
        curGpuUsage -= (*it).second;
#endif
#ifdef USE_CUDA_ALLOCATOR
        cudaFree(addr);
#else
        get_gpu_allocator()->deallocate(addr, size);
#endif
      }
  }


  void *GpuHostAllocator(size_t  size)
  {
    void *hostBase(0);

    gHeapManager.NeMalloc(CPU_HEAP, (void**)&hostBase, size);
    //  std::cout << " allocating host " << hostBase << std::endl;
    return hostBase;
  }

  void *GpuDeviceAllocator(size_t size )
  {
    void *dvceBase(0);
    gHeapManager.NeMalloc(GPU_HEAP, (void**)&dvceBase, size);
    //  std::cout << " allocating device " << dvceBase << std::endl;
    return dvceBase;
  }

  void GpuHostDeleter(void *ptr, size_t size)
  {
    //  std::cout << " releasing host " << ptr << std::endl;
    gHeapManager.NeFree(CPU_HEAP, ptr, size);
  }

  void EmptyDeviceDeleter(void *ptr, size_t size)
  {
    //  std::cout << " releasing device " << ptr << std::endl;
  }

  void GpuDeviceDeleter(void *ptr, size_t size)
  {
    //  std::cout << " releasing device " << ptr << std::endl;
    gHeapManager.NeFree(GPU_HEAP, ptr, size);
  }

}
