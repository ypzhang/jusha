#pragma once

#include <map>

namespace jusha {
/* a heap manager that helps debugging memory alloc/free bugs */
typedef enum{
  CPU_HEAP,
  GPU_HEAP
} Memory_Type;

void *GpuHostAllocator(int size);
void *GpuDeviceAllocator(int size);
void GpuHostDeleter(void *ptr);
void GpuDeviceDeleter(void *ptr);
void EmptyDeviceDeleter(void *ptr);


class HeapManager{
 public:
 HeapManager():maxCpuUsage(0),
    maxGpuUsage(0)
      {}
  ~HeapManager();
  void NeMalloc(Memory_Type type, void ** addr, int size);
  void NeFree(Memory_Type type, void *addr);
  int find(Memory_Type type, void *addr);
 private:
  std::map <void *, int> mGpuMemoryTracker;
  std::map <void *, int> mCpuMemoryTracker;
  int maxCpuUsage;
  int maxGpuUsage;
  int curCpuUsage;
  int curGpuUsage;
};

}
