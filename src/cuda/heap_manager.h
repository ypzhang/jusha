#pragma once

#include <map>
#include <vector>

namespace jusha {
/* a heap manager that helps debugging memory alloc/free bugs */
typedef enum{
  CPU_HEAP,
  GPU_HEAP
} Memory_Type;

void *GpuHostAllocator(size_t size);
void *GpuDeviceAllocator(size_t size);
  void GpuHostDeleter(void *ptr, size_t size);
  void GpuDeviceDeleter(void *ptr, size_t size);
  void EmptyDeviceDeleter(void *ptr, size_t size);

  class HeapAllocator;
class HeapManager{
 public:
  HeapManager();
  ~HeapManager();
  void NeMalloc(Memory_Type type, void ** addr, const size_t &size);
  void NeFree(Memory_Type type, void *addr, const size_t &size);
  int find(Memory_Type type, void *addr);
 private:
  HeapAllocator *get_gpu_allocator();
  static int max_device_ids;
  std::map <void *, int> mGpuMemoryTracker;
  std::map <void *, int> mCpuMemoryTracker;
  // for multiple GPUs
  std::vector<HeapAllocator *> mGpuHeapAllocators;
  /* int maxCpuUsage; */
  /* int maxGpuUsage; */
  /* int curCpuUsage; */
  /* int curGpuUsage; */
};

}
