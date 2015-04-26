#pragma once

class GpuAllocator {
public:
  virtual ~GpuAllocator() {}
  virtual void *allocate(size_t bsize) = 0;
  virtual void deallocate(void *p, size_t bsize) = 0;
};
