#pragma once
#include <cassert>

namespace jusha {
  
template <int groupsize>
class ForEachPolicy {
public:
  virtual __device__ void group_sync() {}
  __device__ int group_size() {
    return groupsize;
  }
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const = 0;
};

template <int groupsize>
class StridePolicy: public ForEachPolicy<groupsize> {
public:
  virtual __device__ int num_batches(int _N, int my_id, int max_id, bool need_sync) const {
    if (!need_sync) {
      return _N/max_id + (_N%max_id)<my_id? 1:0 ;
    } else {
      return (_N+max_id-1)/max_id;
    }
  }
};

template <int groupsize>
class BlockPolicy: public ForEachPolicy<groupsize> {
public:
  virtual __device__ int num_batches(int _N, int my_id, int max_id, int batchsize, bool need_sync) const {
    assert((max_id % groupsize) == 0);
    int num_groups = max_id /groupsize;
    int group_id = my_id/batchsize;
    int total_batches = (_N+batchsize-1)/batchsize;
    int batch_per_group = (total_batches + num_groups - 1)/num_groups;
    int batch_start = batch_per_group * group_id;
    int batch_end = batch_start + batch_per_group;
    batch_end = batch_end > total_batches? batch_end : total_batches;
    int num_batches = batch_end - batch_start;
    if (!need_sync)
      return num_batches;

    //    if (batch_end == total_batches)
    {
      int elem_in_last_batch = _N - batch_end * batchsize;
      int id_in_group = my_id % batchsize;
      return num_batches - (id_in_group > elem_in_last_batch? 1:0);
    }
    
  }
};

template <template<int> class Policy, int group_size, bool need_sync>
class ForEach {
public:
  __host__ __device__ ForEach(int32_t _N, int32_t my_id, int32_t max_id):
    N(_N), m_id(my_id), m_max_id (max_id), m_group_size(group_size) {
    Policy<group_size> policy;
    
    //    int lane_id = my_id % group_size;
  }
  
  __host__ __device__ bool is_active() const {
    return m_is_active;
  }

  __host__ __device__ void execute_wrapper()
  {
    
  }

  __host__ __device__ int num_batches() const {
   return m_batches;
  }
  
  __host__ __device__ int next_batch() const {
    
  }
  
private:
  int N = 0;
  int m_id = 0;
  int m_max_id = 0;
  int m_group_size = 0;
  int m_batches = 0;
  bool m_is_active = false;

};


}
