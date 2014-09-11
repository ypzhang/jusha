#pragma once 

#include <cuda_runtime_api.h>

namespace jusha {
  class CudaEvent {
  public:
    CudaEvent();
    void start(cudaStream_t stream = 0);
    void stop(cudaStream_t stream = 0);
    void sync();
    void print(bool terse);
    void destroy();
    float get_min() const { return _min; }
    float get_max() const { return _max; }
    float get_avg() const { return _total / _count; }
    int get_count() const { return _count; }
  private:
    float _min;
    float _max;
    float _total;
    int _count;
    int _max_count ;
    int _min_count;
    bool is_synced;
    cudaEvent_t begin_event;
    cudaEvent_t end_event;
  };

  void cuda_event_start(const char*, cudaStream_t stream = 0);
  void cuda_event_stop(const char*, cudaStream_t stream = 0);
  void cuda_event_sync();
  void cuda_event_destroy();
  void cuda_event_print(bool terse = false) ;
}

