#include <cassert>
#include <map>
#include "timer.h"
#include "utility.h"

namespace jusha {
  CudaEvent::CudaEvent(): _min(100000000.0f),
                          _max(-1.0f),
                          _total(0.0f),
                          _count(0),
                          _max_count(-1),
                          _min_count(-1),
                          is_synced(true)
  {
    cudaError_t error = cudaEventCreate(&begin_event);
    jassert(error == cudaSuccess);
    error = cudaEventCreate(&end_event);
    assert(error == cudaSuccess);
    check_cuda_error("after event start", __FILE__, __LINE__);
  }

  void CudaEvent::start(cudaStream_t stream) {
    if (!is_synced) sync();
    cudaError_t error = cudaEventRecord(begin_event, stream);
    jassert(error == cudaSuccess);
    is_synced = false;
    check_cuda_error("after event start", __FILE__, __LINE__);
  }

  void CudaEvent::stop(cudaStream_t stream) {
    cudaError_t error = cudaEventRecord(end_event, stream);
    jassert(error == cudaSuccess);
    is_synced = false;
    check_cuda_error("after event stop", __FILE__, __LINE__);
  }

  void CudaEvent::sync() {
    if (!is_synced) {
      cudaEventSynchronize(end_event);
      float elapsed;
      cudaEventElapsedTime(&elapsed, begin_event, end_event);
      _min_count = elapsed > _min? _min_count:_count;
      _max_count = elapsed > _max? _count: _max_count;
      _min = elapsed > _min? _min : elapsed;
      _max = elapsed > _max? elapsed: _max;
      _total += elapsed;
      ++_count;
      is_synced = true;
    }
  }

  void CudaEvent::destroy() {
    cudaEventDestroy(begin_event);
    cudaEventDestroy(end_event);
    check_cuda_error("after event start", __FILE__, __LINE__);
  }

  void CudaEvent::print(bool terse) {
    if (terse)
      printf(" %1.12f \n",get_avg()/1000.f);
    else
      printf(" min: %1.12fs @%d, max %1.12fs @%d, total %1.12fs, avg %1.12fs, max/min %1.12f, avg/min %1.12f, count %d \n", _min/1000.f, _min_count, _max/1000.f, _max_count, _total/1000.f, get_avg()/1000.f, _max/_min, get_avg()/_min, _count );
  }


  std::map<std::string, CudaEvent> g_cuda_events;

  void cuda_event_start(const char *name, cudaStream_t stream) {
    std::string e(name);
    auto iter = g_cuda_events.find(e);
    if (iter == g_cuda_events.end()) {
      CudaEvent event;
      std::pair<std::map<std::string,CudaEvent>::iterator,bool> ret;
      ret = g_cuda_events.insert(std::pair<std::string, CudaEvent>(e, event));
      jassert(ret.second);
      iter =  ret.first;
    }
    (*iter).second.start();
  }

  void cuda_event_stop(const char *name, cudaStream_t stream) {
    std::string e(name);
    std::map<std::string, CudaEvent>::iterator iter = g_cuda_events.find(e);
    jassert(iter != g_cuda_events.end());
    if (iter != g_cuda_events.end())
      (*iter).second.stop();
  }


  void cuda_event_sync() {
    std::map<std::string, CudaEvent>::iterator iter;
    for (iter = g_cuda_events.begin(); iter != g_cuda_events.end(); ++iter)
      {
        (*iter).second.sync();
      }
  }

  void cuda_event_destroy() {
    std::map<std::string, CudaEvent>::iterator iter;
    for (iter = g_cuda_events.begin(); iter != g_cuda_events.end(); ++iter)
      {
        (*iter).second.destroy();
      }
    g_cuda_events.clear();
  }


  void cuda_event_print(bool terse) {
    cuda_event_sync();
    if (!terse)
      printf("************* outputing event stats ******************\n");
    std::map<std::string, CudaEvent>::iterator iter;
    if (terse) {
      for (iter = g_cuda_events.begin(); iter != g_cuda_events.end(); ++iter)
        {
          printf("%40s\n",  (*iter).first.c_str());
        }
      printf("\n");
      for (iter = g_cuda_events.begin(); iter != g_cuda_events.end(); ++iter)
        {
          (*iter).second.print(terse);
        }
    } else {
      for (iter = g_cuda_events.begin(); iter != g_cuda_events.end(); ++iter)
        {
          printf("%40s  ",  (*iter).first.c_str());
          (*iter).second.print(terse);
        }
    }
    if (!terse)
      printf("******************************************************\n");
  }
}
