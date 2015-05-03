#include "./array.h"
#include "cuda/for_each.hpp"

namespace jusha {
  //  namespace cuda {
  
  template <class T>
    class plus_run_nv: public nvstd::function<void(T)> {
    public:
      __device__ void operator()(int gid, std::tuple<const T*, const T*, T *> &tuple) const {
        std::get<2>(tuple)[gid] = std::get<0>(tuple)[gid] + std::get<1>(tuple)[gid];
      }
    };

  template <class T>
    class minus_run_nv: public nvstd::function<void(T)> {
    public:
      __device__ void operator()(int gid, std::tuple<const T*, const T*, T *> &tuple) const {
        std::get<2>(tuple)[gid] = std::get<0>(tuple)[gid] - std::get<1>(tuple)[gid];
      }
    };


    template <class T>
    void plus(const JVector<T> &lhs, const JVector<T> &rhs, JVector<T> &result)
    {
      assert(lhs.size() == rhs.size());
      assert(result.size() == rhs.size());
      ForEachKernel<StridePolicy, JC_cuda_blocksize, false> kernel(lhs.size());
      kernel.run<plus_run_nv<T>, const T*, const T*, T*>(lhs.getReadOnlyGpuPtr(), rhs.getReadOnlyGpuPtr(), result.getGpuPtr());
    }

    template <class T>
    void minus(const JVector<T> &lhs, const JVector<T> &rhs, JVector<T> &result)
    {
      assert(lhs.size() == rhs.size());
      assert(result.size() == rhs.size());
      ForEachKernel<StridePolicy, JC_cuda_blocksize, false> kernel(lhs.size());
      kernel.run<minus_run_nv<T>, const T*, const T*, T*>(lhs.getReadOnlyGpuPtr(), rhs.getReadOnlyGpuPtr(), result.getGpuPtr());
    }

  //  }
}
