#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <algorithm>
#include "./for_each.hpp"
#include "./array.h"

namespace jusha {
  namespace cuda {
    // A simple kernel to initialize a batch of (ptr, size) pairs.
    template <typename Batch, typename T>
    __global__ void batch_fill_kernel(int num_arrays, int num_big_arrays, Batch batch)
    {
      int id = blockIdx.x;
      
      // small arrays are done by one block each
      if (id < num_arrays) {
        T *ptr = batch.ptrs[id];
        T val = batch.vals[id];
        size_t size = batch.sizes[id];
        for (size_t tid = threadIdx.x; tid < size; tid += blockDim.x) {
          ptr[tid] = val;
        }
      }
      // all block takes part in initializing big array
      for (int big_array = 0; big_array < num_big_arrays; big_array++) {
        T *ptr = batch.big_ptrs[big_array];
        T val = batch.vals2[big_array];
        size_t size = batch.big_sizes[big_array];
        for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
             tid < size; tid += blockDim.x * gridDim.x) {
          ptr[tid] = val;
        }
      }
    }

    template <typename T, int BATCH>
    void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<T, BATCH> &init, cudaStream_t stream)
    {
      int blocks = num_arrays;
      if (num_big_arrays)
        blocks = std::max(64, blocks);
      if (blocks > 0)
        batch_fill_kernel<BatchInit<T, BATCH>, T> 
          <<<blocks, 1024, 0, stream>>>(num_arrays, num_big_arrays, init);
    }

    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<float, 4> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<float, 8> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<float, 12> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<float, 16> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<float, 20> &init, cudaStream_t stream);

    template class BatchInitializer<float, 4>;
    template class BatchInitializer<float, 8>;
    template class BatchInitializer<float, 12>;
    template class BatchInitializer<float, 16>;
    template class BatchInitializer<float, 20>;

    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<double, 4> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<double, 8> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<double, 12> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<double, 16> &init, cudaStream_t stream);
    template void batch_fill_wrapper(int num_arrays, int num_big_arrays, const BatchInit<double, 20> &init, cudaStream_t stream);

    template class BatchInitializer<double, 4>;
    template class BatchInitializer<double, 8>;
    template class BatchInitializer<double, 12>;
    template class BatchInitializer<double, 16>;
    template class BatchInitializer<double, 20>;
  }



  /*********************************************************************************
         Multiply
   *********************************************************************************/
  // Implementation
  template <class T>
  void multiply(const JVector<T> &x0, const JVector<T> &x1, JVector<T> &y)
  {
    //    thrust::transform(x0.gbegin(), x0.gend(), x1.gbegin(), y.gbegin(), [](double v1, double v2)->double { return v1 * v2; });
    thrust::transform(x0.gbegin(), x0.gend(), x1.gbegin(), y.gbegin(), thrust::multiplies<T>());
    check_cuda_error("array multiply", __FILE__, __LINE__);
  }

  // Instantiation
  template
  void multiply(const JVector<double> &x0, const JVector<double> &x1, JVector<double> &y);
  template
  void multiply(const JVector<float> &x0, const JVector<float> &x1, JVector<float> &y);
  template
  void multiply(const JVector<int> &x0, const JVector<int> &x1, JVector<int> &y);
  
  
  /*********************************************************************************
         scale
   *********************************************************************************/
  namespace cuda {
    template <class T>
    void MirroredArray<T>::scale(const T &ratio) {
      thrust::transform(gbegin(), gend(), thrust::constant_iterator<T>(ratio), gbegin(), thrust::multiplies<T>());
      check_cuda_error("array scale", __FILE__, __LINE__);
    }
    
    // Instantiation
    template void MirroredArray<double>::scale(const double &ratio);
    template void MirroredArray<float>::scale(const float &ratio);
    template void MirroredArray<int>::scale(const int &ratio);  
  }


  
  /*********************************************************************************
         setVal
   *********************************************************************************/

  namespace cuda {
#if 0      
    template <class T>
    void MirroredArray<T>::fill(const T &val) {
      if (isGpuArray) {
	thrust::fill(gbegin(), gend(), val);
	check_cuda_error("array fill", __FILE__, __LINE__);
      } else {
	std::fill(getPtr(), getPtr()+size(), val);
      }
    }

#endif
    template <typename T>
    void fill(T *begin, T *end, const T & val) {
      thrust::fill(begin, end, val);      
    }

  
  template <class T>
    class fill_run_nv: public nvstd::function<void(T)> {
    public:
      __device__ void operator()(int gid, thrust::tuple<T*, T> &tuple) const {
        thrust::get<0>(tuple)[gid] = thrust::get<1>(tuple);
      }
    };

    template <typename T>
    void fill(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, const T&val)
    {
#if 0 // thrust call
      thrust::fill(begin, end, val);
#else
      ForEachKernel<StridePolicy, JC_cuda_blocksize, false> kernel(end-begin, "Fill"); 
      kernel.run<fill_run_nv<T>, T*, T>(thrust::raw_pointer_cast(begin), val);
#endif
    }
    
    // // Instantiation
    // template void MirroredArray<double>::fill(const double &ratio);
    // template void MirroredArray<float>::fill(const float &ratio);
    // template void MirroredArray<int>::fill(const int &ratio);
    template void fill(bool *, bool *, const bool &);
    template void fill(double *, double *, const double &);
    template void fill(float *, float *, const float &);
    template void fill(int *, int *, const int &);
    template void fill(long long*, long long*, const long long &);
    template void fill(float2 *, float2 *, const float2 &);
    template void fill(float4 *, float4 *, const float4 &);    

    template void fill(thrust::device_ptr<bool> begin, thrust::device_ptr<bool> end, const bool&val);
    template void fill(thrust::device_ptr<double> begin, thrust::device_ptr<double> end, const double&val);
    template void fill(thrust::device_ptr<float> begin, thrust::device_ptr<float> end, const float&val);
    template void fill(thrust::device_ptr<int> begin, thrust::device_ptr<int> end, const int&val);
    template void fill(thrust::device_ptr<unsigned int> begin, thrust::device_ptr<unsigned int> end, const unsigned int&val);
    template void fill(thrust::device_ptr<float2> begin, thrust::device_ptr<float2> end, const float2&val);
    template void fill(thrust::device_ptr<float4> begin, thrust::device_ptr<float4> end, const float4&val);

  }


  /*********************************************************************************
   *      addConst
   *********************************************************************************/
  // Implementation
  template <class T>
  void addConst(JVector<T> &vec, T val)
  {
    thrust::transform(vec.gbegin(), vec.gend(),  thrust::make_constant_iterator(val),
                      vec.gbegin(), thrust::plus<T>());
  }
  // Instantiation
  template void addConst(JVector<int> &vec, int);
  template void addConst(JVector<double> &vec, double);
  template void addConst(JVector<long long> &vec, long long);  
  template void addConst(JVector<float> &vec, float);
  template void addConst(JVector<long> &vec, long);  



  /*********************************************************************************
         norm
   *********************************************************************************/
  template <typename T>
struct square
  {
    __host__ __device__
    T operator()(const T& x) const { 
      return x * x;
    }
  };
  // Implementation
  template <class T>
    T norm(const JVector<T> &vec)
  {
    if (!vec.size()) return 0.0;
    // prefer GPU implementation
    if (vec.GpuHasLatest())  {
      square<T>        unary_op;
      thrust::plus<T> binary_op;
      return std::sqrt( thrust::transform_reduce(vec.gbegin(), vec.gend(), unary_op, 0.0, binary_op) );
    } else {
      assert(vec.CpuHasLatest());
      T sum(0.0);
      const T* vec_ptr = vec.getReadOnlyPtr();
      for (int i = 0; i != vec.size(); i++)
        sum += vec_ptr[i] * vec_ptr[i];
      return std::sqrt(sum);
    }
  }

  // Instantiation
  template float norm (const JVector<float> &vec);
  template double norm (const JVector<double> &vec);

  /*********************************************************************************
         Next
   *********************************************************************************/
  // Implementation
  // Instantiation

}
