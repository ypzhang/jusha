#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <algorithm>
#include "./array.h"

namespace jusha {

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

    template <typename T>
    void fill(thrust::device_ptr<T> begin, thrust::device_ptr<T> end, const T&val)
    {
      thrust::fill(begin, end, val);
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
         Next
   *********************************************************************************/
  // Implementation
  // Instantiation

}
