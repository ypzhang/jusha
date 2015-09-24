#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
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
    template <class T>
    void MirroredArray<T>::fill(const T &val) {
      thrust::fill(gbegin(), gend(), val);
      check_cuda_error("array fill", __FILE__, __LINE__);
    }
    
    // Instantiation
    template void MirroredArray<double>::fill(const double &ratio);
    template void MirroredArray<float>::fill(const float &ratio);
    template void MirroredArray<int>::fill(const int &ratio);  
  }
  
  /*********************************************************************************
         Next
   *********************************************************************************/
  // Implementation
  // Instantiation

}
