#include <thrust/transform.h>
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
  }

  // Instantiation
  template
  void multiply(const JVector<double> &x0, const JVector<double> &x1, JVector<double> &y);
  template
  void multiply(const JVector<float> &x0, const JVector<float> &x1, JVector<float> &y);
  template
  void multiply(const JVector<int> &x0, const JVector<int> &x1, JVector<int> &y);
  
  
  
  /*********************************************************************************
         Next
   *********************************************************************************/
  // Implementation
  // Instantiation

}
