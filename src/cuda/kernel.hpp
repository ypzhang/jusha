#pragma once
#include <string>

namespace jusha {
/* Base class for all cuda kernel classes
 * Do some loggings
 */
class CudaKernel {
  public: 
  CudaKernel() : m_tag(std::string("UnknownName:UnknownDim")) {}
  std::string get_tag() const { return m_tag; }
  void set_tag(std::string str) { m_tag = str; }
  
protected:
  std::string m_tag;
};
 


}
