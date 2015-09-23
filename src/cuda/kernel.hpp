#pragma once
#include <cxxabi.h>
#include <sstream>
#include <string>
#include "./cuda_config.h"


namespace jusha {
/* Base class for all cuda kernel classes
 * Do some loggings
 */
class CudaKernel {
  public: 
  CudaKernel(int32_t N, const std::string &tag) {
      std::stringstream tag_stream ;
#ifdef __GNUG__
      int status;
      char * demangled = abi::__cxa_demangle(typeid(*this).name(),0,0,&status);
      //      m_tag = std::string(demangled);
      tag_stream << demangled;
      free(demangled);
#else
      tag_stream <<       typeid(this).name();
#endif
      tag_stream << ":" << tag << ":Dim_" << N;
      set_tag(tag_stream.str());
  }

  std::string get_tag() const { return m_tag; }
  void set_tag(std::string str) { m_tag = str; }

  void set_block_size(int block_size) { 
    m_block_size = block_size;
    disable_auto_tuning();
  }

  void set_max_block(int max_block) { 
    m_max_blocks = max_block;
  }


  void set_auto_tuning() {
    m_auto_tuning = true;
  }

  void disable_auto_tuning() {
    m_auto_tuning = false;
  }

protected:
  std::string m_tag;
  int m_block_size = jusha::cuda::JCKonst::cuda_blocksize;
  int m_max_blocks = jusha::cuda::JCKonst::cuda_max_blocks * 2;
  bool m_auto_tuning = true;
  cudaStream_t m_stream = 0;
};
 


}
