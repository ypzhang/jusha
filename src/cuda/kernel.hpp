#pragma once
#include <string>
#include "cuda/cuda_config.h"

namespace jusha {
/* Base class for all cuda kernel classes
 * Do some loggings
 */
class CudaKernel {
  public: 
  CudaKernel() {}
  std::string get_tag() const { return m_tag; }
  void set_tag(std::string str) { m_tag = str; }

  void set_block_size(int block_size) { 
    m_block_size = block_size;
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
  bool m_auto_tuning = true;
};
 


}
