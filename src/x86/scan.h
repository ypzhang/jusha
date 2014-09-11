#ifndef JUSHA_X86_SCAN_H
#define JUSHA_X86_SCAN_H

#include <thrust/device_ptr.h>

namespace jusha {
  namespace x86 {
    #include "./detail/exclusive_scan_v1.h"
    #include "./detail/exclusive_scan_v2.h"

  }
}

#endif

