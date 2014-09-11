#pragma once

#include <iostream>
#include <cuda_runtime_api.h>

namespace jusha {

  inline void jusha_assert_fail (const char *expr, const char *file, int line, const char *func)
  {
    fprintf(stderr, "Assertion %s failed:  File %s , line %d, function %s\n",
                   expr, file, line, func);
  }

#ifdef WIN32
#define __JUSHA_FUNC__ __FUNCTION__
#else
#define __JUSHA_FUNC__ __PRETTY_FUNCTION__
#endif

#define jassert(expression) \
  ((expression) ? (void) 0  \
   : jusha_assert_fail(#expression, __FILE__, __LINE__, __JUSHA_FUNC__))



void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream = 0);


}
