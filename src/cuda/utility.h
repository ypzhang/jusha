#pragma once

#include <cuda_runtime_api.h>

namespace jusha {
void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream = 0);

}
