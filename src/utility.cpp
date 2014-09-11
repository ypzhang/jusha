#include "utility.h"

namespace jusha {
  void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream)  {
#ifdef DEBUG
    //#if 1
    cudaError_t err;
    if (stream)
      cudaStreamSynchronize(stream);
    else
      err = cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf (stdout, "CUDA error in at thread %d kernel %s @ file %s line %d:\n  %s\n",
               omp_get_thread_num(), kernelname, file, line_no, cudaGetErrorString(err));
      abort();
    }
#endif
  }
}
