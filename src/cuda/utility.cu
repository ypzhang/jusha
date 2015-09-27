#include <iostream>

namespace jusha {
  void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream )
  {
#ifdef DEBUG
    check_cuda_error_always(kernelname, file, line_no, stream);
#endif
  }

  void check_cuda_error_always(const char *kernelname, const char *file, int line_no, cudaStream_t stream )
  {
    cudaError_t err;
    if (stream)
    cudaStreamSynchronize(stream);
    else
      err = cudaThreadSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA error at kernel " << kernelname << " @ file " << file << " line " << line_no << " for reason: "  << 
	cudaGetErrorString(err) << std::endl;
      //    printBacktrace(10);
      abort();
  }
  

  }
}
