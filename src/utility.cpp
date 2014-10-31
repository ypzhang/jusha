#include <chrono>
#include "utility.h"

using namespace std;
using namespace std::chrono;

namespace jusha {
  //typedef std::chrono::high_resolution_clock jusha_perf_clock;
typedef std::chrono::system_clock jusha_perf_clock;

jusha_perf_clock::time_point jusha_g_world_start = jusha_perf_clock::now();

double jusha_get_wtime()
{
  auto start = jusha_perf_clock::now();
  std::chrono::microseconds elapse_since_world_start = duration_cast<microseconds>(start - jusha_g_world_start);
  //  cout << "count  "<<elapse_since_world_start.count() << std::endl;
  return double(elapse_since_world_start.count()) / 1000000.0;
}

//   void check_cuda_error(const char *kernelname, const char *file, int line_no, cudaStream_t stream)  {
// #ifdef DEBUG
//     //#if 1
//     cudaError_t err;
//     if (stream)
//       cudaStreamSynchronize(stream);
//     else
//       err = cudaDeviceSynchronize();

//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//       fprintf (stdout, "CUDA error in at thread %d kernel %s @ file %s line %d:\n  %s\n",
//                omp_get_thread_num(), kernelname, file, line_no, cudaGetErrorString(err));
//       abort();
//     }
// #endif
//   }
}
