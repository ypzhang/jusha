#include <chrono>

#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>


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

  // http://panthema.net/2008/0901-stacktrace-demangled/
  /** Print a demangled stack backtrace of the caller function to FILE* out. */
  void print_stacktrace(FILE *out, unsigned int max_frames)
  {
    fprintf(out, "stack trace:\n");

    // storage array for stack trace address data
    void* addrlist[max_frames+1];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));
    if (addrlen == 0) {
      fprintf(out, "  <empty, possibly corrupt>\n");
      return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < addrlen; i++)
      {
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

        // find parentheses and +address offset surrounding the mangled name:
        // ./module(function+0x15c) [0x8048a6d]
        for (char *p = symbollist[i]; *p; ++p)
          {
            if (*p == '(')
              begin_name = p;
            else if (*p == '+')
              begin_offset = p;
            else if (*p == ')' && begin_offset) {
              end_offset = p;
              break;
            }
          }

        if (begin_name && begin_offset && end_offset
            && begin_name < begin_offset)
          {
            *begin_name++ = '\0';
            *begin_offset++ = '\0';
            *end_offset = '\0';

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // __cxa_demangle():

            int status;
            char* ret = abi::__cxa_demangle(begin_name,
                                            funcname, &funcnamesize, &status);
            if (status == 0) {
              funcname = ret; // use possibly realloc()-ed string
              fprintf(out, "  %s : %s+%s\n",
                      symbollist[i], funcname, begin_offset);
            }
            else {
              // demangling failed. Output function name as a C function with
              // no arguments.
              fprintf(out, "  %s : %s()+%s\n",
                      symbollist[i], begin_name, begin_offset);
            }
          }
        else
          {
            // couldn't parse the line? print the whole line.
            fprintf(out, "  %s\n", symbollist[i]);
          }
      }

    free(funcname);
    free(symbollist);
    //    fprintf(stderr, "Now run addr2line -e MODULE_NAME  -j .text HEX_ADDRESS from bottom to up to retrieve line numbers!\n, Need to accumulate HEX_ADDRESS.\n");
  }

}
