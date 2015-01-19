#include "./csr_matrix.h"
#include "cuda/cuda_config.h"

namespace jusha {

  // TODO optimize it for coelesced memory
template<typename T>
__global__ void split_diag_coefs_kernel(const int32_t *row_ptrs, const int64_t *cols, const T* coefs, T *diag, T *offd, int64_t N)
{
    int64_t row = kernel_get_1d_gid;
    int stride = kernel_get_1d_stride;
    for (; row < N; row += stride) {
      int row_start = row_ptrs[row];
      int row_end = row_ptrs[row+1];
      //      if (row == 0) printf("row start %d row end %d.\n", row_start, row_end);      
      for (int this_row = row_start; this_row < row_end; this_row++) {

        bool found_diag(false);
        if (cols[this_row] == row)  {
          found_diag = true;
          // if (row == 0)
          //   printf("copying diag %f cols %zd row %d this row %d\n", coefs[this_row], cols[this_row], row, this_row);
          diag[row] = coefs[this_row];
        } else {
          offd[this_row - row - found_diag?1:0] = coefs[this_row];
        }
      }
    }
}
  
template<typename T>
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const T* coefs, T *diag, T *offd)
{
  printf("callign separate diag\n");
   int blocks = GET_BLOCKS(num_rows);
   blocks = CAP_BLOCK_SIZE(blocks);
   if (blocks > 0)
      split_diag_coefs_kernel<<<blocks, jusha::cuda::JCKonst::cuda_blocksize>>>(row_ptrs, cols, coefs, diag, offd, num_rows);
}


template
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const double* coefs, double *diag, double *offd);

template
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const float* coefs, float *diag, float *offd);

}
