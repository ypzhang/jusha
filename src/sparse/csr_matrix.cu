#include "./csr_matrix.h"
#include "cuda/cuda_config.h"
#include "cuda/device/partition.cuh"

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
   int blocks = GET_BLOCKS(num_rows);
   blocks = CAP_BLOCK_SIZE(blocks);
   if (blocks > 0)
      split_diag_coefs_kernel<<<blocks, jusha::cuda::JCKonst::cuda_blocksize>>>(row_ptrs, cols, coefs, diag, offd, num_rows);
}


template
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const double* coefs, double *diag, double *offd);

template
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const float* coefs, float *diag, float *offd);


  template <bool safe_boundary>
  __device__ void load_csr_row_to_shm(int *sh_csr_row, const int * __restrict__ csr_row, int idx, int bound_guard) {
    if (safe_boundary || idx < bound_guard)
      sh_csr_row[threadIdx.x] = csr_row[idx];
    if (threadIdx.x == 0 && (safe_boundary || (idx + blockDim.x) < bound_guard))
      sh_csr_row[blockDim.x] = csr_row[idx + blockDim.x];
    __syncthreads();
  }



__global__ void  
csr_row_to_coo_row_kernel(const int32_t * __restrict__ csr_rows, int32_t * __restrict__ coo_rows, int32_t N)
{
  __shared__ int32_t sh_csr[JC_cuda_blocksize+1];
  __shared__ int32_t sh_coo[JC_cuda_blocksize];
  __shared__ int32_t last_rowptr;
  int32_t row = kernel_get_1d_gid;
  int stride = kernel_get_1d_stride;

  sh_csr[threadIdx.x] = 0;
  //  assert(blockDim.x == JC_cuda_blocksize);
  bool is_last(false);
  int bs_start, bs_end;
  const int batch_size = blockDim.x;
  cuda::block_partition(N, batch_size, bs_start, bs_end, is_last);

  if (bs_start == bs_end) return;
  int curr_csr_in, next_csr_in;
  int ele_start = bs_start * batch_size;
  int ele_end = bs_end * batch_size;
  ele_end = ele_end >= N? N: ele_end;

  curr_csr_in = ele_start + threadIdx.x < ele_end? csr_rows[ele_start + threadIdx.x] : -1;
  next_csr_in = ele_start + batch_size + threadIdx.x < ele_end? csr_rows[ele_start + threadIdx.x + batch_size] : -1;
  if (threadIdx.x == 0) printf("bs_start %d bs_end %d for blockIdx %d islast? %d ele %d to %d cur_csr_in %d. %d\n", bs_start, bs_end, blockIdx.x, is_last, ele_start, ele_end, curr_csr_in, csr_rows[0]);  
  for (int bs = bs_start; bs < bs_end /*- 2*/; bs++) {
    int elem_block_base = bs * batch_size;
    sh_csr[threadIdx.x] = curr_csr_in;
    int my_row_start = curr_csr_in;    
    curr_csr_in = next_csr_in;
    next_csr_in = bs * batch_size + (bs<<1) + threadIdx.x < ele_end? -1 : csr_rows[bs * batch_size + (bs<<1) + threadIdx.x];  // preload the next batch
    if (threadIdx.x == 0) sh_csr[blockDim.x] = next_csr_in;
    __syncthreads();
    int col_start = sh_csr[0];
    int col_end = sh_csr[elem_block_base + batch_size < ele_end? blockDim.x : ele_end - elem_block_base];
    printf("col_start %d end %d ele_end %d sh_csr[5] %d [4] %d id %d.\n", col_start, col_end, ele_end, sh_csr[5], sh_csr[4], ele_end-elem_block_base);
    int iters = (col_end - col_start)+batch_size -1 / batch_size;
    int my_row_end = sh_csr[threadidx.x+1];
    int my_iter = -1;
    if (my_row_end != -1) {
      
    }
    
  }
  //  if (
  /*  
  for (; row < N - blockDim.x; row += stride) {
    //    load_csr_row_to_shm(sh_csr,  csr_rows, row, 
  }
  */
}
  
  void csr_row_to_coo_row(const JVector<int32_t> &csr_rows, const int32_t nrows, const int32_t nnzs, JVector<int32_t> &coo_rows) {
    
    assert(csr_rows.size() == (nrows+1));
    coo_rows.resize(nnzs);
    int blocks = GET_BLOCKS(nrows);
    blocks = CAP_BLOCK_SIZE(blocks);
    if (blocks > 0)
      csr_row_to_coo_row_kernel<<<blocks, jusha::cuda::JCKonst::cuda_blocksize>>>(csr_rows.getReadOnlyGpuPtr(),
                                                                                  coo_rows.getGpuPtr(),
                                                                                  nrows+1);
  }
}
