// reduce primitives on device
// #include <cub/cub.h>
namespace jusha {
  namespace cuda {

    __device__ void block_partition(int N, int batch_size, int &bs_start, int &bs_end, bool &is_last)
 {
   int num_batches = (N + batch_size - 1) / batch_size;
   int nbatch_per_block = (num_batches + gridDim.x - 1)/ gridDim.x;
   bs_start = nbatch_per_block * blockIdx.x;
   bs_end = nbatch_per_block * (blockIdx.x + 1);
   bs_start = bs_start > num_batches? num_batches: bs_start;
   bs_end = bs_end > num_batches? num_batches: bs_end;
   is_last = (bs_start != bs_end) && (bs_end == num_batches);
 }
 
  }
}
