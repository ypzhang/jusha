
namespace jusha {
  namespace cuda {
    namespace device {
      __inline__ __device__ void pre_g_barrier(volatile int *counter)
      {
        __syncthreads();
        if (threadIdx.x == 0) {
          while (*counter != blockIdx.x)  ;
          int cache_counter = *counter;
          if (cache_counter == (gridDim.x-1)) 
            (*counter) = 0; // last one to reset counter
          else
            *counter = cache_counter + 1;
        }
      }

      __inline__ __device__ void post_g_barrier(volatile int *counter)
      {
        if (threadIdx.x == 0) {
          while (*counter != 0) ; // all wait for the last to reset
        }
        __syncthreads();
      }

      __inline__ __device__ void global_barrier(volatile int *counter)
      {
#if 0
        __syncthreads();
        if (threadIdx.x == 0) {

          while (*counter != blockIdx.x)  ;
          *counter = *counter + 1;
          if ((*counter) == gridDim.x) (*counter) = 0; // last one to reset counter
          while (*counter != 0) ; // all wait for the last to reset
        }
        __syncthreads();
#endif
        pre_g_barrier(counter);
        post_g_barrier(counter);
      }

    }
  }
}
