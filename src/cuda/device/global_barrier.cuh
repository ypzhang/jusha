
namespace jusha {
  namespace cuda {
    namespace device {
      __inline__ __device__ void g_barrier_phase1(volatile int *counter)
      {
        __syncthreads();
        if (threadIdx.x == 0) {
          while (*counter != blockIdx.x)  ;
        }
        __syncthreads();
      }

      __inline__ __device__ void g_barrier_phase2(volatile int *counter)
      {
        if (blockIdx.x == (gridDim.x-1)) 
          (*counter) = 0; // last one to reset counter
        else
          *counter = blockIdx.x + 1;

        if (threadIdx.x == 0) {
          while (*counter != 0) ; // all wait for the last to reset
        }
        __syncthreads();
      }

      __inline__ __device__ void g_barrier_phase3(volatile int *counter)
      {

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
        g_barrier_phase1(counter);
        g_barrier_phase2(counter);
      }

    }
  }
}
