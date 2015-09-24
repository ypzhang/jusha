#pragma once

namespace jusha {
  namespace cuda {

    static  __device__ int atomic_add(int* address, int val) {
      return atomicAdd(address, val);
    }

    static  __device__ uint atomic_add(uint* address, uint val) {
      return atomicAdd(address, val);
    }

    static  __device__ float atomic_add(float* address, float val) {
      return atomicAdd(address, val);
    }
    
    /* Specialization of Atomic addition for double precision floating point */
    static  __device__ double atomic_add(double* address, double val) {
          unsigned long long int* address_as_ull =
	    (unsigned long long int*)address;
	  unsigned long long int old = *address_as_ull, assumed;
	  do {assumed = old;
	    old = atomicCAS(address_as_ull, assumed,
			    __double_as_longlong(val +
						 __longlong_as_double(assumed)));
	  } while (assumed != old);
	  return __longlong_as_double(old);
    }
  }
}
