#pragma once

#include <cuda.h>

namespace jusha {
  namespace cuda {
    typedef float3 Float3;
    // struct Float3 {
    //   float3 val;

    //   __host__ __device__ Float3() 
    //   { val.x = 0.0f;
    //     val.y = 0.0f;
    //     val.z = 0.0f;
    //   }

    //   __host__ __device__ Float3(const Float3 &rhs) {
    //      *this = rhs;
    //    }

    //   __host__ __device__ Float3(int i) {
    //     val.x = i;
    //     val.y = i;
    //     val.z = i;
    //    }


    //   __host__ __device__ Float3 &operator=(const Float3 &rhs) {
    //     val = rhs.val;
    //     return *this;
    //   }

    // };

  }
}
