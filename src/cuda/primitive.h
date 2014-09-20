#ifndef JUSHA_CUDA_PRIMITIVE_H
#define JUSHA_CUDA_PRIMITIVE_H

namespace jusha {
  namespace cuda {
    class Primitive {
    public:
      virtual void run() = 0;
      void get_gpu_property(cudaDeviceProp &gpu_property) { 
        jusha::cuda::get_cuda_property(gpu_property);
      }
    protected:
      //     
      //    private:
    };
  }
}

#endif
