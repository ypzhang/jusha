#ifndef JUSHA_CUDA_PRIMITIVE_H
#define JUSHA_CUDA_PRIMITIVE_H

namespace jusha {
  namespace cuda {
    class Primitive {
    public:
      virtual void run() = 0;
    };
  }
}

#endif
