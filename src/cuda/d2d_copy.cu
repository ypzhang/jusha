
namespace jusha {

template <typename T> 
d2d_memcpy_kernel(T *dst, const T *src, size_t _size) {
  size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  for (; gid < _size; gid++) {
    dst[gid] = src[gid];
  }
}


  void d2d_memcpy(T *dst, const T *src, size_t _size) {
    
  }  


}
