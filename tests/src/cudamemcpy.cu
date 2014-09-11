template <typename T>
void d2d_cudamemcpy(T *dst, const T *src, size_t _size)
{
  cudaMemcpy(dst, src, _size*sizeof(T), cudaMemcpyDefault);
}
