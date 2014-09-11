template <typename T>
void d2d_cudamemcpy(void *dst, const void *src, size_t _size)
{
  cudaMemcpy(dst, src, _size*sizeof(T), cudaMemcpyDefault);
}
