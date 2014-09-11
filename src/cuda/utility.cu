namespace jusha {
void check_cuda_error(const char *kernelname, int line_no, const char *file, cudaStream_t stream)  {
#ifdef DEBUG
  //#if 1
  cudaError_t err;
  if (stream)
    gampack_stream_synchronize(stream) ;//    cudaStreamSynchronize(stream);
  else
    err = cudaThreadSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf (stdout, "CUDA error in at thread %d kernel %s @ file %s line %d:\n  %s\n",
             omp_get_thread_num(), kernelname, file, line_no, cudaGetErrorString(err));
    printBacktrace(10);
    abort();
  }
#endif
}

}
