cudaEvent_t begin_event, end_event;
/* create two events */
cudaError_t error = cudaEventCreate(&begin_event);
error = cudaEventCreate(&end_event);
/* start recording */
error = cudaEventRecord(begin_event, stream);
/* call some kernels */
kernel<<<...>>>(...);
/* end recording */
error = cudaEventRecord(end_event, stream);
/* The following synchronization is important to get the accurate timing */
cudaEventSynchronize(end_event);
float elapsed;
cudaEventElapsedTime(&elapsed, begin_event, end_event);
printf("elapsed %3.12f second.\n", elapsed/1000.0f);
