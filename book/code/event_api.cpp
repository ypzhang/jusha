cudaError_t cudaEventCreate(cudaEvent_t * event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event) ;
cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventQuery(cudaEvent_t event );
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event );

