#include <thrust/scan.h>
#include "cuda/scan_primitive.h"
#include "timer.h"

using namespace jusha;
using namespace jusha::cuda;

void test_scan(ScanPrimitive<kexclusive, int> &scan, int size) {
  thrust::device_vector<int> input(size);
  thrust::device_vector<int> output(size);
  scan.scan(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(input.data()) + size, thrust::raw_pointer_cast(output.data()));
  
  }


void test_thrust_scan(ScanPrimitive<kexclusive, int> &scan, int size) {
  thrust::device_vector<int> input(size);
  thrust::device_vector<int> output(size);
  thrust::exclusive_scan(input.begin(), input.end(), output.begin());
}


int main()
{
  ScanPrimitive<kexclusive, int> scan;
  
  // 
  // scan.run();
  // jusha::cuda_event_stop("barrier");
jusha::cuda_event_start("barrier");
   test_scan(scan, 1<<26);
     jusha::cuda_event_stop("barrier");

jusha::cuda_event_start("thrust_scan");
   test_thrust_scan(scan, 1<<26);
   jusha::cuda_event_stop("thrust_scan");

  // jusha::cuda_event_start("barrier");
  // scan.run();
  // // scan.run();

  // jusha::cuda_event_stop("barrier");
  jusha::cuda_event_print();
}
