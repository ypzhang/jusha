#include <catch.hpp>
#include <thrust/scan.h>
#include "utility.h"

#include "cuda/cuda_intrinsic.hpp"
#include "cuda/cuda_sort.hpp"

#include "cuda/array.h"

using namespace jusha;

__global__ void sum_kernel(float *val, int size)
{
  float _sum = 0.0;
  for (int i = threadIdx.x ; i < size; i+=blockDim.x)
    _sum += val[i];
  _sum = jusha::cuda::blockReduceSum(_sum);
  if (threadIdx.x == 0)
    val[0] = _sum;
}

TEST_CASE( "BlockSum", "[sum]" ) {
  cuda::MirroredArray<float> to_sum(480);
  float *sum_ptr = to_sum.getPtr();
  for (auto i = 0; i != to_sum.size(); i++)
    sum_ptr[i] = (float)i;

  sum_kernel<<<1, 512>>>(to_sum.getGpuPtr(), to_sum.size());
  printf("BLock sum is %f\n", to_sum.getElementAt(0));
}


template <bool exclusive = true>
__global__ void scan_kernel(int *scan, int size) {
  jusha::cuda::blockScan<int, thrust::plus<int>, 1024, exclusive>(scan, scan+size);
}

TEST_CASE( "BlockScan", "[scan]" ) { 
  {
  cuda::MirroredArray<int> to_scan(480);
  int *scan_ptr = to_scan.getPtr();
  for (auto i = 0; i != to_scan.size(); i++)
    scan_ptr[i] = (int)1;

  scan_kernel<true><<<1, 1024>>>(to_scan.getGpuPtr(), to_scan.size());
  //    to_scan.print("scan result", to_scan.size());
  REQUIRE((to_scan.size()-1) == to_scan[479]);
  }
  if (1)
  {
  cuda::MirroredArray<int> to_scan(2800);
  int *scan_ptr = to_scan.getPtr();
  for (auto i = 0; i != to_scan.size(); i++)
    scan_ptr[i] = (int)1;

  scan_kernel<false><<<1, 1024>>>(to_scan.getGpuPtr(), to_scan.size());
  REQUIRE(to_scan.size() == to_scan[2799]);
  //    to_scan.print("scan result", to_scan.size());

  }
  //  printf("BLock sum is %f\n", to_sum.getElementAt(0));

}

  template <typename T>
  __global__ void sort_kernel(T *sort_in, T *sort_out, int size) {
    jusha::cuda::blockSort<T, 1024>(sort_in, sort_out,  size);
  }

static void sort_test(int size, bool zero = true) {
  cuda::MirroredArray<unsigned int> to_sort(size);
  cuda::MirroredArray<unsigned int> sort_done(size);
  if (zero)
    to_sort.fill(0); //randomize();
  else
    to_sort.randomize();
  sort_kernel<<<1, 1024>>>(to_sort.getGpuPtr(), sort_done.getGpuPtr(), to_sort.size());
  REQUIRE(sort_done.isFSorted(0, sort_done.size() == true));
}

TEST_CASE( "BlockSortAllzero", "[zero]" ) { 
  sort_test(3000, true);
  }

TEST_CASE( "BlockSortCorner", "[random]" ) { 
    for (int size = 0; size != 200; size++) {
      sort_test(size);
    }
    for (int size = 201; size < 21000; size*=10) {
      sort_test(size);
    }

  }

TEST_CASE( "BlockSort", "[sort]" ) { 
  {
  cuda::MirroredArray<float> to_sort(480);
  cuda::MirroredArray<float> sort_done(480);
  float *sort_ptr = to_sort.getPtr();
  for (auto i = 0; i != to_sort.size(); i++)
    sort_ptr[i] = (to_sort.size()-i) * 2;

  sort_kernel<<<1, 1024>>>(to_sort.getGpuPtr(), sort_done.getGpuPtr(), to_sort.size());
  //  sort_done.print("sort result", to_sort.size());
  REQUIRE((sort_done.size()*2) == sort_done[479]);

  {
  cuda::MirroredArray<float> to_sort(56000);
  cuda::MirroredArray<float> sort_done(56000);
  float *sort_ptr = to_sort.getPtr();
  for (auto i = 0; i != to_sort.size(); i++)
    sort_ptr[i] = (to_sort.size()-i) * 2;

  sort_kernel<<<1, 1024>>>(to_sort.getGpuPtr(), sort_done.getGpuPtr(), to_sort.size());
  //  sort_done.print("sort result", to_sort.size());
  REQUIRE((to_sort.size()*2) == sort_done[55999]);
  }

  }

}
