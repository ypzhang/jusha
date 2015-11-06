#pragma once
#include <limits.h>
#include <assert.h>




template <class T>
struct AddOP
{
  static __device__ __host__  T apply(T a, T b) { return a + b; }
  static __device__ __host__  T identity() { return T(0); }
};

template <class T>
struct MulOP
{
  static __device__ __host__  T apply(T a, T b) { return a * b; }
  static __device__ __host__  T identity() { return T(0); }
};


template <class T>
struct MinMax
{
  static __device__ __host__  T Min() { assert(0); return 0; } 
  static __device__ __host__  T Max() { assert(0); return 0; }
};

// specials 
template <>
struct AddOP <float>
{
  static __device__ __host__  float apply(float a, float b) { return a + b; }
  static __device__ __host__  float identity() { return 0.0f; }
};



template <>
struct MinMax <int>
{
  static __device__ __host__  int Min() { return INT_MIN; } 
  static __device__ __host__  int Max() { return INT_MAX; } 
};

template <>
struct MinMax <unsigned int>
{
  static __device__ __host__  unsigned int Min() { return 0; } 
  static __device__ __host__  unsigned int Max() { return 0xFFFFFFFF; } 
};

// TODO
template <>
struct MinMax <float>
{
  static __device__ __host__  float Min() {  return 0; }
  static __device__ __host__  float Max() {  return 30e30; } 
};

template <>
struct MinMax <double>
{
  static __device__ __host__  double Min() { assert(0); return 0/*DBL_MIN*/; } 
  static __device__ __host__  double Max() { assert(0); return 0/*DBL_MAX*/; } 
};



template <class T>
struct MinOP
{
  static __device__ __host__  T apply(T a, T b) { /*printf("compare %d and %d.\n", a, b);*/ return a < b? a : b; };
  static __device__ __host__  T identity() { return MinMax<T>::Max(); /* return max */}
};

template <class T>
struct MaxOP
{
  static __device__ __host__  T apply(T a, T b) { return a < b? b : a; }
  static __device__ __host__  T identity() { return MinMax<T>::Min(); /* return min */ }
};


template <class T>
__device__ __host__  inline void swap(T& a, T& b)
{
  //  printf("swapping %d and %d.\n", a, b);
  T tmp = a;
  a = b;
  b = tmp;
}

