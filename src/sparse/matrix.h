#pragma once
#include "cuda/array.h"

namespace jusha {

  template <typename T>
  class Matrix {
  public:
    Matrix(int64_t nrows, int64_t ncols): m_num_rows(nrows), m_num_cols(ncols){}
    virtual const JVector<T> &get_diag() const = 0;

    int64_t get_num_rows() {  return m_num_rows; }
    int64_t get_num_cols() {  return m_num_cols; }    
  protected:
    int64_t m_num_rows;
    int64_t m_num_cols;    
  };

  // utilities:
  /* template <typename T> */
  /*   void AXPY(const T alpha, const Matrix<T> &x, const T beta, cuda::MirroredArray &y); */
}

