#pragma once
#include "cuda/array.h"

namespace jusha {

  typedef int32_t row_type;
  typedef int64_t col_type;

  template <typename T>
  class Matrix {
  public:
    Matrix(): m_total_rows(0), m_num_rows(0), m_num_cols(0){}    
    Matrix(int64_t nrows, int64_t ncols): m_num_rows(nrows), m_num_cols(ncols){}
    //    virtual const JVector<T> &get_offd() const = 0;
    //    virtual const JVector<T> &get_coef() const = 0;        

    int64_t get_total_rows() const {  return m_total_rows; }    
    int32_t get_num_rows() const {  return m_num_rows; }
    int64_t get_num_cols() const {  return m_num_cols; }

    virtual const JVector<int> &get_rows() const = 0;
    virtual const JVector<col_type> &get_cols() const = 0;

    int64_t num_rows() const { return m_num_rows; }
  protected:
    int64_t m_total_rows;
    int64_t m_num_rows;
    int64_t m_num_cols;    
  };

  // utilities:
  /* template <typename T> */
  /*   void AXPY(const T alpha, const Matrix<T> &x, const T beta, cuda::MirroredArray &y); */
}

