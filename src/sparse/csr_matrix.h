#pragma once

#include "./matrix.h"
#include "cuda/array.h"

namespace jusha {
  
template <typename T>
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const T* coefs, T *diag, T *offd);

template <typename T>
void make_diag_first(const int64_t &num_rows, const int32_t *row_ptrs, int64_t *cols, T *coefs);
  
void csr_row_to_coo_row(const JVector<int32_t> &csr_rows, const int32_t nrows, const int32_t nnzs, JVector<int32_t> &coo_rows); 
/* CSR matrix format
 * If it is a square matrix, the coeficients are separated to diagnal and off-diagonal 
 * if not, only m_coefs are available 
 * m_rowptrs are changed accordingly 
 */
  typedef uint32_t rowptr_t;
template <typename T>
class CsrMatrix: public Matrix<T>{
public:
  CsrMatrix() : m_nnz(0) {} 
 CsrMatrix(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, const T *coefs): Matrix<T>(nrows, ncols) {
    init(nrows, ncols, row_ptrs, cols, coefs);
  }
    
  virtual void init(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, const T *coefs);
  virtual void init_from_coo(int64_t nrows, int64_t ncols, int64_t nnz, const int32_t *rows, const int64_t *cols, const T *coefs);  


  const JVector<int> &get_rows() const ;
  const JVector<col_type> &get_cols() const ;
  const JVector<int> &get_csr_rows() const;  
  
  //  virtual const JVector<T> &get_offd() const ;
  virtual const JVector<T> &get_coef() const { return m_coefs; }
private:
  void convert_rowptrs_to_rows();
  size_t m_nnz;
  JVector<row_type> m_row_ptrs;
  JVector<int32_t> m_rows; // coo style rows
  JVector<col_type> m_cols;
  JVector<T>       m_coefs;
  
  //  JVector<T>       m_diag;
  //  JVector<T>       m_offd;
  
};

  template <typename T>
  void jacobi_smooth(const CsrMatrix<T> &matrix, const JVector<T> &x, const JVector<T> &y);
} // namespace jusha
