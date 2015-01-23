#pragma once

#include "./matrix.h"
#include "cuda/array.h"

namespace jusha {
  
template <typename T>
void split_diag_coefs(const int64_t &num_rows, const int32_t *row_ptrs, const int64_t *cols, const T* coefs, T *diag, T *offd);


/* CSR matrix format
 * If it is a square matrix, the coeficients are separated to diagnal and off-diagonal 
 * if not, only m_coefs are available 
 * m_rowptrs are changed accordingly 
 */
template <typename T>
class CsrMatrix: public Matrix<T>{
public:
 CsrMatrix(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, const T *coefs): Matrix<T>(nrows, ncols) {
    init(nrows, ncols, row_ptrs, cols, coefs);
  }
    
  virtual void init(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, const T *coefs);

  virtual const JVector<T> &get_diag() const ;
  //  virtual const JVector<T> &get_offd() const ;
  //  virtual const JVector<T> &get_coef() const ;
private:
  size_t m_nnz;
  JVector<int32_t> m_row_ptrs;
  JVector<int64_t> m_cols;
  JVector<T>       m_coefs;
  JVector<T>       m_diag;
  JVector<T>       m_offd;
  
};


} // namespace jusha
