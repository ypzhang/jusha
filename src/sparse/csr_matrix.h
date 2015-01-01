#pragma once

#include "./matrix.h"
#include "cuda/array.h"

namespace jusha {
template <typename T>
class CsrMatrix: public Matrix<T>{
public:
 CsrMatrix(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, T *coefs): Matrix<T>(nrows, ncols) {
    init(nrows, ncols, row_ptrs, cols, coefs);
  }
    
  virtual void init(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, T *coefs);
private:
  size_t m_nnz;
  cuda::MirroredArray<int32_t> m_row_ptrs;
  cuda::MirroredArray<int64_t> m_cols;
  cuda::MirroredArray<T> m_coefs;
  
};


} // namespace jusha
