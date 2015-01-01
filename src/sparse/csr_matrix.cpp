#include "./csr_matrix.h"

namespace jusha {
template <class T>
void CsrMatrix<T>::init(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, T *coefs)
{
  Matrix<T>::m_num_rows = nrows;
  Matrix<T>::m_num_cols = ncols;
  m_nnz = row_ptrs[nrows];
  m_row_ptrs.init(row_ptrs, nrows+1);
  m_cols.init(cols, m_nnz);
  m_coefs.init(coefs, m_nnz);
}


} // namespace 
