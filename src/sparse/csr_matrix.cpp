#include "./csr_matrix.h"

namespace jusha {

  /* separate the coefs to off-diagnal and diagnals */
template <class T>
void  split_diag_coefs(const int64_t num_rows, const JVector<int32_t> &row_ptrs, const JVector<int64_t> &cols, const JVector<T> &coefs, JVector<T> &diag, JVector<T> &offd)
{
  diag.resize(num_rows);
  assert(coefs.size() >= num_rows);
  offd.resize(coefs.size() - num_rows);
  split_diag_coefs(num_rows, row_ptrs.getReadOnlyGpuPtr(), cols.getReadOnlyGpuPtr(), coefs.getReadOnlyGpuPtr(), diag.getGpuPtr(), offd.getGpuPtr());
}
  
template <class T>
void CsrMatrix<T>::init(int64_t nrows, int64_t ncols, const int32_t *row_ptrs, const int64_t *cols, const T *coefs)
{
  Matrix<T>::m_num_rows = nrows;
  Matrix<T>::m_num_cols = ncols;
  m_nnz = row_ptrs[nrows];
  m_row_ptrs.init(row_ptrs, nrows+1);
  m_cols.init(cols, m_nnz);
  m_coefs.init(coefs, m_nnz);

  if (nrows == ncols) {
    split_diag_coefs(nrows, m_row_ptrs, m_cols, m_coefs, m_diag, m_offd);
    //    m_coefs.clear(); // save memory for square matrix
  }
}

template <class T>  
const JVector<T> &CsrMatrix<T>::get_diag() const
{
  return m_diag;
}

// template <class T>  
// const JVector<T> &CsrMatrix<T>::get_offd() const
// {
//   return m_offd;
// }


  /* Instantiating */
template class CsrMatrix<double>;

} // namespace 
