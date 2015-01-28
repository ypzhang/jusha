#include <vector>
#include "./csr_matrix.h"
using namespace std;
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
void CsrMatrix<T>::init_from_coo(int64_t nrows, int64_t ncols, int64_t nnz, const int32_t *rows, const int64_t *cols, const T *coefs)
{
  // if square matrix, separate diag and off-diag on CPU (will do this on GPU if necessary)
  std::vector<int> nodiag_rows;
  std::vector<int64_t> nodiag_cols;
  std::vector<T>  nodiag_coefs;
  std::vector<T>  diag_coefs(nrows);
  int64_t n_diags(0);
  if (nrows == ncols) {
    for (auto i = 0; i != nnz; i++) {
      if (rows[i] != cols[i]) {
	nodiag_rows.push_back(rows[i]);
	nodiag_cols.push_back(cols[i]);
	nodiag_coefs.push_back(coefs[i]);
      } else {
	diag_coefs[rows[i]] = coefs[i];
	n_diags++;
      }
    }
  }
  if (n_diags != nrows)
    fprintf(stderr, "Square matrix expecting %zd diagnal values, only found %zd.\n", nrows, n_diags);

  int64_t offd_nnz = nnz - n_diags;
  Matrix<T>::m_num_rows = nrows;
  Matrix<T>::m_num_cols = ncols;
  m_nnz = nnz;
  //  m_row_ptrs.init(row_ptrs, nrows+1);
  m_rows.init(nodiag_rows.data(), offd_nnz);
  m_cols.init(nodiag_cols.data(), offd_nnz);
  m_diag.init(diag_coefs.data(), nrows);
  m_offd.init(nodiag_coefs.data(), offd_nnz);
}

template <class T>  
const JVector<T> &CsrMatrix<T>::get_diag() const
{
  return m_diag;
}

template <class T>  
const JVector<int> &CsrMatrix<T>::get_rows() const
{
  return m_rows;
}

template <class T>  
const JVector<int64_t> &CsrMatrix<T>::get_cols() const
{
  return m_cols;
}
  
// template <class T>  
// const JVector<T> &CsrMatrix<T>::get_offd() const
// {
//   return m_offd;
// }


  /* Instantiating */
template class CsrMatrix<double>;

} // namespace 
