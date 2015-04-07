#pragma once

namespace jusha {
  template <class M, class Precond, class T>
  void jacobi_precondition(const M<T> &matrix, const Precond &precond, const Vector<T> &x,
			   Vector<T> &y)
  {
    const Vector<row_type> row_ptrs = matrix.get_csr_rows();
    const Vector<T> &coefs = matrix.get_coef();
    const Vector<col_type> cols = matrix.get_cols();
    assert(x.size() == matrix.get_num_rows();
    
    
  }
}
