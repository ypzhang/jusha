#include "./precond.h"

namespace jusha {
  template <class T>
    void jacobi(const Matrix<T> &matrix, const JVector<T> &x, JVector<T> &y)
  {
    const JVector<T> &offd = matrix.get_offd();
    const JVector<T> &diag = matrix.get_diag();
    
  }
  
  
  JacobiPrecond::JacobiPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
  }


  void JacobiPrecond::solve (const JVector<double> &x, JVector<double> &y) const {
    assert(x.size() == m_matrix->get_num_rows());
    assert(y.size() == m_matrix->get_num_rows());

    ///    jacobi(m_matrix, x, y);
  }

}
