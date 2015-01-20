#include "./precond.h"

namespace jusha {
  JacobiPrecond::JacobiPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
  }


  void JacobiPrecond::solve (const JVector<double> &x, JVector<double> &y) const {
    assert(x.size() == m_matrix->get_num_rows());
    assert(y.size() == m_matrix->get_num_rows());

    //    jacobi(m_matrix, x, y);
  }

}
