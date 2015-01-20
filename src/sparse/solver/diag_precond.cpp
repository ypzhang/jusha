#include "./precond.h"

namespace jusha {
  DiagPrecond::DiagPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
    m_diag = matrix.get_diag();
  }


  void DiagPrecond::solve (const JVector<double> &x, JVector<double> &y) const {
    assert(x.size() == m_diag.size());
    assert(y.size() == m_diag.size());

    multiply(x, m_diag, y);
  }
  

}
