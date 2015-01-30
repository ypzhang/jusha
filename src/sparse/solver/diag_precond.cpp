#include "./precond.h"

namespace jusha {
  DiagPrecond::DiagPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
  }


  void DiagPrecond::solve (const JVector<double> &x, JVector<double> &y) const {
    assert(x.size() == m_diag.size());
    assert(y.size() == m_diag.size());

    multiply(x, m_diag, y);
  }
  

}
