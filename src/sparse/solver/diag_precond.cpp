#include "./precond.h"

namespace jusha {
  DiagPrecond::DiagPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
    m_diag = matrix.get_diag();
  }

  JVector<double> DiagPrecond::solve (const JVector<double> &x) const {
    assert(x.size() == m_diag.size());    
    JVector<double> y(x.size());

    multiply(x, m_diag, y);
    // for (int i = 0; i < x.size(); i++)
    //   y(i) = x(i) * diag(i);

    return y;
  }

}
