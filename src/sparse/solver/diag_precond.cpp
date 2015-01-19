#include "./precond.h"

namespace jusha {
  DiagPrecond::DiagPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
    
  }

  JVector<double> DiagPrecond::solve (const JVector<double> &x) const {
    JVector<double> sol;
    return sol;
  }

}
