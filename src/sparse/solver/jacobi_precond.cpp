#include "./precond.h"

namespace jusha {
  
  
  JacobiPrecond::JacobiPrecond(const Matrix<double> &matrix): PrecondBase(matrix) {
  }


  void JacobiPrecond::solve (const JVector<double> &x, JVector<double> &y) const {
    const Matrix<double> *matrix = get_matrix();
    assert(x.size() == matrix->get_num_rows());
    assert(y.size() == matrix->get_num_rows());

    // const JVector<int> &rows = matrix->get_rows();
    // const JVector<int> &cols = matrix->get_cols();    
    ///    jacobi(m_matrix, x, y);
  }

}
