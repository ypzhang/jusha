#pragma once
#include "./sparse/matrix.h"

namespace jusha {
  class PrecondBase{
  public:
    PrecondBase(const Matrix<double> &matrix): m_matrix(&matrix) {}
    virtual ~PrecondBase() {}

    // All preconditioner needs to implement this interface:
    virtual JVector<double> solve (const JVector<double> &x) const = 0;
    //    JVector<double> trans_solve (const JVector<double> &x) const;

    /* const double&         diag(int i) const { return diag_(i); } */
    /* double&           diag(int i) { return diag_(i); } */
  protected:
    const Matrix<double> *m_matrix = 0;
  };

  class DiagPrecond : public PrecondBase {
  public:
    DiagPrecond(const Matrix<double> &);

    JVector<double> solve (const JVector<double> &x) const;

  private:
    JVector<double> m_diag;
  };
}
