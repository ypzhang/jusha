#pragma once
#include "./sparse/matrix.h"

namespace jusha {
  /********************************************************************
   *           Preconditioner Base Class
   ********************************************************************/
  class PrecondBase{
  public:
    PrecondBase(const Matrix<double> &matrix): m_matrix(&matrix) {}
    virtual ~PrecondBase() {}

    // All preconditioner needs to implement this interface:
    virtual void solve (const JVector<double> &x, JVector<double> &y) const = 0;

    virtual JVector<double> solve (const JVector<double> &x) const
    {
      JVector<double> y(x.size());
      solve(x, y);
      return y;
    }
    
    //    JVector<double> trans_solve (const JVector<double> &x) const;

    /* const double&         diag(int i) const { return diag_(i); } */
    /* double&           diag(int i) { return diag_(i); } */

    const Matrix<double> *get_matrix() const { return m_matrix; }
  private:
    const Matrix<double> *m_matrix = 0;
  };


  /********************************************************************
   *           Diagonal Preconditioner 
   ********************************************************************/
  class DiagPrecond : public PrecondBase {
  public:
    DiagPrecond(const Matrix<double> &);

    //    JVector<double> solve (const JVector<double> &x) const;
    void solve (const JVector<double> &x, JVector<double> &y) const;    

  private:
    JVector<double> m_diag;
  };

  

  /********************************************************************
   *           Jacobi Preconditioner 
   ********************************************************************/
  /* template <class M> */
  /*   void jacobi(const M &matrix, const JVector<T> &x, JVector<T> &y) */
  /*   { */
  /*     const JVector<T> &diag = matrix.get_diag(); */
  /*     const Jvector<T> &csr_rowptr = matrix */
  /*   } */
  
  
  class JacobiPrecond : public PrecondBase {
  public:
    JacobiPrecond(const Matrix<double> &);

    //    JVector<double> solve (const JVector<double> &x) const;
    void solve (const JVector<double> &x, JVector<double> &y) const;
    
  private:
    
  };
  
}
