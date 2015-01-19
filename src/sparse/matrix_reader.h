#pragma once

#include <vector>

namespace jusha {
  class MatrixImporter {
  public:
    virtual void read_matrix(const char *filename) = 0;
  protected:
    std::vector<int> m_row_ptrs;
    std::vector<int> m_cols;
    std::vector<double> m_coefs;
    std::vector<double> m_sol;
    std::vector<double> m_rhs;
  };    


  class Hdf5Matrix: public MatrixImporter {
  public:
    virtual void read_matrix(const char *filename);
  private:
  };


  class MatrixMarket: public MatrixImporter {
  public:
    virtual void read_matrix(const char *filename);
  };
  
}
