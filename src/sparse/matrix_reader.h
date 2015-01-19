#pragma once

#include <vector>

namespace jusha {
  class MatrixImporter {
  public:
    virtual void read_matrix(const char *filename) = 0;

    int64_t num_rows() const { return m_num_rows; }
    int64_t num_cols() const { return m_num_cols; }
    int64_t num_nnzs() const { return m_num_nnzs; }

    const int *get_row_ptrs() const {  return m_row_ptrs.data(); }
    const int *get_rows() const {  return m_rows.data(); }
    const int64_t *get_cols() const {  return m_cols.data(); }
    const double *get_coefs() const {  return m_coefs.data(); }
    const double *get_sol() const {  return m_sol.data(); }
    const double *get_rhs() const {  return m_rhs.data(); }
    
  protected:
    int64_t m_num_rows;
    int64_t m_num_cols;
    int64_t m_num_nnzs;
    std::vector<int> m_row_ptrs;
    std::vector<int> m_rows;
    std::vector<int64_t> m_cols;
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

  /* convert from COO rows to CSR row ptrs 
     assume coo is sorted by row indexes
     Example: input : [0 0 0 1 1  2 3 3 3 3 ], num_rows = 4
     output is: [0 3 5 6 10]
     
   */
  void coo_to_csr_rows(const std::vector<int> coo_rows, const int &num_rows, std::vector<int> &row_ptrs);
  
}
