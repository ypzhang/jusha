#include "catch.hpp"
#include "./test_data.h"
#include "sparse/csr_matrix.h"
#include "sparse/matrix_reader.h"

using namespace jusha;

namespace {

TEST_CASE( "TestCsrMatrixDiag", "[manual]" ) {
  MatrixMarket mm;
  mm.read_matrix(cubes2_sphere_filename.c_str());
  REQUIRE(mm.num_rows() == 101492);
  REQUIRE(mm.num_nnzs() == 874378);

  CsrMatrix<double> csr_matrix(mm.num_rows(), mm.num_cols(), mm.get_row_ptrs(), mm.get_cols(),
                               mm.get_coefs());

  REQUIRE(csr_matrix.get_num_rows() == mm.num_rows());
  JVector<double> diag = csr_matrix.get_diag();
  REQUIRE(diag.size() == mm.num_rows());
  REQUIRE(diag[0] == 364006.0716019724);
  REQUIRE(diag[1] == 504358.74952916644);

}
  
}
