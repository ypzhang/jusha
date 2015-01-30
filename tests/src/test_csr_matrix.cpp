#include "catch.hpp"
#include "./test_data.h"
#include "sparse/csr_matrix.h"
#include "sparse/matrix_reader.h"

using namespace jusha;

namespace {

TEST_CASE( "test constructor", "[CSR Matrix]" ) {
  MatrixMarket mm;
  cudaSetDevice(0);
  check_cuda_error("cudasetdevice", __FILE__, __LINE__);
  mm.read_matrix(cubes2_sphere_filename.c_str());
  REQUIRE(mm.num_rows() == 101492);
  REQUIRE(mm.num_nnzs() == 874378);

  SECTION ("init csr from csr format", "[CSR Matrix]")  {

    CsrMatrix<double> csr_matrix(mm.num_rows(), mm.num_cols(), mm.get_row_ptrs(), mm.get_cols(),
				 mm.get_coefs());
    
    REQUIRE(csr_matrix.get_num_rows() == mm.num_rows());
    
    JVector<double> coefs = csr_matrix.get_coef();
    //  REQUIRE(coefs.size() == mm.num_rows());
    REQUIRE(coefs[0] == 364006.0716019724);
    
    //  REQUIRE(diag[1] == 504358.74952916644);
  }
  
  SECTION ("init csr from coo format", "[CSR Matrix]")  {

    CsrMatrix<double> csr_matrix;
    csr_matrix.init_from_coo(mm.num_rows(), mm.num_cols(), mm.num_nnzs(), mm.get_rows(), mm.get_cols(), mm.get_coefs());
    REQUIRE(csr_matrix.get_num_rows() == mm.num_rows());
    
    JVector<double> coefs = csr_matrix.get_coef();
    //  REQUIRE(coefs.size() == mm.num_rows());
    REQUIRE(coefs[0] == 364006.0716019724);
    
    //  REQUIRE(diag[1] == 504358.74952916644);
  }
}

// TEST_CASE( "test csr row to coo row", "[CSR Matrix]" ) {
//   std::vector<int> coo{0, 0,0, 1,1, 2, 3, 3, 3, 3 };
//   int num_rows = 4;
//   std::vector<int> csr;
//   coo_to_csr_rows(coo, num_rows, csr);

//   MatrixMarket mm;
//   mm.read_matrix(cubes2_sphere_filename.c_str());
//   REQUIRE(mm.num_rows() == 101492);
//   REQUIRE(mm.num_nnzs() == 874378);

//   CsrMatrix<double> csr_matrix(mm.num_rows(), mm.num_cols(), mm.get_row_ptrs(), mm.get_cols(),
//                                mm.get_coefs());

//   REQUIRE(csr_matrix.get_num_rows() == mm.num_rows());
//   JVector<double> diag = csr_matrix.get_diag();
//   REQUIRE(diag.size() == mm.num_rows());
//   REQUIRE(diag[0] == 364006.0716019724);
//   REQUIRE(diag[1] == 504358.74952916644);
// }
TEST_CASE( "A simple case", "[Test CSR to COO]" ) {
  //  std::vector<int> coo{0, 0,0, 1,1, 2, 3, 3, 3, 3 };
  std::vector<int> stl_csr_row{0, 3,5, 6, 10};
  JVector<int> csr_row;
  csr_row.init(stl_csr_row.data(), stl_csr_row.size());
  JVector<int> coo_row(10);  
  csr_row_to_coo_row(csr_row, stl_csr_row.size()-1, stl_csr_row.back(), coo_row);
  // REQUIRE(csr.size() == (num_rows+1));
  // REQUIRE(csr[0] == 0);
  // REQUIRE(csr[1] == 3);
  // REQUIRE(csr[2] == 5);
  // REQUIRE(csr[3] == 6);
  // REQUIRE(csr[4] == 10);
  //  printf("rows %d nnzs %d.\n", mm.num_rows(), mm.num_nnzs());
}
  
TEST_CASE( "A simple case2", "[Test CSR to COO]" ) {
  //  std::vector<int> coo{0, 0,0, 1,1, 2, 3, 3, 3, 3 };
  // std::vector<int> stl_csr_row{0, 3,5, 6, 10};
  // JVector<int> csr_row;
  // csr_row.init(stl_csr_row.data(), stl_csr_row.size());
  // JVector<int> coo_row(10);  
  // csr_row_to_coo_row(csr_row, stl_csr_row.size()-1, stl_csr_row.back(), coo_row);
  // REQUIRE(csr.size() == (num_rows+1));
  // REQUIRE(csr[0] == 0);
  // REQUIRE(csr[1] == 3);
  // REQUIRE(csr[2] == 5);
  // REQUIRE(csr[3] == 6);
  // REQUIRE(csr[4] == 10);
  //  printf("rows %d nnzs %d.\n", mm.num_rows(), mm.num_nnzs());
}
}
