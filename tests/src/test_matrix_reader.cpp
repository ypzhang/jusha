#include "catch.hpp"
#include "./test_data.h"
#include "sparse/matrix_reader.h"

using namespace jusha;

namespace {
TEST_CASE( "TestMatrixReader", "[2cubes_sphere]" ) {

  MatrixMarket mm;
  mm.read_matrix(cubes2_sphere_filename.c_str());  
  REQUIRE(mm.num_rows() == 101492);
  REQUIRE(mm.num_nnzs() == 874378);
  //  printf("rows %d nnzs %d.\n", mm.num_rows(), mm.num_nnzs());
}

TEST_CASE( "TestCoo2Csr", "[simple01]" ) {
  std::vector<int> coo{0, 0,0, 1,1, 2, 3, 3, 3, 3 };
  int num_rows = 4;
  std::vector<int> csr;
  coo_to_csr_rows(coo, num_rows, csr);
  REQUIRE(csr.size() == (num_rows+1));
  REQUIRE(csr[0] == 0);
  REQUIRE(csr[1] == 3);
  REQUIRE(csr[2] == 5);
  REQUIRE(csr[3] == 6);
  REQUIRE(csr[4] == 10);
  //  printf("rows %d nnzs %d.\n", mm.num_rows(), mm.num_nnzs());
}


TEST_CASE( "TestCoo2Csr02", "[simple]" ) {
  std::vector<int> coo{0, 0, 0,  2, 3, 3, 3, 3 };
  int num_rows = 4;
  std::vector<int> csr;
  coo_to_csr_rows(coo, num_rows, csr);
  REQUIRE(csr.size() == (num_rows+1));
  REQUIRE(csr[0] == 0);
  REQUIRE(csr[1] == 3);
  REQUIRE(csr[2] == 3);
  REQUIRE(csr[3] == 4);
  REQUIRE(csr[4] == 8);  
}

TEST_CASE( "TestCoo2Csr03", "[simple]" ) {
  std::vector<int> coo{1, 1, 1,  2,  5, 6, 6 };
  int num_rows = 7;
  std::vector<int> csr;
  coo_to_csr_rows(coo, num_rows, csr);
  REQUIRE(csr.size() == (num_rows+1));
  REQUIRE(csr[0] == 0);
  REQUIRE(csr[1] == 0);
  REQUIRE(csr[2] == 3);
  REQUIRE(csr[3] == 4);
  REQUIRE(csr[4] == 4);
  REQUIRE(csr[5] == 4);
  REQUIRE(csr[6] == 5);
  REQUIRE(csr[7] == 7);    
}

}
