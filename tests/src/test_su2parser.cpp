#include "catch.hpp"
#include "./test_data.h"
#include "sparse/parser/su2_mesh_parser.h"
#include "sparse/mesh.h"

using namespace yac;
TEST_CASE( "TestSuMeshParser", "[File_not_exist]" ) {
  Su2MeshParser parser;
  Mesh mesh;
  parser.parse_file("not_existing.su2", mesh);
}

TEST_CASE( "naca0012", "[naca0012]" ) {
  Su2MeshParser parser;
  Mesh mesh;
  parser.parse_file(mesh_NACA0012_inv_filename.c_str(), mesh);
  REQUIRE( mesh.get_dims() == 2);
  REQUIRE( mesh.get_num_nodes() == 5233);
}

