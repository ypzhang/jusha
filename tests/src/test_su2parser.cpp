#include "catch.hpp"
#include "sparse/parser/su2_mesh_parser.h"

using namespace yac;
TEST_CASE( "TestSuMeshParser", "[File_not_exist]" ) {
  Su2MeshParser parser;
  parser.parse_file("not_existing.su2");
}

TEST_CASE( "naca0012", "[naca0012]" ) {
  Su2MeshParser parser;
  parser.parse_file("mesh_NACA0012_inv.su2");
      
}

