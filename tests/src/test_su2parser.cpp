#include "catch.hpp"
#include "sparse/parser/su2_mesh_parser.h"

using namespace yac;
TEST_CASE( "TestSuMeshParser", "[File_not_exist]" ) {
  Su2MeshParser parser;
  parser.parse_file("not_existing.su2");
      
}

