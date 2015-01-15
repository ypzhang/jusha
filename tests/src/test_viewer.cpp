#include "catch.hpp"
#include "sparse/parser/su2_mesh_parser.h"
#include "sparse/mesh.h"
#include "2d_viewer.h"

using namespace yac;

TEST_CASE( "Test2dViewer", "[naca0012]" ) {
  Su2MeshParser parser;
  Mesh mesh;
  parser.parse_file("mesh_NACA0012_inv.su2", mesh);
  REQUIRE( mesh.get_dims() == 2);
  REQUIRE( mesh.get_num_nodes() == 5233);
  Viewer2d viewer;
  viewer.init(0, NULL, mesh);
  
}

