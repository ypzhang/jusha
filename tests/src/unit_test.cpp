#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch.hpp>

#include "utility.h"
using namespace jusha;
TEST_CASE( "Alwayspass", "[Simple]" ) {

}


TEST_CASE( "Utility", "[perf_timer]" ) {
  double start = jusha_get_wtime();
  double end = jusha_get_wtime();
    fprintf(stderr, "start %1.5f end %1.5f.\n", start, end);
  REQUIRE (start < end);
  REQUIRE (start > 0.0);
  REQUIRE (  end > 0.0);

}
