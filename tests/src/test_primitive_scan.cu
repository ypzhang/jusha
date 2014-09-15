#include "cuda/scan_primivite.h"

using namespace jusha;
using namespace jusha::cuda;

int main()
{
  ScanPrimitive<kexclusive, int> scan;
  scan.run();
  scan.run();
  scan.run();
}
