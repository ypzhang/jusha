#include <cassert>
#include "pure_diffusion.h"

namespace yac {
void create_pure_diffusion_1d(double L, int N,  double gamma, double S, double T_left, double T_right
                              , matrix<double> &M, std::vector<double> &rhs)
{
  // at least include an non-boundary node
  assert(N >= 3);
  assert(L > 0.0);
  double delta_x = L/N;
  int nrows = N;
  std::vector<int32_t> row_ptrs(nrows+1);
  std::vector<int64_t> cols;
  std::vector<double> coefs;
  row_ptrs[0] = 0;
  
  M.init(nrows, nrows, &(row_ptrs[0]), &(cols[0]), &(coefs[0]));
  
         
}

} // namespace 
