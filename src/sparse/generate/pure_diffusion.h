#pragma once

#include "matrix.h"
// 1D pure diffusion problem (heat conduction)
//  d         d \phi
// -- (\gamma ------ ) + S_{\phi} = 0
// dx         dx
// boundary condition: fixed temperature
/*! \param[in] L: length in meter
 *  \param[in] N: number of points
 *  \param[in] gamma: 
 *  \param[in] S:
 *  \param[in] T_left:
 *  \param[in] T_right:
 *  \param[output] M: the matrix
 *  \param[output] rhs: the right hand side
 */
namespace yac {
void create_pure_diffusion_1d(double L, int N,  double gamma, double S, double T_left, double T_right
                              , jusha::matrix<double> &M, std::vector<double> &rhs);

} // namespace 
