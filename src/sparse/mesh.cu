#include "mesh.h"

namespace yac {
  void Mesh::set_node_pos(int index, float2 &pos)
  {
    assert(m_dims == 2);
    assert(m_2d_node_pos.size() > index);
    float2 *h_pos = m_2d_node_pos.getPtr();
    h_pos[index] = pos;
  }

  void Mesh::set_node_pos(int index, float4 &pos)
  {
    assert(m_dims == 3);
    assert(m_3d_node_pos.size() > index);
    float4 *h_pos = m_3d_node_pos.getPtr();
    h_pos[index] = pos;
  }

}
