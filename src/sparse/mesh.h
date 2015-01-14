#pragma once

#include "cuda/array.h"

namespace yac {
  class Mesh {
  public:
  Mesh():m_dims(0), m_num_nodes(0){}
    
    void set_num_nodes(int num_nodes)
    {
      m_num_nodes = num_nodes;
      if (m_dims == 2)
        m_2d_node_pos.resize(num_nodes);
      else if (m_dims == 3)
        m_3d_node_pos.resize(num_nodes);
      else
        fprintf(stderr, "mesh dimension not supported %d\n", m_dims);
      
    }
    void set_dims(int dim) {
      m_dims = dim;
    }

    int  get_dims() const { return m_dims; }
    int  get_num_nodes() const { return m_num_nodes; }
    
    void set_node_pos(int index, float2 &pos);
    void set_node_pos(int index, float4 &pos);
  private:
    int m_dims;
    int m_num_nodes;
    jusha::cuda::MirroredArray<float2> m_2d_node_pos;
    jusha::cuda::MirroredArray<float4> m_3d_node_pos;
  };

} // namespace 
