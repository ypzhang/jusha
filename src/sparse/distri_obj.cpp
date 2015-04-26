#include <numeric> 
#include <functional>
#include "distri_obj.h"

namespace jusha {
  void DistriObj::set_partition(const std::vector<int> &partition, int rank) {
    m_rank = rank;
    m_size = (int)partition.size();
    m_partition = partition;
    m_partition_scan.resize(m_size+1);
    m_partition_scan[0] = 0;
    std::partial_sum(m_partition.begin(), m_partition.end(), m_partition_scan.begin()+1);
  }
  
}
