#pragma once
#include <inttypes.h>
#include <vector>
#include <cassert>

namespace jusha {
  class DistriObj {
  public:
    DistriObj(): m_rank(0), 
                 m_size(1)
    {
      m_partition.resize(1, 0);
      m_partition_scan.resize(2, 0);
    }
    
    void set_partition(const std::vector<int> &partition, int rank);

    int get_my_size() const {
      assert(m_rank < static_cast<int>(m_partition.size()));
      return m_partition[m_rank];
    }

    int64_t get_my_start() const {
      assert(m_rank < static_cast<int>(m_partition_scan.size()));
      return m_partition_scan[m_rank];
    }

    int64_t get_my_end() const {
      assert((m_rank+1) < static_cast<int>(m_partition_scan.size()));
      return m_partition_scan[m_rank+1];
    }

    int get_partition_size() const { return m_size; }
  private:
    int m_rank = 0;
    int m_size = 1;
    std::vector<int> m_partition;
    std::vector<int64_t> m_partition_scan;
  };

}
