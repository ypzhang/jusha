#include "./array.h"
#include "./for_each.h"

namespace jusha {
  namespace cuda {
    template <class T>
    void plus(const JVector<T> &lhs, const JVector<T> &rhs, JVector<T> &result)
    {
      assert(lhs.size() == rhs.size());
      assert(result.size() == rhs.size());
      
    }
  }
}
