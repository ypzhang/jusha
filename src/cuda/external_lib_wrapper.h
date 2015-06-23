/* wrapper classes from template to third-party libraries */
#include <curand.h>
#include <assert.h>
//#include <>

namespace jusha {
  namespace cuda {
    template <curandRngType_t rng_type, class T>
      struct RandomWrapper
      {
        static curandGenerator_t cuRandgen;
        static curandGenerator_t getGen() { 
          if (cuRandgen == 0)
            {
              curandStatus_t status = curandCreateGenerator(&cuRandgen,  rng_type);
              jassert(status == CURAND_STATUS_SUCCESS);
            }
          return cuRandgen;
        }

        void apply(T *buffer, size_t num)
        {
          //    #error "can't be here"
          assert(0);
        }
      };


    template <curandRngType_t rng_type>
      struct RandomWrapper<rng_type, unsigned int>
    {
      static curandGenerator_t cuRandgen;
      static curandGenerator_t getGen() { 
        if (cuRandgen == 0)
          {
            curandStatus_t status = curandCreateGenerator(&cuRandgen,  rng_type);
            jassert(status == CURAND_STATUS_SUCCESS);
          }
        return cuRandgen;
      }
      void apply(unsigned int *buffer, size_t num)
      {
        curandStatus_t status = curandGenerate(getGen(), buffer, num);
        jassert(status == CURAND_STATUS_SUCCESS);
      }
    };


    /*template <curandRngType_t rng_type>
      struct RandomWrapper<rng_type, int>
      {
      static curandGenerator_t cuRandgen;
      static curandGenerator_t getGen() { 
      if (cuRandgen == 0)
      {
      curandStatus_t status = curandCreateGenerator(&cuRandgen,  rng_type);
      assert(status == CURAND_STATUS_SUCCESS);
      }
      return cuRandgen;
      }
      void apply(int *buffer, size_t num)
      {
      curandStatus_t status = curandGenerate(getGen(), buffer, num);
      assert(status == CURAND_STATUS_SUCCESS);
      }
      };*/


    // TODO move them to cpp
    template <curandRngType_t rng_type> curandGenerator_t RandomWrapper<rng_type, unsigned int>::cuRandgen = 0;
    //template <curandRngType_t rng_type> curandGenerator_t RandomWrapper<rng_type, int>::cuRandgen = 0;
    template <curandRngType_t rng_type, class T> curandGenerator_t RandomWrapper<rng_type, T>::cuRandgen = 0;


  } // cuda
} // jusha
