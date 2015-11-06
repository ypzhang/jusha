#pragma once
#include "./Operator.h"

namespace jusha {
  namespace cuda{ 
#define THREADS blockDim.x
#define BITONIC_SORT_SH_SIZE 1024
//#define BITONIC_SORT_SH_SIZE 256
/**
* Perform a cumulative count on two arrays
* @param lblock Array one
* @param rblock Array two
*/
__device__ inline void cumcount(unsigned int *lblock, unsigned int *rblock)
{

	int tx = threadIdx.x;

    int offset = 1; 
 
    __syncthreads();

	for (int d = blockDim.x>>1; d > 0; d >>= 1) // build sum in place up the tree 
    {
        __syncthreads();

		if (tx < d)    
        { 
			int aii = offset*(2*tx+1)-1;
            int bii = offset*(2*tx+2)-1;
            lblock[bii] += lblock[aii];
			rblock[bii] += rblock[aii];
		} 
        offset *= 2; 
    } 
	__syncthreads(); 
    if (tx == 0) 
	{ 
		lblock[blockDim.x] = lblock[blockDim.x-1];
		rblock[blockDim.x] = rblock[blockDim.x-1];
		lblock[blockDim.x - 1] =0;
		rblock[blockDim.x - 1] =0; 
	} // clear the last unsigned int */
	__syncthreads(); 

    for (int d = 1; d < blockDim.x; d *= 2) // traverse down tree & build scan 
    { 
        offset >>= 1; 
        __syncthreads(); 
 
        if (tx < d) 
        { 
			int aii = offset*(2*tx+1)-1; 
            int bii = offset*(2*tx+2)-1; 
 
            int t   = lblock[aii]; 
			lblock[aii]  = lblock[bii]; 
            lblock[bii] += t; 

            t   = rblock[aii]; 
            rblock[aii]  = rblock[bii]; 
            rblock[bii] += t; 

        } 
    } 
}

template<typename T, uint sortDir> static inline __device__ uint binarySearchInclusive(T val, T *data, uint L, uint stride){
    if(L == 0)
        return 0;

    uint pos = 0;
    for(; stride > 0; stride >>= 1){
        uint newPos = umin(pos + stride, L);
        if( (sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)) )
            pos = newPos;
    }

    return pos;
}

template<typename T, uint sortDir> static inline __device__ uint binarySearchExclusive(T val, T *data, uint L, uint stride){
    if(L == 0)
        return 0;

    uint pos = 0;
    for(; stride > 0; stride >>= 1){
        uint newPos = umin(pos + stride, L);
        if( (sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)) )
            pos = newPos;
    }

    return pos;
}



template <typename element, typename minmax>
__device__ inline 
void mergeSortSharedKernel(element* fromvalues, element* tovalues, unsigned int from0, unsigned int size, element *shared)
//    element *d_SrcKey,    element *d_DstKey,
//    unsigned int from0, unsigned int arrayLength,
//    element *shared
{
  element *s_key = shared;

  unsigned int coal = (from0&0xf);

  size = size + coal;
  from0 = from0 - coal;

  int sb = 2 << (int)(__log2f(size));
  //    printf("coal %d sb %d.\n", coal, sb);
  // Buffer data to be sorted in the shared memory
  for(int i=threadIdx.x;i<size;i+=THREADS)
	{
      //      printf("loading %d to idnex %d.\n", from0values[i+from], i);
		shared[i] = fromvalues[i+from0];
	}

	for(int i=threadIdx.x;i<coal;i+=THREADS)
      shared[i]=minmax::Min();

	// Pad the data
   for(int i=threadIdx.x+size;i<sb;i+=THREADS)
     {
       shared[i] = minmax::Max();
       //       printf("loading %x to index %d.\n", shared[i], i);
     }

   __syncthreads();
   /*   if (threadIdx.x == 0)
        printf("sb is %d.\n", sb);*/
    for(uint stride = 1; stride < sb; stride <<= 1){
        uint     lPos = threadIdx.x & (stride - 1);
        element *baseKey = s_key + 2 * (threadIdx.x - lPos);

        __syncthreads();
        uint keyA = baseKey[lPos +      0];
        uint keyB = baseKey[lPos + stride];
        uint posA = binarySearchExclusive<element, 1>(keyA, baseKey + stride, stride, stride) + lPos;
        uint posB = binarySearchInclusive<element, 1>(keyB, baseKey +      0, stride, stride) + lPos;

        __syncthreads();
        baseKey[posA] = keyA;
        baseKey[posB] = keyB;
    }

    __syncthreads();
	// Write back the sorted data to its correct position
	for(int i=threadIdx.x;i<size;i+=THREADS)
		if(i>=coal)
          {
            //            printf("writing back %d to %d from(%d) coal %d.\n", shared[i], i+from0, from0, coal);
            //            if (shared[i] == 65)
            //              printf("block id %d for 65.\n", blockIdx.x);
		    tovalues[i+from0] = shared[i];
          }

	__syncthreads();
    /*    if (threadIdx.x == 0 && blockIdx.x == 0)
      {
        printf("after merge sorting: \n");
        for (int i = 0; i != size; i ++)
          {
            if (i >= coal)
              printf("%d ", tovalues[i+from0]);
          }
        printf("\n");
      }
    */
}


/**
* Perform a bitonic sort
* @param values The unsigned ints to be sorted
* @param target Where to place the sorted unsigned int when done
* @param size The number of unsigned ints
*/
template <typename element, typename minmax>
__device__ inline
void bitonicSort(element* fromvalues, element* tovalues, unsigned int from0, unsigned int size, element *shared)
{
  /*  printf("calling bitonic sort from %d, size %d.\n", from0, size);
      return;*/
  //  assert(0);
  //  return;
#if 1
  //	element* shared = sarray;

	unsigned int coal = (from0&0xf);

	size = size + coal;
	from0 = from0 - coal;

	int sb = 2 << (int)(__log2f(size));
    //    printf("coal %d sb %d.\n", coal, sb);
	// Buffer data to be sorted in the shared memory
	for(int i=threadIdx.x;i<size;i+=THREADS)
	{
      //      printf("loading %d to idnex %d.\n", from0values[i+from], i);
		shared[i] = fromvalues[i+from0];
	}

	for(int i=threadIdx.x;i<coal;i+=THREADS)
      shared[i]=minmax::Min();

	// Pad the data
   for(int i=threadIdx.x+size;i<sb;i+=THREADS)
     {
       shared[i] = minmax::Max();
       //       printf("loading %x to index %d.\n", shared[i], i);
     }

    __syncthreads();
    /*    if (threadIdx.x == 0)
      {
        for (int jj = 0; jj < 512; jj++)
          {
            element tmp = shared[jj];
            printf("0x%x ", tmp);
          }
        printf("\n");
      }
      __syncthreads();*/
    // Parallel bitonic sort.
     for (int k = 2; k <= sb; k <<= 1)
       //      for (int k = 2; k <= sb; k *= 2)
    {
        // Bitonic merge:
       for (int j = k >> 1; j>0; j >>= 1)
      //      for (int j = k / 2; j>0; j /= 2)
        {
			for(int tid=threadIdx.x;tid<sb;tid+=THREADS)
			{
				unsigned int ixj = tid ^ j;
	            
				if (ixj > tid)
				{
					if ((tid & k) == 0)
					{
						if (shared[tid] > shared[ixj])
						{
                          //                          printf("swapping %d(%d) and %d(%d).\n", shared[tid], tid, shared[ixj], ixj);
                          swap<element>(shared[tid], shared[ixj]);
						}
					}
					else
					{
						if (shared[tid] < shared[ixj])
                          {
                            //                         printf("swapping %d(%d) and %d(%d).\n", shared[tid], tid, shared[ixj], ixj);
                          swap<element>(shared[tid], shared[ixj]);
						}
					}
				}
            }
            
            __syncthreads();
        }
    }
	__syncthreads();

	// Write back the sorted data to its correct position
	for(int i=threadIdx.x;i<size;i+=THREADS)
		if(i>=coal)
          {
            //            printf("writing back %d to %d from(%d) coal %d.\n", shared[i], i+from0, from0, coal);
            //            if (shared[i] == 65)
            //              printf("block id %d for 65.\n", blockIdx.x);
		    tovalues[i+from0] = shared[i];
          }
	__syncthreads();
#endif
}


/* Sort num_segs of segments in input array and write to output array
 * ranges are given in low and high arrays, inclusively
 * segment i is between input[low[i]] and input[high[i]]
 */
template <class T, int BLOCK_SIZE>
__device__ void blockSort(T *input, T *output, int size)
{
   // each segment is handled by a block
   /*   __shared__ unsigned int lphase;
        lphase=phase;*/

	// Shorthand for the threadid
	int tx = threadIdx.x;

	// Stack pointer
	__shared__ int bi;
    __shared__ unsigned int sarray[BLOCK_SIZE*2+2];
	
	// Stack unsigned ints
	__shared__ unsigned int beg[32];
	__shared__ unsigned int end[32];
	__shared__ bool flip[32];

    //TODO make its own declaration, get rid of sarray
	unsigned int* lblock = (unsigned int*)sarray;
	unsigned int* rblock = (unsigned int*)(&lblock[(blockDim.x+1)]);


	// The current pivot
	__shared__ T pivot;

	// The sequence to be sorted
	__shared__ unsigned int from;
	__shared__ unsigned int to;

	// Since we switch between the primary and the auxillary buffer,
	// these variables are required to keep track on which role
	// a buffer currently has
	__shared__ T* data;
	__shared__ T* data2;
    //	__shared__ unsigned int sbsize;

    //	__shared__ int bx;

    __shared__ T bi_shared[BITONIC_SORT_SH_SIZE];
    //	while(bx<gridDim.x)
	{

	// Thread 0 is in charge of the stack operations
	if(tx==0)
	{
		// We push our first block on the stack
		// This is the block given by the bs parameter
      beg[0] = 0;
      end[0] = size;
      //TODO?		flip[0] = bs[bx].flip;
      flip[0] = 0;
      //      sbsize = end[0] - beg[0]; 
      //      printf("size is %d begin %d value %d %d block %d.\n", end[0] - beg[0], beg[0], input[beg[0]], input[beg[0]+1], blockIdx.x);

      bi = 0;
	}

	__syncthreads();

	// If we were given an empty block there is no need to continue
	if(end[0]==beg[0])
		return;

	// While there are items left on the stack to sort
	while(bi>=0)
	{
		__syncthreads();
		// Thread 0 pops a fresh sequence from the stack
		if(tx==0)
		{
			from = beg[bi];
			to = end[bi];

            //            printf("lqsort from %d to %d size %d sbsize %d.\n", from, to, to-from, sbsize);
			// Check which buffer the sequence is in
			if(!flip[bi])
			{
				data = input;
				data2 = output;
			}
			else
			{
				data = output;
				data2 = input;
			}

		}
	

		__syncthreads();

		// If the sequence is smaller than SBSIZE we sort it using
		// an alternative sort. Otherwise each thread would sort just one
		// or two unsigned ints and that wouldn't be efficient
		if((to-from)<(BITONIC_SORT_SH_SIZE-16))
		{
			// Sort it using bitonic sort. This could be changed to some other
			// sorting method. Store the result in the final destination buffer
			// TODO
          //          if (threadIdx.x == 0) printf("switching to bitonic sort at size %d\n", to-from);
          // if((to-from>=1)&&(lphase!=2))
          
          if((to-from>=1))
            {
              //              if (threadIdx.x == 0)
              //                printf("calling bitonic sort to %d from %d, size %d blockIdx.x %d.\n", to, from, to-from, blockIdx.x);
              //                      if (bi > 4)
              //                        {
              //                          if (threadIdx.x == 0)
              //                            printf("return bi = %d.\n", bi);
              //                          return;
              //                        }
              //                      else
              bitonicSort<T, MinMax<T> >(data,output,from,to-from, bi_shared);
              //              mergeSortSharedKernel<T, MinMax<T> >(data,output,from,to-from, bi_shared);
            //            bitonicSort<T, MinMax<T> >(data,output,from,to-from, &pivot);
            }
          __syncthreads();

			// Decrement the stack pointer
			if(tx==0)
				bi--;
			__syncthreads();
			// and continue with the next sequence
			continue;
		}

 
		if(tx==0)
		{
			// Create a new pivot for the sequence
			// Try to optimize this for your input distribution
			// if you have some information about it
			T mip = min(min(data[from],data[to-1]),data[(from+to)/2]);
			T map = max(max(data[from],data[to-1]),data[(from+to)/2]);
			pivot = min(max(mip/2+map/2,mip),map);
            //            printf("first %d last %d med %d.\n", data[from].key, data[to-1].key, data[(from+to)/2].key);
            //            printf("pivot is %d mip %d map %d.\n", pivot.key, mip.key, map.key);
		}


		unsigned int ll=0;
		unsigned int lr=0;

		__syncthreads();
		
		unsigned int coal = (from)&0xf;

		if(tx+from-coal<to)
		{
			T d = data[tx+from-coal];

			if(!(tx<coal))
			{
				// Counting unsigned ints that have a higher value than the pivot
				if(d<pivot)
					ll++;
				else
				// or a lower
                  //				if(d>pivot)
					lr++;
			}
		}


		// Go through the current sequence
		for(int i=from+tx+BLOCK_SIZE-coal;i<to;i+=BLOCK_SIZE)
		{
			T d = data[i];

			// Counting unsigned ints that have a higher value than the pivot
			if(d<pivot)
				ll++;
			else
			// or a lower
              //			if(d>pivot)
				lr++;
		}

		// Store the result in a shared array so that we can calculate a
		// cumulative sum
		lblock[tx]=ll;
		rblock[tx]=lr;
		
		__syncthreads();

		// Calculate the cumulative sum
        cumcount((unsigned int*)lblock,(unsigned int*)rblock);
        //        scan_block(

		__syncthreads();

		// Let thread 0 add the new resulting subsequences to the stack
		if(tx==0)
		{
			// The sequences are in the other buffer now
			flip[bi+1] = !flip[bi];
			flip[bi] = !flip[bi];

			// We need to place the smallest object on top of the stack
			// to ensure that we don't run out of stack space
            //            printf("lblock %d rblock %d.\n", lblock[BLOCK_SIZE], rblock[BLOCK_SIZE]);
			if(lblock[BLOCK_SIZE]<rblock[BLOCK_SIZE])
			{
				beg[bi+1]=beg[bi];
				beg[bi]=to-rblock[BLOCK_SIZE];
				end[bi+1]=from+lblock[BLOCK_SIZE];

			}
			else
			{
				end[bi+1]=end[bi];
				end[bi]=from+lblock[BLOCK_SIZE];
				beg[bi+1]=to-rblock[BLOCK_SIZE];
			}
			// Increment the stack pointer
			bi++;
            assert(beg[bi] < end[bi]);  // does not work for this case 
            //            if (tx == 0)
            //              printf("bi is now %d beg %d end %d size %d block %d\n", bi, beg[bi], end[bi], end[bi]-beg[bi], blockIdx.x);
            /*            assert(bi < 32);
            if (bi > 3) 
            bi--;*/
		}
		
		__syncthreads();

		int x = from+lblock[tx+1]-1;
		int y = to-rblock[tx+1];
		
		coal = from&0xf;

		if(tx+from-coal<to)
		{
			T d = data[tx+from-coal];

			if(!(tx<coal))
			{
				if(d<pivot)
					data2[x--] = d;
				else
                  //				if(d>pivot)
					data2[y++] = d;
			}
		}

		// Go through the data once again
		// writing it to its correct position
		for(unsigned int i=from+tx+BLOCK_SIZE-coal;i<to;i+=BLOCK_SIZE)
		{	
			T d = data[i];
			
			if(d<pivot)
				data2[x--] = d;
			else
              //			if(d>pivot)
				data2[y++] = d;
			
		}

		__syncthreads();

		// As a final step, write the pivot value between the right and left
		// subsequence. Write it to the final destination since this pivot
		// is always correctly sorted
        // ignore this step because we don't differentiate between greater and equal
        /*		for(unsigned int i=from+lblock[BLOCK_SIZE]+tx;i<to-rblock[BLOCK_SIZE];i+=BLOCK_SIZE)
		{
			output[i]=pivot;
            }*/

		__syncthreads();

	}

	}

	__syncthreads();

}

// template <class T, int BLOCK_SIZE>
// __global__ void sort_block_global(T *input, T *output, 
//                           const int *low, const int *high, int num_segs)
// {
//   sort_block<T, BLOCK_SIZE>(input, output, low, high, num_segs);
//   //each block sort input from low to high 
// }


// template <class T>
// void sort_block_wrapper(int num_blocks, HiArray<T, 1> &array_input, HiArray<T, 1> &array_output, const HiArray<int, 1> &low, const HiArray<int, 1> &high)
// {
//   sort_block_global<T, BLOCK_SIZE_COMMON><<<num_blocks, BLOCK_SIZE_COMMON>>>(array_input.getGpuPtr(), array_output.getGpuPtr(), low.getReadOnlyGpuPtr(), high.getReadOnlyGpuPtr(), num_blocks);
// }

  }
}
