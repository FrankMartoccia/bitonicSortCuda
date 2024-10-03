#include "utils.h"
/*
Executes one step of bitonic merge.
"OffsetGlobal" is needed to calculate correct thread index for global bitonic merge.
"TableLen" is needed for global bitonic merge to verify if elements are still inside array boundaries.
*/
template <unsigned int threadsKernel, int sortOrder, bool isFirstStepOfPhase>
__device__ void bitonicMergeStep(
    uint32_t *keys, uint32_t *values, unsigned int offsetGlobal, unsigned int tableLen, unsigned int dataBlockLen, unsigned int stride
)
{
    // Every thread compares and exchanges 2 elements
    for (unsigned int tx = threadIdx.x; tx < dataBlockLen >> 1; tx += threadsKernel)
    {
        unsigned int indexThread = offsetGlobal + tx;
        unsigned int offset = stride;

        // In NORMALIZED bitonic sort, first STEP of every PHASE demands different offset than all other
        // STEPS. Also, in first step of every phase, offset sizes are generated in ASCENDING order
        // (normalized bitnic sort requires DESCENDING order). Because of that, we can break the loop if
        // index + offset >= length (bellow). If we want to generate offset sizes in ASCENDING order,
        // than thread indexes inside every sub-block have to be reversed.
        if (isFirstStepOfPhase)
        {
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        unsigned int index = (indexThread << 1) - (indexThread & (stride - 1));
        if (index + offset >= tableLen)
        {
            break;
        }

        compareExchange<sortOrder>(&keys[index], &keys[index + offset], &values[index], &values[index + offset]);
    }
}

/*
Sorts data with NORMALIZED bitonic sort.
*/
template <unsigned int threadsBitonicSort, unsigned int elemsBitonicSort, int sortOrder>
__global__ void normalizedBitonicSort(uint32_t *keysInput, uint32_t *keysOutput, uint32_t tableLen)
{
    extern __shared__ uint32_t bitonicSortTile[];
    unsigned int offset, dataBlockLength;
    calcDataBlockLength<threadsBitonicSort, elemsBitonicSort>(offset, dataBlockLength, tableLen);

    // Reads data from global to shared memory.
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        bitonicSortTile[tx] = keysInput[offset + tx];
    }
    __syncthreads();

    // Bitonic sort PHASES
    for (unsigned int subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        // Bitonic merge STEPS
        for (unsigned int stride = subBlockSize; stride > 0; stride >>= 1)
        {
            if (stride == subBlockSize)
            {
                bitonicMergeStep<threadsBitonicSort, sortOrder, true>(
                    bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride
                );
            }
            else
            {
                bitonicMergeStep<threadsBitonicSort, sortOrder, false>(
                    bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride
                );
            }
            __syncthreads();
        }
    }

    // Stores data from shared to global memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        keysOutput[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Global bitonic merge for sections, where stride IS GREATER than max shared memory size.
*/
template <unsigned int threadsMerge, unsigned int elemsMerge, int sortOrder, bool isFirstStepOfPhase>
__global__ void bitonicMergeGlobalKernel(uint32_t *dataTable, unsigned int tableLen, unsigned int step)
{
    unsigned int offset, dataBlockLength;
    calcDataBlockLength<threadsMerge, elemsMerge>(offset, dataBlockLength, tableLen);

    bitonicMergeStep<threadsMerge, sortOrder, isFirstStepOfPhase>(
        dataTable, offset / 2, tableLen, dataBlockLength, 1 << (step - 1)
    );
}