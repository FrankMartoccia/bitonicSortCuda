#include "utils.cuh"
#include "constants.h"
#include "cuda_runtime.h"
/*
Executes one step of bitonic merge.
"OffsetGlobal" is needed to calculate correct thread index for global bitonic merge.
"TableLen" is needed for global bitonic merge to verify if elements are still inside array boundaries.
*/
__device__ void bitonicMergeStep(
    uint32_t *values, unsigned int offsetGlobal, unsigned int tableLen, unsigned int dataBlockLen, unsigned int stride, unsigned int threadsKernel,
    int sortOrder, bool isFirstStepOfPhase
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

        compareExchange(&values[index], &values[index + offset], sortOrder);
    }
}

/*
Sorts data with NORMALIZED bitonic sort.
*/
__global__ void normalizedBitonicSort(uint32_t *keysInput, uint32_t *keysOutput, uint32_t tableLen,
    unsigned int threadsBitonicSort, unsigned int elemsBitonicSort, int sortOrder)
{
    extern __shared__ uint32_t bitonicSortTile[];
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, tableLen, threadsBitonicSort, elemsBitonicSort);

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
                bitonicMergeStep(
                    bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, threadsBitonicSort, sortOrder, true
                );
            }
            else
            {
                bitonicMergeStep(
                    bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride,
                    threadsBitonicSort, sortOrder, false
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
__global__ void bitonicMergeGlobalKernel(uint32_t *dataTable, unsigned int tableLen, unsigned int step, unsigned int threadsMerge,
    unsigned int elemsMerge, int sortOrder, bool isFirstStepOfPhase)
{
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, tableLen, threadsMerge, elemsMerge);

    bitonicMergeStep(
        dataTable, offset / 2, tableLen, dataBlockLength, 1 << (step - 1), threadsMerge, sortOrder,
        isFirstStepOfPhase
    );
}

/*
Sorts sub-blocks of input data with bitonic sort.
*/
void runBitonicSortKernel(uint32_t *d_values, unsigned int arrayLength, int sortOrder)
{
    unsigned int elemsPerThreadBlock = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;
    unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_BITONIC_SORT, 1, 1);

    normalizedBitonicSort <<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, d_values, arrayLength, THREADS_BITONIC_SORT, ELEMENTS_BITONIC_SORT,
            sortOrder);
}

/*
Merges array, if data blocks are larger than shared memory size. It executes only one STEP of one PHASE per
kernel launch.
*/
void runBitonicMergeGlobalKernel(uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder)
{
    unsigned int elemsPerThreadBlock = THREADS_GLOBAL_MERGE * ELEMENTS_GLOBAL_MERGE;

    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_GLOBAL_MERGE, 1, 1);

    bool isFirstStepOfPhase = phase == step;
    if (isFirstStepOfPhase)
        {
            bitonicMergeGlobalKernel
                <<<dimGrid, dimBlock>>>(
                d_values, arrayLength, step, THREADS_GLOBAL_MERGE, ELEMENTS_GLOBAL_MERGE, sortOrder, true
            );
        }
    else
        {
            bitonicMergeGlobalKernel
                <<<dimGrid, dimBlock>>>(
                d_values, arrayLength, step, THREADS_GLOBAL_MERGE, ELEMENTS_GLOBAL_MERGE, sortOrder,
                false
            );
        }
}


void bitonicSortParallel(uint32_t *d_values, unsigned int array_length, int sortOrder) {
    unsigned int arrayLenPower2 = nextPowerOf2(array_length);
    unsigned int elemsPerBlockBitonicSort = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;

    // Number of phases, which can be executed in shared memory (stride is lower than shared memory size)
    unsigned int phasesBitonicSort = log2((double)min(arrayLenPower2, elemsPerBlockBitonicSort));
    unsigned int phasesAll = log2((double)arrayLenPower2);

    // Sorts blocks of input data with bitonic sort
    runBitonicSortKernel(
        d_values, array_length, sortOrder
    );

    // Bitonic merge using only the global merge kernel
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        for (unsigned int step = phase; step >= 1; step--)
        {
            runBitonicMergeGlobalKernel(
                d_values, array_length, phase, step, sortOrder
            );
        }
    }
}