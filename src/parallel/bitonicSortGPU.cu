#include <iostream>
#include "utils.cuh"
#include "constants.h"
#include "cuda_runtime.h"

/*
Executes one step of bitonic merge.
This device function compares and exchanges two elements for each thread.
"offsetGlobal" is the global index offset used for accessing the correct elements.
*/
__device__ void bitonicMergeStep(
    uint32_t *values, unsigned int offsetGlobal, unsigned int tableLen, unsigned int dataBlockLen, unsigned int stride, unsigned int threadsKernel,
    int sortOrder, bool isFirstStepOfPhase
)
{
    // Each thread will compare and exchange 2 elements in the bitonic merge step
    for (unsigned int tx = threadIdx.x; tx < dataBlockLen >> 1; tx += threadsKernel)
    {
        unsigned int indexThread = offsetGlobal + tx;
        unsigned int offset = stride;

        // Special handling for the first step of every phase (normalized bitonic sort requires this)
        if (isFirstStepOfPhase)
        {
            // Calculate offset and reverse thread indices within sub-blocks for ascending order
            offset = ((indexThread & (stride - 1)) << 1) + 1;
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        unsigned int index = (indexThread << 1) - (indexThread & (stride - 1));

        // Check array bounds to avoid invalid memory access
        if (index + offset >= tableLen)
        {
            break;
        }

        // Compare and exchange elements based on the sort order
        compareExchange(&values[index], &values[index + offset], sortOrder);
    }
}

/*
Normalized Bitonic Sort Kernel.
This kernel sorts blocks of input data using shared memory for better performance.
*/
__global__ void normalizedBitonicSort(
    uint32_t *keysInput, uint32_t *keysOutput, uint32_t tableLen, unsigned int threadsBitonicSort,
    unsigned int elemsBitonicSort, int sortOrder)
{
    // Shared memory to hold the tile being sorted
    extern __shared__ uint32_t bitonicSortTile[];
    unsigned int offset, dataBlockLength;

    // Calculate block-specific data length
    calcDataBlockLength(offset, dataBlockLength, tableLen, threadsBitonicSort, elemsBitonicSort);

    // Copy data from global memory to shared memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        bitonicSortTile[tx] = keysInput[offset + tx];
    }
    __syncthreads();

    // Perform the bitonic sorting phases
    for (unsigned int subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
    {
        for (unsigned int stride = subBlockSize; stride > 0; stride >>= 1)
        {
            if (stride == subBlockSize)
            {
                // First step of each phase
                bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, threadsBitonicSort, sortOrder, true);
            }
            else
            {
                // Subsequent steps
                bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, threadsBitonicSort, sortOrder, false);
            }
            __syncthreads();
        }
    }

    // Copy the sorted data back to global memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += threadsBitonicSort)
    {
        keysOutput[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Global Bitonic Merge Kernel.
Handles merging of data blocks larger than shared memory.
*/
__global__ void bitonicMergeGlobalKernel(
    uint32_t *dataTable, unsigned int tableLen, unsigned int step, unsigned int threadsMerge,
    unsigned int elemsMerge, int sortOrder, bool isFirstStepOfPhase)
{
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, tableLen, threadsMerge, elemsMerge);

    bitonicMergeStep(
        dataTable, offset / 2, tableLen, dataBlockLength, 1 << (step - 1), threadsMerge, sortOrder, isFirstStepOfPhase
    );
}

/*
Launches the kernel for bitonic sorting using shared memory.
*/
void runBitonicSortKernel(uint32_t *d_values, unsigned int arrayLength, int sortOrder)
{
    unsigned int elemsPerThreadBlock = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;
    unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

    // Define grid and block dimensions for kernel launch
    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_BITONIC_SORT, 1, 1);

    // Launch the normalized bitonic sort kernel
    normalizedBitonicSort <<<dimGrid, dimBlock, sharedMemSize>>>(
        d_values, d_values, arrayLength, THREADS_BITONIC_SORT, ELEMENTS_BITONIC_SORT, sortOrder
    );
}

/*
Launches the kernel for global bitonic merging.
*/
void runBitonicMergeGlobalKernel(
    uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder)
{
    unsigned int elemsPerThreadBlock = THREADS_GLOBAL_MERGE * ELEMENTS_GLOBAL_MERGE;

    // Define grid and block dimensions for the merge kernel
    dim3 dimGrid((arrayLength - 1) / elemsPerThreadBlock + 1, 1, 1);
    dim3 dimBlock(THREADS_GLOBAL_MERGE, 1, 1);

    // Launch the global bitonic merge kernel
    bitonicMergeGlobalKernel <<<dimGrid, dimBlock>>>(
        d_values, arrayLength, step, THREADS_GLOBAL_MERGE, ELEMENTS_GLOBAL_MERGE, sortOrder, phase == step
    );
}

/*
Main function to execute parallel bitonic sort on GPU.
*/
void bitonicSortParallel(uint32_t *d_values, unsigned int array_length, int sortOrder)
{
    // Calculate the next power of 2 for the array length
    unsigned int arrayLenPower2 = nextPowerOf2(array_length);
    unsigned int elemsPerBlockBitonicSort = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;

    // Calculate the number of phases for the bitonic sort and merge
    unsigned int phasesBitonicSort = log2(static_cast<double>(min(arrayLenPower2, elemsPerBlockBitonicSort)));
    unsigned int phasesAll = log2(static_cast<double>(arrayLenPower2));

    // Sort sub-blocks of input data using bitonic sort
    runBitonicSortKernel(d_values, array_length, sortOrder);

    // Perform global bitonic merge
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        for (unsigned int step = phase; step >= 1; step--)
        {
            runBitonicMergeGlobalKernel(d_values, array_length, phase, step, sortOrder);
        }
    }
}
