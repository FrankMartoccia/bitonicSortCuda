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
    uint32_t *values, unsigned int offsetGlobal, unsigned int arrayLength, unsigned int dataBlockLen, unsigned int stride,
    int sortOrder, int numThreads, bool isFirstStepOfPhase
)
{
    // Each thread will compare and exchange 2 elements in the bitonic merge step
    for (unsigned int tx = threadIdx.x; tx < dataBlockLen >> 1; tx += numThreads)
    {
        unsigned int indexThread = offsetGlobal + tx;
        unsigned int offset = stride;

        // Special handling for the first step of every phase (normalized bitonic sort requires this)
        if (isFirstStepOfPhase)
        {
            // Calculate offset and reverse thread indices within sub-blocks for ascending order
            offset = ((indexThread % stride) * 2) + 1; // "+1 is added to have the offset odd"
            // Recalculate indexThread to mirror its corresponding index within the current stride group.
            // 1. (indexThread / stride) * stride:
            //    - Aligns indexThread to the start of its group (of size 'stride').
            // 2. (indexThread % stride):
            //    - Finds the relative position of indexThread within its group.
            // 3. (stride - 1) - (indexThread % stride):
            //    - Mirrors the thread's position within the group, reversing the order of threads in the current stride.
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        // Calculate the index used in compareExchange()
        unsigned int index = (indexThread * 2) - (indexThread % stride);

        // Check array bounds to avoid invalid memory access
        if (index + offset >= arrayLength)
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
    uint32_t *valuesGlobal, uint32_t arrayLength, int sortOrder)
{
    // Shared memory to hold the tile being sorted
    extern __shared__ uint32_t bitonicSortTile[];
    unsigned int offset, dataBlockLength;

    // Calculate block-specific data length
    calcDataBlockLength(offset, dataBlockLength, arrayLength, BITONIC_BLOCKS);

    // Copy data from global memory to shared memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_BITONIC_SORT)
    {
        bitonicSortTile[tx] = valuesGlobal[offset + tx];
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
                bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, sortOrder,
                    THREADS_BITONIC_SORT, true);
            }
            else
            {
                // Subsequent steps
                bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, sortOrder,
                    THREADS_BITONIC_SORT, false);
            }
            __syncthreads();
        }
    }

    // Copy the sorted data back to global memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += THREADS_BITONIC_SORT)
    {
        valuesGlobal[offset + tx] = bitonicSortTile[tx];
    }
}

/*
Global Bitonic Merge Kernel.
Handles merging of data blocks larger than shared memory.
*/
__global__ void bitonicMergeGlobalKernel(
    uint32_t *dataTable, unsigned int arrayLength, unsigned int step, int sortOrder, bool isFirstStepOfPhase)
{
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, arrayLength, MERGE_BLOCKS);

    bitonicMergeStep(
        dataTable, offset / 2, arrayLength, dataBlockLength, 1 << (step - 1), sortOrder,
        THREADS_GLOBAL_MERGE, isFirstStepOfPhase
    );
}

/*
Launches the kernel for bitonic sorting using shared memory.
*/
void runBitonicSortKernel(uint32_t *d_values, unsigned int arrayLength, int sortOrder)
{
    unsigned int elemsPerThreadBlock = arrayLength / BITONIC_BLOCKS;
    unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

    // Define grid and block dimensions for kernel launch
    dim3 dimGrid(BITONIC_BLOCKS, 1, 1);
    dim3 dimBlock(THREADS_BITONIC_SORT, 1, 1);

    // Launch the normalized bitonic sort kernel
    normalizedBitonicSort <<<dimGrid, dimBlock, sharedMemSize>>>(
        d_values, arrayLength, sortOrder
    );
}

/*
Launches the kernel for global bitonic merging.
*/
void runBitonicMergeGlobalKernel(
    uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder)
{
    // unsigned int elemsPerThreadBlock = arrayLength / (THREADS_GLOBAL_MERGE * MERGE_BLOCKS);

    // Define grid and block dimensions for the merge kernel
    dim3 dimGrid(MERGE_BLOCKS, 1, 1);
    dim3 dimBlock(THREADS_GLOBAL_MERGE, 1, 1);

    // Launch the global bitonic merge kernel
    bitonicMergeGlobalKernel <<<dimGrid, dimBlock>>>(
        d_values, arrayLength, step, sortOrder, phase == step
    );
}

/*
Main function to execute parallel bitonic sort on GPU.
*/
void bitonicSortParallel(uint32_t *d_values, unsigned int array_length, int sortOrder)
{
    // Calculate the next power of 2 for the array length
    unsigned int arrayLenPower2 = nextPowerOf2(array_length);
    unsigned int elemsPerBlockBitonicSort = array_length / BITONIC_BLOCKS;

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
