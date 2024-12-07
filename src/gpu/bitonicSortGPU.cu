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
            offset = ((indexThread % stride) * 2) + 1; // +1 is added to have the offset odd
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
This kernel sorts blocks of input data.
*/
__global__ void normalizedBitonicSort(
    uint32_t *valuesGlobal, uint32_t arrayLength, int sortOrder, bool isOptimized)
{
    unsigned int offset, dataBlockLength;

    // Calculate block-specific data length
    calcDataBlockLength(offset, dataBlockLength, arrayLength, BITONIC_SORT_BLOCKS);

    if (isOptimized) // Use shared memory version
    {
        // Shared memory to hold the tile being sorted
        extern __shared__ uint32_t bitonicSortTile[];

        // Copy data from global memory to shared memory
        for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
        {
            bitonicSortTile[tx] = valuesGlobal[offset + tx];
        }
        __syncthreads();

        // Perform the bitonic sorting phases using shared memory
        for (unsigned int subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
        {
            for (unsigned int stride = subBlockSize; stride > 0; stride >>= 1)
            {
                if (stride == subBlockSize)
                {
                    // First step of each phase
                    bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, true);
                }
                else
                {
                    // Subsequent steps
                    bitonicMergeStep(bitonicSortTile, 0, dataBlockLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, false);
                }
                __syncthreads();
            }
        }

        // Copy the sorted data back to global memory
        for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
        {
            valuesGlobal[offset + tx] = bitonicSortTile[tx];
        }
    }
    else
    {
        // Perform bitonic sorting phases directly using global memory
        for (unsigned int subBlockSize = 1; subBlockSize < dataBlockLength; subBlockSize <<= 1)
        {
            for (unsigned int stride = subBlockSize; stride > 0; stride >>= 1)
            {
                if (stride == subBlockSize)
                {
                    // First step of each phase
                    bitonicMergeStep(valuesGlobal, offset / 2, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, true);
                }
                else
                {
                    // Subsequent steps
                    bitonicMergeStep(valuesGlobal, offset / 2, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, false);
                }
                __syncthreads();
            }
        }
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
    calcDataBlockLength(offset, dataBlockLength, arrayLength, BITONIC_SORT_BLOCKS);

    bitonicMergeStep(
        dataTable, offset / 2, arrayLength, dataBlockLength, 1 << (step - 1), sortOrder,
        BITONIC_SORT_THREADS, isFirstStepOfPhase
    );
}

/*
This kernel merges blocks of data in shared memory to improve performance.
Each thread block handles a portion of the data, performing the merge locally
within shared memory before writing the results back to global memory.
*/
__global__ void bitonicMergeLocalKernel(
    uint32_t *d_values, unsigned int arrayLength, unsigned int step, int sortOrder, bool isFirstStepOfPhase) {

    extern __shared__ uint32_t mergeTile[];
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, arrayLength, BITONIC_SORT_BLOCKS);

    uint32_t *valuesTile = mergeTile;

    // Reads data from global to shared memory.
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
    {
        valuesTile[tx] = d_values[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    for (unsigned int stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeStep(valuesTile, 0, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, true);
        }
        else
        {
            bitonicMergeStep(valuesTile, 0, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, false);
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
    {
        d_values[offset + tx] = valuesTile[tx];
    }
}

/*
Launches the kernel for bitonic sorting using shared memory.
*/
void runBitonicSortKernel(uint32_t *d_values, unsigned int arrayLength, int sortOrder, bool isOptimized)
{
    // Define grid and block dimensions for kernel launch
    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    if (isOptimized) {
        // Calculate shared memory size
        unsigned int elemsPerThreadBlock = arrayLength / BITONIC_SORT_BLOCKS;
        unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

        // Launch the normalized bitonic sort kernel with shared memory
        normalizedBitonicSort <<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, arrayLength, sortOrder, true
        );
    }
    else {
        // Launch the normalized bitonic sort kernel without shared memory
        normalizedBitonicSort <<<dimGrid, dimBlock>>>(
            d_values, arrayLength, sortOrder, false
        );
    }
}

/*
Launches the kernel for global bitonic merging.
*/
void runBitonicMergeGlobalKernel(
    uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder)
{
    // Define grid and block dimensions for the merge kernel
    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    // Launch the global bitonic merge kernel
    bitonicMergeGlobalKernel <<<dimGrid, dimBlock>>>(
        d_values, arrayLength, step, sortOrder, phase == step
    );
}

/*
Launches the kernel for local bitonic merging using shared memory.
This function calculates the required shared memory size and launches
the `bitonicMergeLocalKernel` for merging smaller blocks of data in parallel.
*/
void runBitonicMergeLocalKernel(uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder) {

    unsigned int elemsPerThreadBlock = arrayLength / BITONIC_SORT_BLOCKS;
    unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    if (phase == step) {
        bitonicMergeLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, arrayLength, step, sortOrder, true);
    }
    else
    {
        bitonicMergeLocalKernel<<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, arrayLength, step, sortOrder, false);
    }
}

/*
Bitonic sort version 1 (non-optimized version)
*/
void bitonicSortV1(uint32_t *d_values, unsigned int arrayLength, int sortOrder,
                   unsigned int phasesBitonicSort, unsigned int phasesAll)
{
    // Sort sub-blocks of input data using bitonic sort
    runBitonicSortKernel(d_values, arrayLength, sortOrder, false);

    // Perform global bitonic merge
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        for (unsigned int step = phase; step >= 1; step--)
        {
            runBitonicMergeGlobalKernel(d_values, arrayLength, phase, step, sortOrder);
        }
    }
}

/*
Bitonic sort version 2 (optimized version)
*/
void bitonicSortV2(uint32_t *d_values, unsigned int arrayLength, int sortOrder,
                   unsigned int phasesBitonicSort, unsigned int phasesAll)
{
    // Sort sub-blocks of input data using bitonic sort
    runBitonicSortKernel(d_values, arrayLength, sortOrder, true);

    // Perform global bitonic merge
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        unsigned int step = phase;
        while (step > phasesBitonicSort)
        {
            runBitonicMergeGlobalKernel(d_values, arrayLength, phase, step, sortOrder);
            step--;
        }
        runBitonicMergeLocalKernel(d_values, arrayLength, phase, step, sortOrder);
    }
}

/*
Main function to execute parallel bitonic sort on GPU
*/
void bitonicSortParallel(uint32_t *d_values, unsigned int arrayLength, int sortOrder)
{
    // Calculate the next power of 2 for the array length
    unsigned int arrayLenPower2 = nextPowerOf2(arrayLength);

    // Calculate block size parameters
    unsigned int elemsPerBlockBitonicSort = arrayLength / BITONIC_SORT_BLOCKS;
    unsigned int sharedMemoryUsage = elemsPerBlockBitonicSort * sizeof(uint32_t);

    // Calculate the number of phases for bitonic sort
    // and the total number of phases for all sub-arrays
    unsigned int phasesBitonicSort = log2(static_cast<double>(min(arrayLenPower2, elemsPerBlockBitonicSort)));
    unsigned int phasesAll = log2(static_cast<double>(arrayLenPower2));

    // Check shared memory availability and call appropriate version
    if (sharedMemoryUsage > MAX_SHARED_MEMORY_SIZE)
    {
        std::cout << "[GPU] - Using non optimized version" << std::endl;
        bitonicSortV1(d_values, arrayLength, sortOrder, phasesBitonicSort, phasesAll);
    }
    else
    {
        std::cout << "[GPU] - Using optimized version" << std::endl;
        bitonicSortV2(d_values, arrayLength, sortOrder, phasesBitonicSort, phasesAll);
    }
}

