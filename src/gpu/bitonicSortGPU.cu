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
    // Each thread processes one or more element pairs within the current data block.
    // Threads advance in steps of `numThreads` to ensure the entire data block is covered.
    for (unsigned int tx = threadIdx.x; tx < dataBlockLen >> 1; tx += numThreads)
    {
        // Calculate the global index of the current thread's element pair.
        unsigned int indexThread = offsetGlobal + tx;
        unsigned int offset = stride; // Default distance between elements being compared.

        // Special handling for the first step of a phase:
        // Normalize thread indices and modify offsets to match the specific bitonic sort structure.
        if (isFirstStepOfPhase)
        {
            // Calculate a stride-based offset and reverse thread indices within the stride.
            offset = ((indexThread % stride) * 2) + 1; // Offset must be odd for the first step.
            // Reverse the thread index within its sub-block:
            // - Each thread operates within a sub-block of size `stride`.
            // - This transformation ensures threads access elements in a reversed order within the sub-block.
            // - Steps:
            //    1. `indexThread / stride` calculates the sub-block index (base index of the block).
            //    2. `indexThread % stride` calculates the thread's offset within the sub-block.
            //    3. `((stride - 1) - (indexThread % stride))` reverses the offset within the block.
            //    4. Combine the sub-block base index and the reversed offset to compute the new `indexThread`.
            indexThread = (indexThread / stride) * stride + ((stride - 1) - (indexThread % stride));
        }

        // Determine the index of the first element in the pair being compared.
        unsigned int index = (indexThread * 2) - (indexThread % stride);

        // Check if the indices are within bounds to avoid accessing out-of-range memory.
        if (index + offset >= arrayLength)
        {
            break;
        }

        // Compare and exchange elements based on the specified sort order (ascending/descending).
        compareExchange(&values[index], &values[index + offset], sortOrder);
    }
}

/*
Normalized Bitonic Sort Kernel.
This kernel sorts blocks of input data.
*/
__global__ void bitonicSort(
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
        for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS) {
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
__global__ void bitonicMergeGlobal(
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
__global__ void bitonicMergeLocal(
    uint32_t *d_values, unsigned int arrayLength, unsigned int step, int sortOrder, bool isFirstStepOfPhase) {

    extern __shared__ uint32_t mergeTile[];
    unsigned int offset, dataBlockLength;
    calcDataBlockLength(offset, dataBlockLength, arrayLength, BITONIC_SORT_BLOCKS);

    // Reads data from global to shared memory.
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
    {
        mergeTile[tx] = d_values[offset + tx];
    }
    __syncthreads();

    // Bitonic merge
    // Loop through decreasing powers of two starting from 2^(step - 1)
    for (unsigned int stride = 1 << (step - 1); stride > 0; stride >>= 1)
    {
        if (isFirstStepOfPhase)
        {
            bitonicMergeStep(mergeTile, 0, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, true);
        }
        else
        {
            bitonicMergeStep(mergeTile, 0, arrayLength, dataBlockLength, stride, sortOrder, BITONIC_SORT_THREADS, false);
        }
        __syncthreads();
    }

    // Stores data from shared to global memory
    for (unsigned int tx = threadIdx.x; tx < dataBlockLength; tx += BITONIC_SORT_THREADS)
    {
        d_values[offset + tx] = mergeTile[tx];
    }
}

/*
Launches the kernel for bitonic sorting.
*/
void runBitonicSort(uint32_t *d_values, unsigned int arrayLength, int sortOrder, bool isOptimized)
{
    // Define grid and block dimensions for kernel launch
    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    if (isOptimized) {
        // Calculate shared memory size
        unsigned int elemsPerThreadBlock = arrayLength / BITONIC_SORT_BLOCKS;
        unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

        // Launch the normalized bitonic sort kernel with shared memory
        bitonicSort <<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, arrayLength, sortOrder, true
        );
    }
    else {
        // Launch the normalized bitonic sort kernel without shared memory
        bitonicSort <<<dimGrid, dimBlock>>>(
            d_values, arrayLength, sortOrder, false
        );
    }
}

/*
Launches the kernel for global bitonic merging.
*/
void runBitonicMergeGlobal(
    uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder)
{
    // Define grid and block dimensions for the merge kernel
    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    // Launch the global bitonic merge kernel
    bitonicMergeGlobal <<<dimGrid, dimBlock>>>(
        d_values, arrayLength, step, sortOrder, phase == step
    );
}

/*
Launches the kernel for local bitonic merging using shared memory.
This function calculates the required shared memory size and launches
the `bitonicMergeLocalKernel` for merging smaller blocks of data in parallel.
*/
void runBitonicMergeLocal(uint32_t *d_values, unsigned int arrayLength, unsigned int phase, unsigned int step, int sortOrder) {

    unsigned int elemsPerThreadBlock = arrayLength / BITONIC_SORT_BLOCKS;
    unsigned int sharedMemSize = elemsPerThreadBlock * sizeof(*d_values);

    dim3 dimGrid(BITONIC_SORT_BLOCKS, 1, 1);
    dim3 dimBlock(BITONIC_SORT_THREADS, 1, 1);

    if (phase == step) {
        bitonicMergeLocal<<<dimGrid, dimBlock, sharedMemSize>>>(
            d_values, arrayLength, step, sortOrder, true);
    }
    else
    {
        bitonicMergeLocal<<<dimGrid, dimBlock, sharedMemSize>>>(
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
    runBitonicSort(d_values, arrayLength, sortOrder, false);

    // Perform global bitonic merge
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        for (unsigned int step = phase; step >= 1; step--)
        {
            runBitonicMergeGlobal(d_values, arrayLength, phase, step, sortOrder);
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
    runBitonicSort(d_values, arrayLength, sortOrder, true);

    // Perform global bitonic merge
    for (unsigned int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        unsigned int step = phase;
        while (step > phasesBitonicSort)
        {
            runBitonicMergeGlobal(d_values, arrayLength, phase, step, sortOrder);
            step--;
        }
        runBitonicMergeLocal(d_values, arrayLength, phase, step, sortOrder);
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

