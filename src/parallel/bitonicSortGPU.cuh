#ifndef BITONIC_SORT_GPU_CUH
#define BITONIC_SORT_GPU_CUH

#include <cstdint>

// Function to perform one step of bitonic merge
// This is a device function executed by each thread during the bitonic merge phase.
// "offsetGlobal" - Global offset to compute the correct thread index.
// "arrayLength" - Total length of the data table being sorted.
// "dataBlockLen" - The length of the data block handled by this thread.
// "stride" - Controls the distance between compared elements in each step.
// "sortOrder" - Order in which to sort (ascending or descending).
// "isFirstStepOfPhase" - Boolean flag indicating whether it's the first step of the phase.
__device__ void bitonicMergeStep(
    uint32_t *values,
    unsigned int offsetGlobal,
    unsigned int arrayLength,
    unsigned int dataBlockLen,
    unsigned int stride,
    int sortOrder,
    bool isFirstStepOfPhase
);

// Kernel function to perform normalized bitonic sort
// This is the main kernel for the bitonic sort algorithm, working on shared memory to achieve better performance.
// "valuesGlobal" - Array of values to be sorted.
// "arrayLength" - Length of the data table.
// "sortOrder" - Order of sorting (ascending or descending).
__global__ void normalizedBitonicSort(
    uint32_t *valuesGlobal,
    uint32_t arrayLength,
    int sortOrder
);

// Kernel function for global bitonic merge
// This kernel handles the merging step for blocks of data larger than shared memory.
// "dataTable" - Data array to merge.
// "tableLen" - Length of the data table.
// "step" - Current step of the merging phase.
// "sortOrder" - Sorting order (ascending or descending).
// "isFirstStepOfPhase" - Flag indicating if this is the first step of the phase.
__global__ void bitonicMergeGlobalKernel(
    uint32_t *dataTable,
    unsigned int tableLen,
    unsigned int step,
    int sortOrder,
    bool isFirstStepOfPhase
);

// Host function to launch the normalized bitonic sort kernel
// This function launches the kernel to perform bitonic sort on blocks of data in shared memory.
// "d_values" - Pointer to the data array on the device.
// "arrayLength" - Length of the array to be sorted.
// "sortOrder" - Order of sorting (ascending or descending).
void runBitonicSortKernel(
    uint32_t *d_values,
    unsigned int arrayLength,
    int sortOrder
);

// Host function to launch the global bitonic merge kernel
// This function is used to launch the bitonic merge kernel on blocks that are too large to fit in shared memory.
// "d_values" - Pointer to the data array on the device.
// "arrayLength" - Length of the array.
// "phase" - Current phase of the merge.
// "step" - Current step within the phase.
// "sortOrder" - Order of sorting (ascending or descending).
void runBitonicMergeGlobalKernel(
    uint32_t *d_values,
    unsigned int arrayLength,
    unsigned int phase,
    unsigned int step,
    int sortOrder
);

// Host function for parallel bitonic sort
// This function orchestrates the entire sorting process on the GPU, including bitonic sorting and merging.
// "d_values" - Pointer to the data array on the device.
// "array_length" - Length of the array.
// "sortOrder" - Order of sorting (ascending or descending).
void bitonicSortParallel(
    uint32_t *d_values,
    unsigned int array_length,
    int sortOrder
);

#endif // BITONIC_SORT_GPU_CUH
