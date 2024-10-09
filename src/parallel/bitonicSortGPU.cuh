#ifndef BITONIC_SORT_GPU_CUH
#define BITONIC_SORT_GPU_CUH

#include <cstdint>

// Function to perform one step of bitonic merge
__device__ void bitonicMergeStep(
    uint32_t *values,
    unsigned int offsetGlobal,
    unsigned int tableLen,
    unsigned int dataBlockLen,
    unsigned int stride,
    unsigned int threadsKernel,
    int sortOrder,
    bool isFirstStepOfPhase
);

// Kernel function to perform normalized bitonic sort
__global__ void normalizedBitonicSort(
    uint32_t *keysInput,
    uint32_t *keysOutput,
    uint32_t tableLen,
    unsigned int threadsBitonicSort,
    unsigned int elemsBitonicSort,
    int sortOrder
);

// Kernel function for global bitonic merge
__global__ void bitonicMergeGlobalKernel(
    uint32_t *dataTable,
    unsigned int tableLen,
    unsigned int step,
    unsigned int threadsMerge,
    unsigned int elemsMerge,
    int sortOrder,
    bool isFirstStepOfPhase
);

// Host function to launch the normalized bitonic sort kernel
void runBitonicSortKernel(
    uint32_t *d_values,
    unsigned int arrayLength,
    int sortOrder
);

// Host function to launch the global bitonic merge kernel
void runBitonicMergeGlobalKernel(
    uint32_t *d_values,
    unsigned int arrayLength,
    unsigned int phase,
    unsigned int step,
    int sortOrder
);

// Host function for parallel bitonic sort
void bitonicSortParallel(
    uint32_t *d_values,
    unsigned int array_length,
    int sortOrder
);

#endif // BITONIC_SORT_GPU_CUH
