#ifndef BITONICSORTCPUV2_H
#define BITONICSORTCPUV2_H

#include <cstdint>

constexpr unsigned int NUM_BUCKETS = 256; // Use 256 buckets for uint32_t
// Threshold for switching between bucket sort and bitonic sort
constexpr unsigned int BUCKET_THRESHOLD = 1024 * 1024; // 1M elements
constexpr unsigned int BITONIC_BLOCK_SIZE = 1024; // Size for bitonic sort blocks

// SIMD-optimized bitonic sort for small arrays.
// "values" Pointer to the array to be sorted
// "length" Number of elements in the array
void simdBitonicSort(uint32_t* values, unsigned int length);

// Bucket sort implementation for large arrays.
// "values" Pointer to the array to be sorted
// "length" Number of elements in the array
// "sortOrder" Determines the final sort order (1 for ascending, 0 for descending)
void bucketSort(uint32_t* values, unsigned int length, int sortOrder);

// Main sorting function that chooses between bucket sort and bitonic sort based on array size.
// "values" Pointer to the array to be sorted
// "arrayLength" Number of elements in the array
// "sortOrder" Determines the final sort order (1 for ascending, 0 for descending)
// "numThreads" Number of threads to use for parallel processing
// Returns the time taken to sort the array in milliseconds.
float sortCPUv2(uint32_t values[], unsigned int arrayLength, int sortOrder, unsigned int numThreads);

#endif // BITONICSORTCPUV2_H
