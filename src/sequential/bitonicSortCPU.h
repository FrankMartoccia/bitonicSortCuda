#ifndef BITONICSORTCPU_H
#define BITONICSORTCPU_H

#include <cstdint>
#include <vector>

// Compares and swaps elements in a chunk of a bitonic sequence.
// "paddedValues" Reference to the vector containing the padded input array
// "threadId" ID of the current thread, used to determine which chunk to process
// "chunkSize" Size of the chunk each thread is responsible for
// "mergeStep" Current step in the merging process, determines which elements to compare
// "bitonicSequenceSize" Size of the current bitonic sequence being processed
void compareAndSwap(std::vector<uint32_t>& paddedValues, unsigned int threadId,
                    unsigned int chunkSize, unsigned int mergeStep, unsigned int bitonicSequenceSize);

// Performs a parallel bitonic sort on the input array.
// "values" Pointer to the array to be sorted
// "arrayLength" Number of elements in the input array
// "numThreads" Number of threads to use for parallel processing
// "sortOrder" Determines the final sort order (0 for descending, non-zero for ascending)
void bitonicSort(uint32_t values[], unsigned int arrayLength, unsigned int numThreads, int sortOrder);

// Function that sorts an entire array using the Bitonic Sort algorithm.
// "values" Pointer to the array to be sorted
// "arrayLength" Number of elements in the input array
// "sortOrder" Determines the final sort order (0 for descending, non-zero for ascending)
// Returns the time taken to sort the array in milliseconds.
float sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder);

#endif // BITONICSORTCPU_H
