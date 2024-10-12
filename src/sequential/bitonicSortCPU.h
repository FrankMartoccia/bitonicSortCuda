#ifndef BITONICSORTCPU_H
#define BITONICSORTCPU_H

#include <cstdint>

// Function that compares and swaps elements based on the direction
void compAndSwap(uint32_t values[], unsigned int i, unsigned int j, int dir);

// Function that merges a bitonic sequence
void bitonicMerge(uint32_t values[], unsigned int low, unsigned int cnt, int dir);

// Recursive function that sorts a bitonic sequence
void bitonicSort(uint32_t values[], unsigned int low, unsigned int cnt, int dir);

// Function that sorts an entire array using Bitonic Sort
// `sortOrder = 0` for ascending, `sortOrder = 1` for descending
float sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder);


#endif //BITONICSORTCPU_H
