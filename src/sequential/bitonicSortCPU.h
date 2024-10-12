#ifndef BITONICSORTCPU_H
#define BITONICSORTCPU_H

#include <cstdint>

// Function that compares and swaps two elements if they are not in the correct order
// according to the sorting direction (dir). `dir = 1` means ascending, `dir = 0` means descending.
void compAndSwap(uint32_t values[], unsigned int i, unsigned int j, int dir);

// Function that merges two halves of a bitonic sequence into one sequence
// sorted according to the specified direction (dir).
void bitonicMerge(uint32_t values[], unsigned int low, unsigned int cnt, int dir);

// Recursive function that divides an array into two halves, sorts each half
// in opposite directions, and then merges them into a bitonic sequence.
void bitonicSort(uint32_t values[], unsigned int low, unsigned int cnt, int dir);

// Function that sorts an entire array using the Bitonic Sort algorithm.
// The parameter `sortOrder = 1` for ascending order and `sortOrder = 0` for descending order.
// Returns the time taken to sort the array in milliseconds.
float sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder);

#endif // BITONICSORTCPU_H
