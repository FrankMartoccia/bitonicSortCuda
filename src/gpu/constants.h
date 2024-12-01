#ifndef CONSTANTS_H
#define CONSTANTS_H

// Sorting order constants
// These constants represent the order in which the sorting algorithms will sort the data
constexpr int ORDER_ASC = 1;  // Ascending order
constexpr int ORDER_DESC = 0;  // Descending order

// Bitonic Sort threads and blocks configuration
constexpr int BITONIC_SORT_THREADS = 384;
constexpr int BITONIC_SORT_BLOCKS = 4096;

// Max shared memory in bytes
constexpr int MAX_SHARED_MEMORY_SIZE = 49152;

#endif // CONSTANTS_H
