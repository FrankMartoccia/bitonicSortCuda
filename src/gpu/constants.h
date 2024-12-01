#ifndef CONSTANTS_H
#define CONSTANTS_H

// Sorting order constants
// These constants represent the order in which the sorting algorithms will sort the data
constexpr int ORDER_ASC = 1;  // Ascending order
constexpr int ORDER_DESC = 0;  // Descending order

// Bitonic Sort thread and element configuration
// These constants define the number of threads and elements used for the Bitonic Sort algorithm
constexpr int BITONIC_SORT_THREADS = 384;  // Number of threads to use in the Bitonic Sort kernel
// constexpr int ELEMENTS_BITONIC_SORT = 4;    // Number of elements processed per thread in Bitonic Sort
constexpr int BITONIC_SORT_BLOCKS = 4096;

constexpr int MAX_SHARED_MEMORY_SIZE = 49152;

#endif // CONSTANTS_H
