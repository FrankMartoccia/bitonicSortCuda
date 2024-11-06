#ifndef CONSTANTS_H
#define CONSTANTS_H

// Sorting order constants
// These constants represent the order in which the sorting algorithms will sort the data
constexpr int ORDER_ASC = 1;  // Ascending order
constexpr int ORDER_DESC = 0;  // Descending order

// Bitonic Sort thread and element configuration
// These constants define the number of threads and elements used for the Bitonic Sort algorithm
constexpr int BITONIC_SORT_THREADS = 32;  // Number of threads to use in the Bitonic Sort kernel
// constexpr int ELEMENTS_BITONIC_SORT = 4;    // Number of elements processed per thread in Bitonic Sort
constexpr int BITONIC_BLOCKS = 8192;

// Global Merge thread and element configuration
// These constants define the number of threads and elements for global merging during sorting
constexpr int MERGE_GLOBAL_THREADS = 64;   // Number of threads to use in the global merge kernel
// constexpr int ELEMENTS_GLOBAL_MERGE = 4;     // Number of elements processed per thread in global merge
constexpr int MERGE_GLOBAL_BLOCKS = 4096;

// Local Merge thread and element configuration
// These constants define the number of threads and elements for local merging during sorting
constexpr int MERGE_LOCAL_THREADS = 64;     // Number of threads to use in the local merge kernel
// constexpr int ELEMENTS_LOCAL_MERGE = 8;       // Number of elements processed per thread in local merge
constexpr int MERGE_LOCAL_BLOCKS = 4096;

#endif // CONSTANTS_H
