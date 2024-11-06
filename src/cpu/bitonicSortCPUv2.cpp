#include "bitonicSortCPUv2.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <omp.h>

/**
 * SIMD-optimized bitonic sort implementation for small arrays
 * Uses OpenMP SIMD directives for vectorization
 *
 * @param values Pointer to array to be sorted
 * @param length Number of elements in the array
 */
void simdBitonicSort(uint32_t* values, unsigned int length) {
    // Pad array to nearest power of 2 which is required for bitonic sort
    unsigned int paddedLength = 1 << static_cast<int>(std::ceil(std::log2(length)));
    // Create temporary array with padding filled with max values
    std::vector<uint32_t> padded(paddedLength, std::numeric_limits<uint32_t>::max());
    // Copy input array to padded array
    std::copy(values, values + length, padded.begin());

    // Main bitonic sort loop
    // k represents the current subsequence size (doubles each iteration)
    for (int k = 2; k <= paddedLength; k *= 2) {
        // j represents the current comparison distance (halves each iteration)
        for (int j = k / 2; j > 0; j /= 2) {
            // Enable SIMD parallelization for the comparison loop
            #pragma omp parallel for simd
            for (int i = 0; i < paddedLength; i++) {
                // Calculate index to compare with using XOR
                int ixj = i ^ j;
                // Only perform comparison if target index is higher (prevents double swapping)
                if ((ixj) > i) {
                    // For ascending sequences (when i's k-th bit is 0)
                    if ((i & k) == 0 && padded[i] > padded[ixj]) {
                        std::swap(padded[i], padded[ixj]);
                    }
                    // For descending sequences (when i's k-th bit is 1)
                    if ((i & k) != 0 && padded[i] < padded[ixj]) {
                        std::swap(padded[i], padded[ixj]);
                    }
                }
            }
        }
    }

    // Copy back only the valid (non-padded) elements
    std::copy_n(padded.begin(), length, values);
}

/**
 * Parallel bucket sort implementation for large arrays
 * Uses the most significant byte for bucket distribution
 *
 * @param values Array to be sorted
 * @param length Number of elements
 * @param sortOrder 1 for ascending, 0 for descending
 */
void bucketSort(uint32_t* values, unsigned int length, int sortOrder) {
    // Initialize main buckets array - one bucket for each possible byte value (256 buckets)
    std::vector<std::vector<uint32_t>> buckets(NUM_BUCKETS);

    // First phase: Distribute elements into buckets
    #pragma omp parallel
    {
        // Create thread-local buckets to avoid synchronization overhead
        std::vector<std::vector<uint32_t>> localBuckets(NUM_BUCKETS);

        // Distribute elements to thread-local buckets
        // nowait allows threads to proceed without synchronization at loop end
        #pragma omp for nowait
        for (int i = 0; i < length; i++) {
            // Extract most significant byte (bits 24-31) for bucket index
            int bucket = (values[i] >> 24) & 0xFF;
            localBuckets[bucket].push_back(values[i]);
        }

        // Merge thread-local buckets into global buckets
        // Critical section prevents concurrent access to shared buckets
        #pragma omp critical
        {
            for (int i = 0; i < NUM_BUCKETS; i++) {
                buckets[i].insert(buckets[i].end(),
                                localBuckets[i].begin(),
                                localBuckets[i].end());
            }
        }
    }

    // Second phase: Sort individual buckets
    // Use dynamic scheduling for better load balancing as bucket sizes may vary
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (buckets[i].size() > 1) {
            // Sort each non-empty bucket using bitonic sort
            simdBitonicSort(buckets[i].data(), buckets[i].size());
        }
    }

    // Final phase: Collect sorted results
    unsigned int pos = 0;
    if (sortOrder == 1) { // Ascending order
        // Concatenate buckets in forward order
        for (const auto& bucket : buckets) {
            std::copy(bucket.begin(), bucket.end(), values + pos);
            pos += bucket.size();
        }
    } else { // Descending order
        // Concatenate buckets in reverse order and reverse elements within buckets
        for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
            std::copy(it->rbegin(), it->rend(), values + pos);
            pos += it->size();
        }
    }
}

/**
 * Main sorting function that selects between bucket sort and bitonic sort
 * based on array size
 *
 * @param values Array to be sorted
 * @param arrayLength Number of elements
 * @param sortOrder 1 for ascending, 0 for descending
 * @param numThreads Number of OpenMP threads to use
 * @return Elapsed time in milliseconds
 */
float sortCPUv2(uint32_t values[], unsigned int arrayLength, int sortOrder, unsigned int numThreads) {
    // Set number of OpenMP threads
    omp_set_num_threads(static_cast<int>(numThreads));

    // Start timing
    double start_time = omp_get_wtime();

    // Choose sorting algorithm based on array size
    if (arrayLength >= BUCKET_THRESHOLD) {
        // Use bucket sort for large arrays
        bucketSort(values, arrayLength, sortOrder);
    } else {
        // Use bitonic sort for small arrays
        simdBitonicSort(values, arrayLength);
        // Reverse if descending order is requested
        if (sortOrder == 0) {
            std::reverse(values, values + arrayLength);
        }
    }

    // Calculate elapsed time
    double end_time = omp_get_wtime();
    float elapsed_ms = (end_time - start_time) * 1000.0f;

    std::cout << "[CPU] - Sorting time: " << elapsed_ms << " ms" << std::endl;
    return elapsed_ms;
}