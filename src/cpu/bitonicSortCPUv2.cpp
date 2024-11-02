#include "bitonicSortCPUv2.h"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <omp.h>

// SIMD-optimized bitonic sort for small arrays
void simdBitonicSort(uint32_t* values, unsigned int length) {
    // Ensure length is power of 2 for bitonic sort
    unsigned int paddedLength = 1 << static_cast<int>(std::ceil(std::log2(length)));
    std::vector<uint32_t> padded(paddedLength, std::numeric_limits<uint32_t>::max());
    std::copy(values, values + length, padded.begin());

    for (unsigned int k = 2; k <= paddedLength; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            #pragma omp parallel for simd
            for (unsigned int i = 0; i < paddedLength; i++) {
                unsigned int ixj = i ^ j;
                if ((ixj) > i) {
                    if ((i & k) == 0 && padded[i] > padded[ixj]) {
                        std::swap(padded[i], padded[ixj]);
                    }
                    if ((i & k) != 0 && padded[i] < padded[ixj]) {
                        std::swap(padded[i], padded[ixj]);
                    }
                }
            }
        }
    }

    std::copy(padded.begin(), padded.begin() + length, values);
}

// Bucket sort implementation for large arrays
void bucketSort(uint32_t* values, unsigned int length, int sortOrder) {
    // Create buckets
    std::vector<std::vector<uint32_t>> buckets(NUM_BUCKETS);

    // First pass - distribute by most significant byte
    #pragma omp parallel
    {
        // Thread-local buckets to avoid contention
        std::vector<std::vector<uint32_t>> localBuckets(NUM_BUCKETS);

        #pragma omp for nowait
        for (unsigned int i = 0; i < length; i++) {
            unsigned int bucket = (values[i] >> 24) & 0xFF;
            localBuckets[bucket].push_back(values[i]);
        }

        // Merge thread-local buckets into global buckets
        #pragma omp critical
        {
            for (unsigned int i = 0; i < NUM_BUCKETS; i++) {
                buckets[i].insert(buckets[i].end(),
                                localBuckets[i].begin(),
                                localBuckets[i].end());
            }
        }
    }

    // Sort individual buckets in parallel using bitonic sort
    #pragma omp parallel for schedule(dynamic)
    for (unsigned int i = 0; i < NUM_BUCKETS; i++) {
        if (buckets[i].size() > 1) {
            simdBitonicSort(buckets[i].data(), buckets[i].size());
        }
    }

    // Collect results
    unsigned int pos = 0;
    if (sortOrder == 1) { // Ascending
        for (const auto& bucket : buckets) {
            std::copy(bucket.begin(), bucket.end(), values + pos);
            pos += bucket.size();
        }
    } else { // Descending
        for (auto it = buckets.rbegin(); it != buckets.rend(); ++it) {
            std::copy(it->rbegin(), it->rend(), values + pos);
            pos += it->size();
        }
    }
}

// Main sorting function that chooses between bucket sort and bitonic sort
float sortCPUv2(uint32_t values[], unsigned int arrayLength, int sortOrder, unsigned int numThreads) {
    omp_set_num_threads(static_cast<int>(numThreads));
    double start_time = omp_get_wtime();

    if (arrayLength >= BUCKET_THRESHOLD) {
        bucketSort(values, arrayLength, sortOrder);
    } else {
        simdBitonicSort(values, arrayLength);
        if (sortOrder == 0) { // Descending
            std::reverse(values, values + arrayLength);
        }
    }

    double end_time = omp_get_wtime();
    float elapsed_ms = (end_time - start_time) * 1000.0f;

    std::cout << "[CPU] - Sorting time: " << elapsed_ms << " ms" << std::endl;
    return elapsed_ms;
}