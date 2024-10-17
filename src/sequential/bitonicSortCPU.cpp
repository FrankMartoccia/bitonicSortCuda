#include "bitonicSortCPU.h"
#include "TimerCPU.h"
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <limits>

// This function is designed to be run in parallel by multiple threads. Each thread
//  processes a chunk of the overall array, comparing and swapping elements to sort
//  the bitonic sequence according to the current merge step.
// The direction of the sort (ascending or descending) alternates within each bitonic sequence.
void compareAndSwap(std::vector<uint32_t>& paddedValues, unsigned int threadId,
                    unsigned int chunkSize, unsigned int mergeStep, unsigned int bitonicSequenceSize)
{
    unsigned int startIndex = threadId * chunkSize;
    unsigned int endIndex = (threadId + 1) * chunkSize;

    // Process the chunk assigned to this thread
    for (unsigned int currentIndex = startIndex; currentIndex < endIndex; currentIndex++)
    {
        // Find the element to compare with
        unsigned int compareIndex = currentIndex ^ mergeStep;

        // Only compare if the compareIndex is greater (to avoid duplicate swaps)
        if (compareIndex > currentIndex)
        {
            bool shouldSwap = false;

            // Determine if we should swap based on the current subarray's sorting direction
            if ((currentIndex & bitonicSequenceSize) == 0)  // First half of subarray (ascending)
            {
                shouldSwap = (paddedValues[currentIndex] > paddedValues[compareIndex]);
            }
            else  // Second half of subarray (descending)
            {
                shouldSwap = (paddedValues[currentIndex] < paddedValues[compareIndex]);
            }

            // Perform the swap if necessary
            if (shouldSwap)
            {
                std::swap(paddedValues[currentIndex], paddedValues[compareIndex]);
            }
        }
    }
}

// This function implements the bitonic sort algorithm using multiple threads for
// parallel processing. It sorts the input array in either ascending or descending order.
// The function follows these steps:
//  1. Pads the input array to the next power of 2 for efficient bitonic sort processing
//  2. Divides the padded array into chunks for parallel processing
//  3. Iteratively builds and merges bitonic sequences of increasing size
//  4. Uses multiple threads to compare and swap elements in parallel
//  5. Copies the sorted elements back to the original array
//  6. Reverses the array if descending order is requested
//
//  Note: The bitonic sort algorithm requires the array length to be a power of 2,
//  which is why padding is necessary for arrays of arbitrary length.
void bitonicSort(uint32_t values[], unsigned int arrayLength, unsigned int numThreads, int sortOrder)
{
    // Step 1: Pad the array to the next power of 2
    unsigned int paddedLength = 1 << static_cast<int>(std::ceil(std::log2(arrayLength)));
    std::vector paddedValues(paddedLength, std::numeric_limits<uint32_t>::max());
    std::copy(values, values + arrayLength, paddedValues.begin());

    // Step 2: Determine chunk size for each thread
    unsigned int chunkSize = paddedLength / numThreads;

    // Step 3: Iteratively build and merge bitonic sequences
    // Outer loop: controls the size of bitonic sequences
    for (unsigned int bitonicSequenceSize = 2; bitonicSequenceSize <= paddedLength; bitonicSequenceSize *= 2)
    {
        // Middle loop: controls the size of sub-sequences being merged
        for (unsigned int mergeStep = bitonicSequenceSize / 2; mergeStep > 0; mergeStep /= 2)
        {
            // Step 4: Use multiple threads to compare and swap elements in parallel
            std::vector<std::thread> threads;
            threads.reserve(numThreads);

            // Thread creation loop
            for (unsigned int threadId = 0; threadId < numThreads; threadId++)
            {
                threads.emplace_back(compareAndSwap,
                                     std::ref(paddedValues),
                                     threadId,
                                     chunkSize,
                                     mergeStep,
                                     bitonicSequenceSize);
            }

            // Wait for all threads to complete this stage
            for (auto& thread : threads)
            {
                thread.join();
            }
        }
    }

    // Step 5: Copy back the sorted values
    std::copy(paddedValues.begin(), paddedValues.begin() + arrayLength, values);

    // Step 6: If descending order is required, reverse the array
    if (sortOrder == 0)
    {
        std::reverse(values, values + arrayLength);
    }
}

/*
    This function serves as the entry point for the Bitonic Sort algorithm.
    It initializes the timer, sorts the entire array using the bitonicSort function,
    and prints the sorting time in milliseconds.

    Parameters:
    - values: array to sort
    - arrayLength: number of elements in the array
    - sortOrder: sorting direction (1 for ascending, 0 for descending)

    Returns the time taken to sort the array in milliseconds.
*/
float sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder, unsigned int numThreads)
{
    TimerCPU timer_cpu;       // Timer to measure the sorting duration
    timer_cpu.start();        // Start the timer

    bitonicSort(values, arrayLength, numThreads, sortOrder); // Sort the array using Bitonic Sort

    float time = timer_cpu.getElapsedMilliseconds(); // Get the elapsed time
    std::cout << "[CPU] - Sorting time: " << time << " ms" << std::endl; // Output the sorting time

    return time;  // Return the time taken for sorting
}
