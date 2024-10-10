#include "bitonicSortCPU.h"
#include "TimerCPU.h"

#include <cstdint>
#include <algorithm> // for std::swap
#include <iostream>

using namespace std;

/* The parameter dir indicates the sorting direction, ASCENDING (0)
   or DESCENDING (1); if (values[i] > values[j]) agrees with the direction,
   then values[i] and values[j] are interchanged. */
void compAndSwap(uint32_t values[], unsigned int i, unsigned int j, int dir)
{
    // Sorting direction is controlled by dir (0 for ascending, 1 for descending)
    if (dir == (values[i] > values[j]))
    {
        swap(values[i], values[j]);
    }
}

/* It recursively sorts a bitonic sequence in ascending order
   if dir = 0, and in descending order if dir = 1.
   The sequence to be sorted starts at index position low.
   The parameter cnt is the number of elements to be sorted. */
void bitonicMerge(uint32_t values[], unsigned int low, unsigned int cnt, int dir)
{
    if (cnt > 1)
    {
        unsigned int k = cnt / 2;
        for (unsigned int i = low; i < low + k; i++)
        {
            compAndSwap(values, i, i + k, dir);
        }
        bitonicMerge(values, low, k, dir);
        bitonicMerge(values, low + k, k, dir);
    }
}

/* This function produces a bitonic sequence by recursively
   sorting its two halves in opposite sorting orders, and then
   calls bitonicMerge to make them in the same order */
void bitonicSort(uint32_t values[], unsigned int low, unsigned int cnt, int dir)
{
    if (cnt > 1)
    {
        unsigned int k = cnt / 2;

        // Sort first half in ascending order (0)
        bitonicSort(values, low, k, 0);

        // Sort second half in descending order (1)
        bitonicSort(values, low + k, k, 1);

        // Merge entire sequence in the order specified by dir
        bitonicMerge(values, low, cnt, dir);
    }
}

/* Caller function for bitonicSort.
   Sorts the entire array in ASCENDING order if sortOrder = 0,
   or in DESCENDING order if sortOrder = 1 */
void sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder)
{
    TimerCPU timer_cpu;
    timer_cpu.start();

    bitonicSort(values, 0, arrayLength, sortOrder);
    std::cout << "[CPU] - Sorting time: " << timer_cpu.getElapsedMilliseconds() << " ms" << std::endl;
}