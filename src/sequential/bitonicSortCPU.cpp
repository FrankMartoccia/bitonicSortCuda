#include "bitonicSortCPU.h"
#include "TimerCPU.h"
#include <cstdint>
#include <algorithm>
#include <iostream>

/*
    Compares and swaps the elements at positions i and j in the array
    based on the sorting direction (dir).
    If dir = 1 (ascending) and values[i] > values[j], or
    if dir = 0 (descending) and values[i] < values[j],
    the elements are swapped.
*/
void compAndSwap(uint32_t values[], unsigned int i, unsigned int j, int dir)
{
    // Swap elements if they are in the wrong order based on dir
    if (dir == (values[i] > values[j]))
    {
        std::swap(values[i], values[j]);
    }
}

/*
    Recursively merges a bitonic sequence into a sorted sequence.
    The bitonic sequence is divided into two halves, which are merged
    in the specified sorting direction (dir).

    Parameters:
    - low: starting index of the sequence
    - cnt: number of elements to merge
    - dir: sorting direction (1 for ascending, 0 for descending)
*/
void bitonicMerge(uint32_t values[], unsigned int low, unsigned int cnt, int dir)
{
    if (cnt > 1)
    {
        unsigned int k = cnt / 2; // Split sequence into two halves
        for (unsigned int i = low; i < low + k; i++)
        {
            compAndSwap(values, i, i + k, dir); // Compare and swap elements between two halves
        }
        // Recursively merge both halves
        bitonicMerge(values, low, k, dir);        // Merge the first half
        bitonicMerge(values, low + k, k, dir);    // Merge the second half
    }
}

/*
    Recursively sorts an array into a bitonic sequence.
    The array is split into two halves: the first half is sorted in descending order
    (dir = 0), and the second half is sorted in ascending order (dir = 1).
    After sorting the two halves, the bitonicMerge function is called to combine
    them into a single sorted sequence in the direction specified by dir.

    Parameters:
    - low: starting index of the sequence
    - cnt: number of elements to sort
    - dir: sorting direction (1 for ascending, 0 for descending)
*/
void bitonicSort(uint32_t values[], unsigned int low, unsigned int cnt, int dir)
{
    if (cnt > 1)
    {
        unsigned int k = cnt / 2;

        // Recursively sort first half in descending order (0)
        bitonicSort(values, low, k, 0);

        // Recursively sort second half in ascending order (1)
        bitonicSort(values, low + k, k, 1);

        // Merge the two halves into a bitonic sequence sorted in the specified direction
        bitonicMerge(values, low, cnt, dir);
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
float sortCPU(uint32_t values[], unsigned int arrayLength, int sortOrder)
{
    TimerCPU timer_cpu;       // Timer to measure the sorting duration
    timer_cpu.start();        // Start the timer

    bitonicSort(values, 0, arrayLength, sortOrder); // Sort the array using Bitonic Sort

    float time = timer_cpu.getElapsedMilliseconds(); // Get the elapsed time
    std::cout << "[CPU] - Sorting time: " << time << " ms" << std::endl; // Output the sorting time

    return time;  // Return the time taken for sorting
}
