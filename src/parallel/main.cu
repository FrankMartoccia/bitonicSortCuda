#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>

#include "constants.h"  // Constants used for sorting orders
#include "Sort.cuh"     // Include the Sort class for GPU sorting
#include "utils.cuh"    // Include utility functions for file handling, directory management, etc.
#include "../sequential/bitonicSortCPU.h"  // Include CPU implementation of Bitonic Sort

// Function to print array values to the console
void printArray(const uint32_t* array, unsigned int length, const std::string& arrayName) {
    std::cout << arrayName << ": ";
    for (unsigned int i = 0; i < length; i++) {
        std::cout << array[i] << " ";  // Print each element in the array
    }
    std::cout << std::endl;  // End the line after printing the array
}

// Function to run sorting tests
void run(unsigned int arrayLength, unsigned int testRepetitions, int sortOrder) {
    // Print the current working directory
    std::cout << "Current working directory: " << getCurrentDirectory() << std::endl;

    // Get the result filename based on array length
    const std::string resultFilename = getResultFilename(arrayLength);
    std::cout << "Results will be saved in: " << resultFilename << std::endl;

    // Compute block and grid sizes
    unsigned int elemsPerThreadBlock = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;
    unsigned int gridSize = (arrayLength + elemsPerThreadBlock - 1) / elemsPerThreadBlock;

    // Initialize result file with grid, block size, and thread info
    initializeResultFile(resultFilename, arrayLength, testRepetitions, sortOrder, gridSize, THREADS_BITONIC_SORT);

    // Allocate memory for the input array and a copy for CPU sorting
    uint32_t *values = new uint32_t[arrayLength];        // Array to hold values for GPU sorting
    uint32_t *valuesCopy = new uint32_t[arrayLength];   // Copy of the array for CPU sorting

    // Initialize the GPU Sort instance with the values array
    Sort sortInstance = Sort(nullptr, values, arrayLength, sortOrder);

    // Loop for the specified number of test repetitions
    for (unsigned int iter = 0; iter < testRepetitions; iter++) {
        std::cout << "Iteration " << iter << std::endl;

        // Fill the values array with random data
        fillArray(values, arrayLength);

        // Debug print: Print the original (unsorted) array
        // printArray(values, arrayLength, "Original Array");

        // Copy the values array to the valuesCopy for CPU sorting
        std::copy_n(values, arrayLength, valuesCopy);

        // Perform sorting on the GPU
        float gpuTime = sortInstance.sortGPU();

        // Debug print: Print the GPU-sorted array
        // printArray(values, arrayLength, "GPU Sorted Array");

        // Perform sorting on the CPU using Bitonic Sort
        float cpuTime = sortCPU(valuesCopy, arrayLength, sortOrder);

        // Debug print: Print the CPU-sorted array
        // printArray(valuesCopy, arrayLength, "CPU Sorted Array");

        // Verify the correctness of the sorting by comparing the two arrays
        bool isCorrect = std::equal(values, values + arrayLength, valuesCopy);
        std::cout << "Is correct: " << (isCorrect ? "true" : "false") << std::endl;
        std::cout << std::endl;  // Print a blank line for readability

        // Write the results of the current iteration to the result file
        writeResultToFile(resultFilename, arrayLength, iter, gpuTime, cpuTime, isCorrect);
    }

    // Free the allocated memory for both arrays
    delete[] values;
    delete[] valuesCopy;
}

// Main function: Entry point of the program
int main(int argc, char* argv[])
{
    // Check if the number of arguments is correct
    if (argc < 3 || argc > 4)
    {
        printf(
            "Two mandatory and one optional argument has to be specified:\n"
            "1. Array length\n"
            "2. Number of test repetitions\n"
            "3. Sort order (1 - ASC, 0 - DESC), optional, default is ASC\n"
        );
        exit(EXIT_FAILURE);  // Exit if the arguments are invalid
    }

    // Parse command line arguments
    unsigned int arrayLength = atoi(argv[1]);  // Convert first argument to array length
    unsigned int testRepetitions = atoi(argv[2]);  // Convert second argument to number of repetitions
    int sortOrder = argc == 3 ? ORDER_ASC : atoi(argv[3]);  // Set sort order based on arguments

    // Execute the sorting tests with the provided parameters
    run(arrayLength, testRepetitions, sortOrder);
}
