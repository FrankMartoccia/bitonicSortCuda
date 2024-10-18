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
void run(unsigned int arrayLength, unsigned int testRepetitions, int sortOrder, unsigned int numThreads, bool skipGPU, const std::string& resultFolder) {
    // Print the current working directory
    std::cout << "Current working directory: " << getCurrentDirectory() << std::endl;

    // Get the result filename based on array length
    const std::string resultFilename = getResultFilename(arrayLength, resultFolder, numThreads);
    std::cout << "Results will be saved in: " << resultFilename << std::endl;

    // Compute block and grid sizes
    unsigned int elemsPerThreadBlock = THREADS_BITONIC_SORT * ELEMENTS_BITONIC_SORT;
    unsigned int gridSize = (arrayLength + elemsPerThreadBlock - 1) / elemsPerThreadBlock;

    // Initialize result file with grid, block size, and thread info
    initializeResultFile(resultFilename, arrayLength, testRepetitions, sortOrder, gridSize, numThreads, skipGPU);

    // Allocate memory for the input array and a copy for CPU sorting
    uint32_t *values = new uint32_t[arrayLength];        // Array to hold values for GPU sorting
    uint32_t *valuesCopy = new uint32_t[arrayLength];   // Copy of the array for CPU sorting

    // Initialize the GPU Sort instance with the values array (only if GPU sorting is not skipped)
    Sort sortInstance = skipGPU ? Sort() : Sort(nullptr, values, arrayLength, sortOrder);

    // Loop for the specified number of test repetitions
    for (unsigned int iter = 0; iter < testRepetitions; iter++) {
        std::cout << "Iteration " << iter << std::endl;

        // Fill the values array with random data
        fillArray(values, arrayLength);

        // Debug print: Print the original (unsorted) array
        // printArray(values, arrayLength, "Original Array");

        // Copy the values array to the valuesCopy for CPU sorting
        std::copy_n(values, arrayLength, valuesCopy);

        // Perform sorting on the GPU (if not skipped)
        float gpuTime = 0.0;
        if (!skipGPU) {
            gpuTime = sortInstance.sortGPU();
            // Debug print: Print the GPU-sorted array
            // printArray(values, arrayLength, "GPU Sorted Array");
        }

        // Perform sorting on the CPU using Bitonic Sort
        float cpuTime = sortCPU(valuesCopy, arrayLength, sortOrder, numThreads);

        // Debug print: Print the CPU-sorted array
        // printArray(valuesCopy, arrayLength, "CPU Sorted Array");

        bool isCorrect = true;  // Initialize with a default value (true or false, based on your preference)

        if (!skipGPU) {
            // Verify the correctness of the sorting by comparing the two arrays
            isCorrect = std::equal(values, values + arrayLength, valuesCopy);
            std::cout << "Is correct: " << (isCorrect ? "true" : "false") << std::endl;
            std::cout << std::endl;  // Print a blank line for readability
        }

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
    if (argc < 3 || argc > 7)
    {
        printf(
            "2 mandatory and 4 optional arguments have to be specified:\n\n"
            "1. Array length\n"
            "2. Number of test repetitions\n"
            "3. Sort order (1 - ASC, 0 - DESC), optional, default is ASC\n"
            "4. Number of threads (CPU), optional, default is 1\n"
            "5. Skip GPU (1 to skip, 0 to run GPU sort), optional, default is 0\n"
            "6. Result folder, optional, default is './results'\n"
        );
        exit(EXIT_FAILURE);  // Exit if the arguments are invalid
    }

    // Parse command line arguments
    unsigned int arrayLength = atoi(argv[1]);  // Convert first argument to array length
    unsigned int testRepetitions = atoi(argv[2]);  // Convert second argument to number of repetitions

    // Parse optional arguments with defaults
    int sortOrder = (argc >= 4) ? atoi(argv[3]) : ORDER_ASC;       // Default to ASC if not provided
    unsigned int numThreads = (argc >= 5) ? atoi(argv[4]) : 1; // Default to 1 thread if not provided
    bool skipGPU = (argc >= 6) ? atoi(argv[5]) : false;  // 1 to skip GPU sort, 0 to run GPU sort
    std::string resultFolder = (argc == 7) ? argv[6] : "../results";

    // Execute the sorting tests with the provided parameters
    run(arrayLength, testRepetitions, sortOrder, numThreads, skipGPU, resultFolder);
}
