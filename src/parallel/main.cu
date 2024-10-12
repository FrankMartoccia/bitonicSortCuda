#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>

#include "constants.h"
#include "Sort.cuh"
#include "utils.cuh"
#include "../sequential/bitonicSortCPU.h"

// Function to print array values
void printArray(const uint32_t* array, unsigned int length, const std::string& arrayName) {
    std::cout << arrayName << ": ";
    for (unsigned int i = 0; i < length; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void run(unsigned int arrayLength, unsigned int testRepetitions, int sortOrder) {
    // Print current working directory
    std::cout << "Current working directory: " << getCurrentDirectory() << std::endl;

    // Get the result filename
    const std::string resultFilename = getResultFilename(arrayLength);

    std::cout << "Results will be saved in: " << resultFilename << std::endl;

    // Initialize result file
    initializeResultFile(resultFilename, arrayLength, testRepetitions, sortOrder);

    // Allocate memory for arrays
    uint32_t *values = new uint32_t[arrayLength];
    uint32_t *valuesCopy = new uint32_t[arrayLength];

    // Initialize the GPU Sort instance
    Sort sortInstance = Sort(nullptr, values, arrayLength, sortOrder);

    for (unsigned int iter = 0; iter < testRepetitions; iter++) {
        std::cout << "Iteration " << iter << std::endl;

        // Fill values array with random data
        fillArray(values, arrayLength);

        // Copy values array to valuesCopy for CPU sorting
        std::copy_n(values, arrayLength, valuesCopy);

        // Perform GPU sorting
        float gpuTime = sortInstance.sortGPU();

        // Perform CPU sorting using Bitonic Sort
        float cpuTime = sortCPU(valuesCopy, arrayLength, sortOrder);

        // Verify the correctness of the sorting
        bool isCorrect = std::equal(values, values + arrayLength, valuesCopy);

        std::cout << "Is correct: " << (isCorrect ? "true" : "false") << std::endl;
        std::cout << std::endl;

        // Write results to file
        writeResultToFile(resultFilename, arrayLength, iter, gpuTime, cpuTime, isCorrect);
    }

    // Free allocated memory
    delete[] values;
    delete[] valuesCopy;
}


int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4)
    {
        printf(
            "Two mandatory and one optional argument has to be specified:\n"
            "1. Array length\n"
            "2. Number of test repetitions\n"
            "3. Sort order (0 - ASC, 1 - DESC), optional, default ASC\n"
        );
        exit(EXIT_FAILURE);
    }

    unsigned int arrayLength = atoi(argv[1]);
    // How many times is the sorting algorithm test repeated
    unsigned int testRepetitions = atoi(argv[2]);
    // Sort order of the data
    int sortOrder = argc == 3 ? ORDER_ASC : atoi(argv[3]);

    run(arrayLength, testRepetitions, sortOrder);
}
