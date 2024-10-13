#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cuda_runtime.h>

// Returns the current working directory as a string
std::string getCurrentDirectory();

// Generates a result filename based on the array length (in log2 format)
std::string getResultFilename(unsigned int arrayLength);

// Ensures that the directory for the given file path exists; creates it if not
void ensureDirectoryExists(const std::string& filePath);

// Appends the result of a sorting test to the specified file
// Includes array length, iteration, GPU time, CPU time, and if the result is correct
void writeResultToFile(const std::string& filename, unsigned int arrayLength, unsigned int iteration,
                       double gpuTime, float cpuTime, bool isCorrect);

// Initializes a result file with metadata like array length, test repetitions, and sort order
void initializeResultFile(const std::string& filename, unsigned int arrayLength, unsigned int testRepetitions,
                          int sortOrder, unsigned int gridSize, unsigned int blockSize);

// Checks if a pointer was successfully allocated; prints an error and exits if not
void checkMallocError(const void *ptr);

// Returns the properties of a CUDA device specified by its index
cudaDeviceProp getCudaDeviceProp(unsigned int deviceIndex);

// Fills an array with random 32-bit unsigned integers
void fillArray(uint32_t* keys, unsigned int tableLen);

// Sorts the array based on the provided sort order (ascending or descending) and verifies correctness
void sortVerification(uint32_t *dataTable, unsigned int tableLen, int sortOrder);

// Comparator for ascending order used in sort functions
int compareAsc(const void* elem1, const void* elem2);

// Comparator for descending order used in sort functions
int compareDesc(const void* elem1, const void* elem2);

// CUDA device function to calculate the offset and length of a data block handled by each thread block
// based on the number of threads and elements per thread
__device__ void calcDataBlockLength(unsigned int &offset, unsigned int &dataBlockLength, unsigned int tableLen,
    unsigned int numThreads, unsigned int elemsThread);

// CUDA device function to compare and exchange two elements based on the sorting order (ascending or descending)
__device__ void compareExchange(uint32_t *elem1, uint32_t *elem2, int sortOrder);

// Returns true if the value is a power of two
bool isPowerOfTwo(unsigned int value);

// Returns the next power of two greater than or equal to the given value
unsigned int nextPowerOf2(unsigned int value);

#endif  // UTILS_H
