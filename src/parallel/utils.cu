#include "utils.cuh"
#include "constants.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

std::string getCurrentDirectory() {
	return fs::current_path().string();
}

std::string getResultFilename(unsigned int arrayLength) {
    int log2Length = static_cast<int>(std::log2(arrayLength));
    std::stringstream ss;
    ss << "../results/sorting_results_" << log2Length << ".csv";
    return ss.str();
}

void ensureDirectoryExists(const std::string& filePath) {
    fs::path dir = fs::path(filePath).parent_path();
    if (!exists(dir)) {
        create_directories(dir);
    }
}

void writeResultToFile(const std::string& filename, unsigned int arrayLength, int iteration,
					   double gpuTime, float cpuTime, bool isCorrect) {
	ensureDirectoryExists(filename);
	std::ofstream outFile(filename, std::ios::app);  // Open file in append mode
	if (!outFile) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Get current timestamp
	auto now = std::chrono::system_clock::now();
	auto time = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

	outFile << ss.str() << ","
			<< arrayLength << ","
			<< iteration << ","
			<< gpuTime << ","
			<< cpuTime << ","
			<< (isCorrect ? "true" : "false") << std::endl;

	outFile.close();
}

void initializeResultFile(const std::string& filename, unsigned int arrayLength, unsigned int testRepetitions, int sortOrder) {
	ensureDirectoryExists(filename);
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	outFile << "Array Length: " << arrayLength << std::endl;
	outFile << "Test Repetitions: " << testRepetitions << std::endl;
	outFile << "Sort Order: " << (sortOrder == ORDER_ASC ? "Ascending\n" : "Descending\n") << std::endl;
	outFile << "Timestamp,Array Length,Iteration,GPU Time (ms),CPU Time (ms),Is Correct" << std::endl;

	outFile.close();
}

/*
Checks if there was an error.
*/
void checkMallocError(const void *ptr)
{
	if (ptr == nullptr)
	{
		std::cerr << "Error in host malloc.\n";
		exit(EXIT_FAILURE);
	}
}

cudaDeviceProp getCudaDeviceProp(unsigned int deviceIndex)
{
	cudaDeviceProp deviceProp{};
	cudaGetDeviceProperties(&deviceProp, deviceIndex);
	return deviceProp;
}

void fillArray(uint32_t* keys, unsigned int tableLen)
{
	// Use high-resolution clock to generate a seed for randomness
	const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

	// Create a random number generator with the seed
	std::mt19937 generator(seed);

	// Define the distribution range (0 to interval)
	std::uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);

	// Fill the array with random numbers within the specified interval
	for (unsigned int i = 0; i < tableLen; ++i)
	{
		keys[i] = distribution(generator);
	}
}

void sortVerification(uint32_t* dataTable, const unsigned int tableLen, int sortOrder) {
	if (sortOrder == ORDER_ASC) {
		std::sort(dataTable, dataTable + tableLen);  // Default is ascending
	} else {
		std::sort(dataTable, dataTable + tableLen, std::greater());  // Use greater<> for descending
	}
}

/*
From provided number of threads in thread block, number of elements processed by one thread and array length
calculates the offset and length of data block, which is processed by current thread block.
*/
__device__ void calcDataBlockLength(unsigned int &offset, unsigned int &dataBlockLength, unsigned int arrayLength,
	unsigned int numThreads, unsigned int elemsThread)
{
	unsigned int elemsPerThreadBlock = numThreads * elemsThread;
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength =  offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;
}

/*
Compares 2 elements and exchanges them according to sortOrder.
*/
__device__ void compareExchange(uint32_t *elem1, uint32_t *elem2, int sortOrder)
{
	if (sortOrder == ORDER_ASC ? (*elem1 > *elem2) : (*elem1 < *elem2))
	{
		uint32_t temp = *elem1;
		*elem1 = *elem2;
		*elem2 = temp;
	}
}

/*
Tests if number is power of 2.
*/
bool isPowerOfTwo(unsigned int value)
{
	return (value != 0) && ((value & (value - 1)) == 0);
}

/*
Return the next power of 2 for provided value. If value is already power of 2, it returns value.
*/
unsigned int nextPowerOf2(unsigned int value)
{
	if (isPowerOfTwo(value))
	{
		return value;
	}

	value--;
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	value++;

	return value;
}