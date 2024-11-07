#include "utils.cuh"
#include "constants.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

/*
 * Returns the current working directory as a string.
 */
std::string getCurrentDirectory() {
	return fs::current_path().string();
}

/*
 * Generates the filename for the result file based on the array length.
 * The filename is formatted as "sorting_results_<log2(arrayLength)>.csv".
*/
std::string getResultFilename(unsigned int arrayLength, const std::string& resultFolder, unsigned int numThreads) {
	std::stringstream ss;

	// Check if arrayLength is a power of 2
	if ((arrayLength & (arrayLength - 1)) == 0) {
		// If arrayLength is a power of 2, display it as 2^log2Length
		int log2Length = static_cast<int>(std::log2(arrayLength));
		ss << resultFolder << "/array_length_2^" << log2Length;
	} else {
		// If not a power of 2, display the size in MB or MiB
		double sizeInMB = static_cast<double>(arrayLength * 4) / 1'000'000; // Use 1'048'576 for MiB if preferred
		ss << resultFolder << "/array_size_" << sizeInMB << "MB";
	}

	ss << "_threads_" << numThreads << ".csv";
	return ss.str();
}

/*
 * Ensures that the directory for the provided file path exists.
 * If it doesn't exist, it creates the necessary directories.
 */
void ensureDirectoryExists(const std::string& filePath) {
    fs::path dir = fs::path(filePath).parent_path();
    if (!exists(dir)) {
        create_directories(dir);
    }
}

/*
 * Writes the result of a sorting test to the specified file.
 * The result includes array length, iteration, GPU time, CPU time, and a correctness flag.
 * The results are appended to the file in CSV format.
 */
void writeResultToFile(const std::string& filename, unsigned int arrayLength, unsigned int iteration,
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

	// Write data to the file
	outFile << ss.str() << ","
			<< arrayLength << ","
			<< iteration << ","
			<< gpuTime << ","
			<< cpuTime << ","
			<< (isCorrect ? "true" : "false") << std::endl;

	outFile.close();
}

void initializeResultFile(const std::string& filename, unsigned int arrayLength, unsigned int testRepetitions,
                          int sortOrder, unsigned int gridSize, unsigned int numThreads, bool skipGPU) {
	ensureDirectoryExists(filename);
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Write metadata
	outFile << "Array Length: " << arrayLength << std::endl;
	outFile << "Test Repetitions: " << testRepetitions << std::endl;
	outFile << "Sort Order: " << (sortOrder == ORDER_ASC ? "Ascending\n" : "Descending\n") << std::endl;

	if (!skipGPU) {
		// Write grid, block, and thread information
		outFile << "Grid Size: " << gridSize << std::endl;
		outFile << "Block Size: " << THREADS_BITONIC_SORT << std::endl;
		outFile << "Threads (GPU): " << gridSize * THREADS_BITONIC_SORT << "\n\n";
	} else {
		outFile << "Threads (CPU): " << numThreads << "\n\n";
	}

	// Header for the results
	outFile << "Timestamp,Array Length,Iteration,GPU Time (ms),CPU Time (ms),Is Correct" << std::endl;

	outFile.close();
}

/*
 * Fills an array of size tableLen with random 32-bit unsigned integers using a uniform distribution.
 */
void fillArray(uint32_t* keys, unsigned int tableLen)
{
	// Use high-resolution clock to generate a seed for randomness
	const auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

	// Create a random number generator with the seed
	std::mt19937 generator(seed);

	// Define the distribution range (0 to UINT32_MAX)
	std::uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);

	// Fill the array with random numbers
	for (unsigned int i = 0; i < tableLen; ++i)
	{
		keys[i] = distribution(generator);
	}
}

/*
 * Sorts the array based on the sortOrder (ascending or descending) and verifies the result using std::sort.
 */
void sortVerification(uint32_t* dataTable, const unsigned int arrayLength, int sortOrder) {
	if (sortOrder == ORDER_ASC) {
		std::sort(dataTable, dataTable + arrayLength);  // Default is ascending
	} else {
		std::sort(dataTable, dataTable + arrayLength, std::greater());  // Use greater<> for descending
	}
}

/*
 * From the number of threads, elements per thread, and the array length, calculates the offset and length
 * of the data block that will be processed by the current thread block.
 */
__device__ void calcDataBlockLength(unsigned int &offset, unsigned int &dataBlockLength, unsigned int arrayLength,
	unsigned int numThreads, unsigned int elemsThread)
{
	unsigned int elemsPerThreadBlock = numThreads * elemsThread;
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength =  offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;
}

/*
 * Compares two elements and exchanges them based on the sorting order.
 * If sortOrder is ascending, swaps if elem1 > elem2. If descending, swaps if elem1 < elem2.
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
 * Checks if the given value is a power of two.
 * Returns true if the value is a power of two; otherwise, false.
 */
bool isPowerOfTwo(unsigned int value)
{
	return (value != 0) && ((value & (value - 1)) == 0);
}

/*
 * Returns the next power of two greater than or equal to the given value.
 * If the value is already a power of two, it returns the value itself.
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