#include "utils.cuh"
#include "constants.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>

bool createFolder(const std::string &folderName)
{
	return std::filesystem::create_directory(folderName);
}

/*
Appends provided text to file.
*/
void appendToFile(const std::string &fileName, const std::string &text)
{
	std::ofstream file;
	file.open(fileName, std::fstream::app);

	file << text;
	file.close();
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
		std::sort(dataTable, dataTable + tableLen, std::greater<uint32_t>());  // Use greater<> for descending
	}
}


// /*
// Sorts an array with C quicksort implementation.
// */
// void quickSort(uint32_t *dataTable, const unsigned int tableLen, int sortOrder)
// {
//     if (sortOrder == ORDER_ASC)
//     {
//         qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
//     }
//     else
//     {
//         qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
//     }
// }
//
// /*
// Compare function for ASCENDING order needed for C++ qsort.
// */
// int compareAsc(const void* elem1, const void* elem2)
// {
// 	// Cannot use subtraction because of unsigned data types. Another option would be to convert to bigger data
// 	// type, but the result has to be converted to int.
// 	if (*(uint32_t*)elem1 > *(uint32_t*)elem2)
// 	{
// 		return 1;
// 	}
// 	if (*(uint32_t*)elem1 < *(uint32_t*)elem2)
// 	{
// 		return -1;
// 	}
//
// 	return 0;
// }
//
// /*
// Compare function for DESCENDING order needed for C++ qsort.
// */
// int compareDesc(const void* elem1, const void* elem2)
// {
// 	// Cannot use subtraction because of unsigned data types. Another option would be to convert to bigger data
// 	// type, but the result has to be converted to int.
// 	if (*(uint32_t*)elem1 < *(uint32_t*)elem2)
// 	{
// 		return 1;
// 	}
// 	if (*(uint32_t*)elem1 > *(uint32_t*)elem2)
// 	{
// 		return -1;
// 	}
//
// 	return 0;
// }

/*
Compares two arrays and prints out if they are the same or if they differ.
*/
// bool compareArrays(const uint32_t* array1, const uint32_t* array2, const unsigned int arrayLen)
// {
// 	for (unsigned int i = 0; i < arrayLen; i++)
// 	{
// 		if (array1[i] != array2[i])
// 		{
// 			return false;
// 		}
// 	}
//
// 	return true;
// }

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