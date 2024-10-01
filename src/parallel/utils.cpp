#include "utils.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <cuda_runtime.h>

bool createFolder(char *folderName)
{
	return CreateDirectory(folderName, NULL);
}

/*
Appends provided text to file.
*/
void appendToFile(const string &fileName, const string &text)
{
	std::ofstream file;
	file.open(fileName, std::fstream::app);

	file << text;
	file.close();
}

/*
Checks if there was an error.
*/
void checkMallocError(void *ptr)
{
	if (ptr == NULL)
	{
		std::cerr << "Error in host malloc.\n";
		exit(EXIT_FAILURE);
	}
}

cudaDeviceProp getCudaDeviceProp(uint_t deviceIndex)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceIndex);
	return deviceProp;
}

void fillArrayKeyOnly(uint32_t *keys, uint_t tableLen, uint_t interval, uint_t bucketSize)
{
	auto seed = chrono::high_resolution_clock::now().time_since_epoch().count() + generatorCalls++;
	auto generator = bind(uniform_int_distribution<uint32_t>(0, interval), mt19937(seed));

	for (uint_t i = 0; i < tableLen; i++)
	{
		keys[i] = generator();
	}

	break;
}

/*
Starts the stopwatch (remembers the current time).
*/
void startStopwatch(LARGE_INTEGER* start)
{
    QueryPerformanceCounter(start);
}

/*
Ends the stopwatch (calculates the difference between current time and parameter "start") and returns time
in milliseconds. Also prints out comment.
*/
double endStopwatch(LARGE_INTEGER start, char* comment)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER end;
    double elapsedTime;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&end);
    elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    if (comment != NULL)
    {
        printf("%s: %.5lf ms\n", comment, elapsedTime);
    }

    return elapsedTime;
}

/*
Sorts data with C++ qsort, which sorts data 100% correctly. This is needed to verify parallel and sequential
sorts.
*/
double sortCorrect(uint32_t *dataTable, uint_t tableLen, order_t sortOrder)
{
	LARGE_INTEGER timer;
	startStopwatch(&timer);

	quickSort<uint32_t>(dataTable, tableLen, sortOrder);

	return endStopwatch(timer);
}

/*
Sorts an array with C quicksort implementation.
*/
void quickSort(uint32_t *dataTable, uint_t tableLen, order_t sortOrder)
{
    if (sortOrder == ORDER_ASC)
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareAsc);
    }
    else
    {
        qsort(dataTable, tableLen, sizeof(*dataTable), compareDesc);
    }
}