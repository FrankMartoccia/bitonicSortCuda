#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cuda_runtime.h> 
#include "data_types_common.h"  


bool createFolder(char* folderName);
void appendToFile(const string& fileName, const string& text);
void checkMallocError(void *ptr);
cudaDeviceProp getCudaDeviceProp(uint_t deviceIndex);
double sortCorrect(uint32_t *dataTable, uint_t tableLen, order_t sortOrder);
void startStopwatch(LARGE_INTEGER* start);
double endStopwatch(LARGE_INTEGER start, char* comment);
void quickSort(uint32_t *dataTable, uint_t tableLen, order_t sortOrder);
bool compareArrays(uint32_t* array1, uint32_t* array2, uint_t arrayLen)

#endif  // UTILS_H