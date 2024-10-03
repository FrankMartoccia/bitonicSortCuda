#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <cuda_runtime.h>


bool createFolder(char* folderName);
void appendToFile(const std::string& fileName, const std::string& text);
void checkMallocError(const void *ptr);
cudaDeviceProp getCudaDeviceProp(unsigned int deviceIndex);
void quickSort(uint32_t *dataTable, unsigned int tableLen, int sortOrder);
int compareAsc(const void* elem1, const void* elem2);
int compareDesc(const void* elem1, const void* elem2);
bool compareArrays(uint32_t* array1, uint32_t* array2, unsigned int arrayLen);

#endif  // UTILS_H