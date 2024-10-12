#include "Sort.cuh"
#include "constants.h"
#include <iostream>

#include "TimerGPU.cuh"
#include "bitonicSortGPU.cuh"
#include "cuda_runtime.h"

// Error checking function (assuming it is defined somewhere in your project)
void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Constructor: Initialize variables as needed
Sort::Sort(): _d_values(nullptr), _h_values(nullptr), _array_length(0), _sort_order(ORDER_ASC) {
}

// Parameterized Constructor: Initializes member variables
Sort::Sort(uint32_t *d_values, uint32_t *h_values, unsigned int array_length, int sort_order)
    : _d_values(d_values),
      _h_values(h_values),
      _array_length(array_length),
      _sort_order(sort_order){
}

// Method for allocating memory
void Sort::memoryAllocate() {
    // Allocates memory for values on the device
    const cudaError_t error = cudaMalloc((void **)&_d_values, _array_length * sizeof(*_d_values));
    checkCudaError(error);
}

/*
Memory copy operations needed before sort. If sorting keys only, than "h_values" contains NULL.
*/
void Sort::memoryCopyBeforeSort() const {
    // Copies values
    cudaError_t error = cudaMemcpy(
        _d_values, _h_values, _array_length * sizeof(*_d_values), cudaMemcpyHostToDevice
    );
    checkCudaError(error);
}

void Sort::memoryCopyAfterSort() const {
    // Copies values
    cudaError_t error = cudaMemcpy(
        _h_values, _d_values, _array_length * sizeof(*_h_values), cudaMemcpyDeviceToHost
    );
    checkCudaError(error);
}

// Method for freeing allocated memory
void Sort::memoryFree() const {
    {
        if (_array_length == 0)
        {
            return;
        }

        // Destroy values
        const cudaError_t error = cudaFree(_d_values);
        checkCudaError(error);
    }
}

/*
Wrapper for bitonic sort method.
The code runs faster if arguments are passed to method. If members are accessed directly, code runs slower.
*/
void Sort::sortValues()
{

    if (_sort_order == ORDER_ASC)
    {
        bitonicSortParallel(_d_values, _array_length, ORDER_ASC);
    }
    else
    {
        bitonicSortParallel(_d_values, _array_length, ORDER_DESC);
    }
}

/*
Wrapper method, which executes all needed memory management and timing. Also calls private sort.
*** Call the constructor first ***
*/
float Sort::sortGPU()
{
    memoryAllocate();

    memoryCopyBeforeSort();

    cudaError_t error = cudaDeviceSynchronize();
    checkCudaError(error);

    TimerGPU timer_gpu;
    timer_gpu.start();
    sortValues();

    error = cudaDeviceSynchronize();
    checkCudaError(error);

    timer_gpu.stop();
    const float time = timer_gpu.getElapsedMilliseconds();
    std::cout << "[GPU] - Sorting time: " << time  << " ms" << std::endl;

    memoryCopyAfterSort();

    memoryFree();

    return time;
}
