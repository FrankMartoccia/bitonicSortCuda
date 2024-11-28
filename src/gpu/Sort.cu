#include "Sort.cuh"
#include "constants.h"
#include <iostream>
#include "TimerGPU.cuh"
#include "bitonicSortGPU.cuh"
#include "cuda_runtime.h"

// Error checking function for CUDA operations
void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Default Constructor: Initializes member variables to default values
Sort::Sort() : _d_values(nullptr), _h_values(nullptr), _array_length(0), _sort_order(ORDER_ASC) {
}

// Parameterized Constructor: Initializes member variables with provided values
Sort::Sort(uint32_t *d_values, uint32_t *h_values, unsigned int array_length, int sort_order)
    : _d_values(d_values),
      _h_values(h_values),
      _array_length(array_length),
      _sort_order(sort_order) {
}

// Method for allocating memory on the device for sorting
void Sort::memoryAllocate() {
    // Allocates memory for device values
    const cudaError_t error = cudaMalloc((void **)&_d_values, _array_length * sizeof(*_d_values));
    checkCudaError(error);
}

/*
 * Memory copy operations needed before sorting.
 * If sorting keys only, then "h_values" contains NULL.
 */
void Sort::memoryCopyBeforeSort() const {
    // Copies data from host (_h_values) to device (_d_values)
    cudaError_t error = cudaMemcpy(
        _d_values, _h_values, _array_length * sizeof(*_d_values), cudaMemcpyHostToDevice
    );
    checkCudaError(error);
}

// Copies sorted data from device (_d_values) to host (_h_values)
void Sort::memoryCopyAfterSort() const {
    // Copies data from device to host
    cudaError_t error = cudaMemcpy(
        _h_values, _d_values, _array_length * sizeof(*_h_values), cudaMemcpyDeviceToHost
    );
    checkCudaError(error);
}

// Method for freeing allocated memory on the device
void Sort::memoryFree() const {
    if (_array_length == 0) {
        return; // No memory to free if array length is 0
    }

    // Frees the allocated device memory
    const cudaError_t error = cudaFree(_d_values);
    checkCudaError(error);
}

/*
 * Wrapper for the bitonic sort method.
 * The code runs faster if arguments are passed to the method rather than accessing member variables directly.
 */
void Sort::sortValues()
{
    // Calls the bitonic sort function based on the specified sort order
    if (_sort_order == ORDER_ASC) {
        bitonicSortParallel(_d_values, _array_length, ORDER_ASC);
    } else {
        bitonicSortParallel(_d_values, _array_length, ORDER_DESC);
    }
}

/*
 * Wrapper method that executes all necessary memory management and timing.
 * Also calls the private sort function.
 *** Call the constructor first ***
 */
float Sort::sortGPU() {
    // Measure Pool Load Time (PLT)
    TimerGPU timer_plt;
    timer_plt.start();
    memoryAllocate(); // Allocate device memory for sorting
    memoryCopyBeforeSort(); // Copy data from host to device before sorting
    timer_plt.stop();
    float plt = timer_plt.getElapsedMilliseconds();
    std::cout << "[GPU] - Pool Load Time: " << plt << " ms" << std::endl;

    // Measure Pool Execution Time (PET)
    TimerGPU timer_pet;
    cudaError_t error = cudaDeviceSynchronize(); // Synchronize before starting
    checkCudaError(error);
    timer_pet.start();
    sortValues(); // Perform sorting on GPU
    error = cudaDeviceSynchronize(); // Synchronize to ensure sorting is complete
    timer_pet.stop();
    checkCudaError(error);
    float pet = timer_pet.getElapsedMilliseconds();
    std::cout << "[GPU] - Pool Execution Time: " << pet << " ms" << std::endl;

    // Measure Data Download Time (DDT) - Optional
    TimerGPU timer_ddt;
    timer_ddt.start();
    memoryCopyAfterSort(); // Copy sorted data from device to host
    timer_ddt.stop();
    float ddt = timer_ddt.getElapsedMilliseconds();
    std::cout << "[GPU] - Data Download Time: " << ddt << " ms" << std::endl;

    memoryFree(); // Free allocated device memory

    // Return total time or individual phase times as needed
    return pet;
}
