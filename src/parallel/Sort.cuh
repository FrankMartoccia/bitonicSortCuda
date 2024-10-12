#ifndef SORT_CUH
#define SORT_CUH

#include <cstdint>

// Declaration of the Sort class for performing sorting operations on GPU.
class Sort {

public:
    // Default Constructor: Initializes member variables to default values.
    Sort();

    // Parameterized Constructor: Initializes member variables with provided values.
    Sort(uint32_t *d_values, uint32_t *h_values, unsigned int array_length, int sort_order);

    // Method for allocating memory on the device for sorting
    void memoryAllocate();

    // Copies data from host to device before sorting
    void memoryCopyBeforeSort() const;

    // Copies sorted data from device to host after sorting
    void memoryCopyAfterSort() const;

    // Frees allocated memory on the device (important for cleanup)
    void memoryFree() const;

    // Calls the bitonic sort function based on the sorting order
    void sortValues();

    // Executes the full sorting process, including memory management and timing
    float sortGPU();

private:
    uint32_t *_d_values; // Pointer to hold device values (allocated on GPU)
    uint32_t *_h_values; // Pointer to hold host values (allocated on CPU)
    unsigned int _array_length; // Length of the array to sort
    int _sort_order; // Sort order: ascending or descending
};

#endif // SORT_CUH
