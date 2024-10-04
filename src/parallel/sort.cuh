#ifndef SORT_CUH
#define SORT_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Declaration of the Sort class
class Sort {

public:
    // Constructor and Destructor
    Sort();
    Sort(uint32_t *d_values, uint32_t *h_values, unsigned int array_length, int sort_order);
    ~Sort();

    // Method for allocating memory
    void memoryAllocate(unsigned int arrayLength);

    void memoryCopyBeforeSort() const;
    void memoryCopyAfterSort() const;

    // Method for freeing memory (important for cleanup)
    void memoryFree() const;

    void sortValues();

    void sort(unsigned int arrayLength);

private:
    uint32_t *_d_values; // Private member variable to hold device values
    uint32_t *_h_values;
    unsigned int _array_length;
    int _sort_order;
};

#endif // SORT_CUH
