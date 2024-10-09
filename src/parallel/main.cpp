#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "constants.h"
#include "sort.cuh"
#include "utils.h"

void run(unsigned int arrayLength, unsigned int testRepetitions, int sortOrder) {

    // createFolder();
    // appendToFile();
    uint32_t *values = new uint32_t[arrayLength * sizeof(uint32_t)];
    uint32_t *valuesCopy = new uint32_t[arrayLength * sizeof(uint32_t)];

    for (unsigned int iter = 0; iter < testRepetitions; iter++)
    {
        fillArray(values, arrayLength);
        std::copy(values, values + arrayLength, valuesCopy);
        Sort sortInstance = Sort(nullptr, values, arrayLength, sortOrder);
        sortInstance.sort();
    }

}

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 4)
    {
        printf(
            "Two mandatory and one optional argument has to be specified:\n1. array length\n2. number of test "
            "repetitions\n3. sort order (0 - ASC, 1 - DESC), optional, default ASC\n"
        );
        exit(EXIT_FAILURE);
    }

    unsigned int arrayLength = atoi(argv[1]);
    // How many times is the sorting algorithm test repeated
    unsigned int testRepetitions = atoi(argv[2]);
    // Sort order of the data
    int sortOrder = argc == 3 ? ORDER_ASC : (int)atoi(argv[3]);

    run(arrayLength, testRepetitions, sortOrder);
}
