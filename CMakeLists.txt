cmake_minimum_required(VERSION 3.29)

# Specify the project name and supported languages
project(bitonicSortCuda LANGUAGES CUDA CXX)

# Set the CUDA standard to C++20
set(CMAKE_CUDA_STANDARD 20)

# Add the executable and specify the source files
add_executable(parallel
        src/parallel/utils.h
        src/parallel/utils.cpp
        src/parallel/bitonicSortGPU.cu
        src/parallel/sort.cu
        src/parallel/main.cpp
        src/parallel/TimerGPU.cu
        src/parallel/TimerGPU.cuh
        src/parallel/constants.h
        src/sequential/TimerCPU.cpp
        src/sequential/TimerCPU.h
)

# Ensure that the target supports CUDA separable compilation
set_target_properties(parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link the CUDA runtime and other necessary CUDA libraries
target_link_libraries(parallel cuda)

# Specify the source file properties for CUDA files if necessary
set_source_files_properties(src/parallel/utils.cpp src/parallel/TimerGPU.cuh PROPERTIES LANGUAGE CUDA)