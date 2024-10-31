cmake_minimum_required(VERSION 3.29)

# Specify the project name and supported languages
project(bitonicSortCuda LANGUAGES CUDA CXX)

# Set the CUDA standard to C++20
set(CMAKE_CUDA_STANDARD 20)

# Add the executable and specify the source files
add_executable(parallel
        src/gpu/utils.cuh
        src/gpu/utils.cu
        src/gpu/bitonicSortGPU.cu
        src/gpu/Sort.cu
        src/gpu/main.cu
        src/gpu/TimerGPU.cu
        src/gpu/TimerGPU.cuh
        src/gpu/constants.h
        src/cpu/TimerCPU.cpp
        src/cpu/TimerCPU.h
        src/gpu/Sort.cuh
        src/gpu/bitonicSortGPU.cuh
        src/cpu/bitonicSortCPUv1.cpp
        src/cpu/bitonicSortCPUv1.h
        src/cpu/bitonicSortCPUv2.cpp
        src/cpu/bitonicSortCPUv2.h
)

# Ensure that the target supports CUDA separable compilation
set_target_properties(parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link the CUDA runtime and other necessary CUDA libraries
target_link_libraries(parallel cuda)

# Specify the source file properties for CUDA files if necessary
set_source_files_properties(src/gpu/utils.cu src/gpu/TimerGPU.cuh PROPERTIES LANGUAGE CUDA)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()