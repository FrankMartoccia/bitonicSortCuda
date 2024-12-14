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

# Ensure CUDA separable compilation
set_target_properties(parallel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Link CUDA runtime
target_link_libraries(parallel cuda)

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    # OpenMP flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# SIMD and Vectorization Optimization Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC Flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -fopenmp-simd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang Flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -fopenmp-simd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel Compiler Flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -qopenmp-simd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -vec")
elseif(MSVC)
    # Microsoft Visual C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()

# Additional Optimization Flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")