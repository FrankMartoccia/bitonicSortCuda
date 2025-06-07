# Parallel Bitonic Sort

A high-performance parallel implementation of the Bitonic Sort algorithm on both CPU (C++20 with OpenMP) and GPU (CUDA), developed as part of the **Computer Architecture** course project at the University of Pisa (MSc in Computer Engineering, A.Y. 2023/2024).

## Structure

- `docs/` – Project documentation
- `results/` – Raw results for CPU and GPU runs
- `scripts/` – Automation scripts to run the experiments and collect data
- `src/`
  - `cpu/` – CPU implementation (v1 and optimized v2)
  - `gpu/` – CUDA kernels and GPU sorting logic

## Technologies

- **Languages**: C++20, CUDA C++
- **Parallelism**: OpenMP, CUDA threads
- **Profiling Tools**: AMD µProf, Visual Studio Profiler, Nsight Compute
- **Hardware Tested**:
  - AMD Ryzen 9 7940HS (CPU)
  - NVIDIA GeForce RTX 4070 (GPU)

## Implementation

- **CPU Version 1**: Multi-threaded bitonic sort using C++20 and OpenMP.
- **CPU Version 2**: Optimized version using bucket sort for large arrays and SIMD-enhanced bitonic sort for small ones. Improved cache efficiency and speedup.
- **GPU Version 1**: CUDA implementation using global memory for sorting and merging.
- **GPU Version 2**: Optimized CUDA version using shared memory to maximize throughput and reduce memory latency.
- **Profiling and Analysis**: Includes performance benchmarks, cache usage analysis, and GPU profiling with AMD µProf and NVIDIA Nsight Compute.
