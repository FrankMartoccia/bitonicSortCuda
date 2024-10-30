#!/bin/bash

# Path to the executable
EXECUTABLE="../cmake-build-release/parallel"

# Array lengths 2^17, 2^19, 2^20, 2^25, 2^26, 2^27
array_lengths=(131072 524288 1048576 33554432 67108864 134217728)

# Array of thread counts
thread_counts=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)

# Number of test repetitions
test_repetitions=10

# Sort order (1 for ascending, 0 for descending)
sort_order=1

# Skip GPU sort? (1 to skip, 0 to run GPU sort)
skip_gpu=1

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    exit 1
fi

# Determine the folder suffix based on skip_gpu (CPU if skip, GPU if not)
folder_suffix=$([ "$skip_gpu" -eq 1 ] && echo "CPU" || echo "GPU")

# Create a unique folder for this execution, with either "CPU" or "GPU" in the name
timestamp=$(date +"%Y%m%d_%H%M%S")
run_folder="../results/run_${folder_suffix}_$timestamp"
mkdir -p "$run_folder"

# Loop over array lengths
for length in "${array_lengths[@]}"; do
    # Loop over number of threads
    echo "----------------------------------------"
    echo "Running experiments with array length 2^$(echo "l($length)/l(2)" | bc -l | xargs printf "%.0f") = $length"
    for threads in "${thread_counts[@]}"; do
        echo "Running with thread count: $threads, GPU skip: $skip_gpu"

        #  Execute the program with the current parameters
        $EXECUTABLE $length $test_repetitions $sort_order $threads $skip_gpu "$run_folder"

        echo "Experiment completed for array length = $length and thread count = $threads"
        echo "----------------------------------------"
    done
done
