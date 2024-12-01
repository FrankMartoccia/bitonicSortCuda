#!/bin/bash

# Path to the executable
EXECUTABLE="../cmake-build-release/parallel"

# Number of test repetitions
test_repetitions=30

# Sort order (1 for ascending, 0 for descending)
sort_order=1

# Number of threads
num_threads=16

# Skip GPU sort (1 to skip, 0 to run GPU sort)
skip_gpu=0

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

# Run experiments for array lengths as the powers of 2 specified
for i in {18..26}
do
    array_length=$((2**i))  # Divide by 4 because each element is 4 bytes (32 bits)
    size_in_mb=$(((2**i * 4) / 1024 / 1024))  # Calculate the size in MB for display
    echo "----------------------------------------"
    echo "Running experiment with array length 2^$i = ${size_in_mb}MB ($array_length elements)"
    $EXECUTABLE $array_length $test_repetitions $sort_order $num_threads $skip_gpu "$run_folder"
    echo "Experiment completed for array length 2^$i = ${size_in_mb}MB"
    echo "----------------------------------------"
done

echo "All experiments completed!"

