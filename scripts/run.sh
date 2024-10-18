#!/bin/bash

# Path to the executable
EXECUTABLE="../cmake-build-debug/parallel"

# Number of test repetitions
test_repetitions=5

# Sort order (1 for ascending, 0 for descending)
sort_order=1

# Number of threads
num_threads=1

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
# For example from 2^12 to 2^30 in steps of 2 (i.e., 2^12, 2^14, 2^16, ...)
for i in {12..30..2}
do
    array_length=$((2**i))
    echo "----------------------------------------"
    echo "Running experiment with array length 2^$i = $array_length"
    $EXECUTABLE $array_length $test_repetitions $sort_order $num_threads $skip_gpu "$run_folder"
    echo "Experiment completed for array length 2^$i"
    echo "----------------------------------------"
done

echo "All experiments completed!"

