#!/bin/bash

# Path to the executable
EXECUTABLE="../cmake-build-release/parallel"

# Path to the constants header file
CONSTANTS_FILE="../src/gpu/constants.h"

# Number of test repetitions
test_repetitions=30

# Sort order (1 for ascending, 0 for descending)
sort_order=1

# Number of threads
num_threads=16

# Skip GPU sort (1 to skip, 0 to run GPU sort)
skip_gpu=0

# Array length for the tests
array_length=$((2**25))  # Replace with the desired fixed size if needed

# List of configurations to test
configurations=(
    "BITONIC_SORT_THREADS=32 BITONIC_SORT_BLOCKS=8192
     MERGE_GLOBAL_THREADS=64 MERGE_GLOBAL_BLOCKS=4096
      MERGE_LOCAL_THREADS=64 MERGE_LOCAL_BLOCKS=4096"

    "BITONIC_SORT_THREADS=64 BITONIC_SORT_BLOCKS=8192
     MERGE_GLOBAL_THREADS=128 MERGE_GLOBAL_BLOCKS=4096
      MERGE_LOCAL_THREADS=128 MERGE_LOCAL_BLOCKS=4096"

    "BITONIC_SORT_THREADS=128 BITONIC_SORT_BLOCKS=8192
     MERGE_GLOBAL_THREADS=256 MERGE_GLOBAL_BLOCKS=4096
      MERGE_LOCAL_THREADS=256 MERGE_LOCAL_BLOCKS=4096"

    "BITONIC_SORT_THREADS=384 BITONIC_SORT_BLOCKS=8192
     MERGE_GLOBAL_THREADS=768 MERGE_GLOBAL_BLOCKS=4096
      MERGE_LOCAL_THREADS=768 MERGE_LOCAL_BLOCKS=4096"

    "BITONIC_SORT_THREADS=384 BITONIC_SORT_BLOCKS=16384
     MERGE_GLOBAL_THREADS=768 MERGE_GLOBAL_BLOCKS=8192
      MERGE_LOCAL_THREADS=768 MERGE_LOCAL_BLOCKS=8192"

    "BITONIC_SORT_THREADS=384 BITONIC_SORT_BLOCKS=32768
     MERGE_GLOBAL_THREADS=768 MERGE_GLOBAL_BLOCKS=16384
      MERGE_LOCAL_THREADS=768 MERGE_LOCAL_BLOCKS=16384"

    "BITONIC_SORT_THREADS=384 BITONIC_SORT_BLOCKS=65536
     MERGE_GLOBAL_THREADS=768 MERGE_GLOBAL_BLOCKS=32768
      MERGE_LOCAL_THREADS=768 MERGE_LOCAL_BLOCKS=32768"
)

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    exit 1
fi

# Determine the folder suffix based on skip_gpu (CPU if skip, GPU if not)
folder_suffix=$([ "$skip_gpu" -eq 1 ] && echo "CPU" || echo "GPU")

# Create a unique results folder for this set of experiments
timestamp=$(date +"%Y%m%d_%H%M%S")
results_folder="../results/run_${folder_suffix}_$timestamp"
mkdir -p "$results_folder"

# Loop over each configuration, modify constants.h, recompile, and execute
for idx in "${!configurations[@]}"; do
    config="${configurations[idx]}"

    # Update constants.h with current configuration
    echo "Updating constants.h with configuration $((idx + 1))"
    echo "#ifndef CONSTANTS_H" > "$CONSTANTS_FILE"
    echo "#define CONSTANTS_H" >> "$CONSTANTS_FILE"
    echo "" >> "$CONSTANTS_FILE"
    echo "// Sorting order constants" >> "$CONSTANTS_FILE"
    echo "constexpr int ORDER_ASC = 1;" >> "$CONSTANTS_FILE"
    echo "constexpr int ORDER_DESC = 0;" >> "$CONSTANTS_FILE"
    echo "" >> "$CONSTANTS_FILE"

    for entry in $config; do
        key=$(echo $entry | cut -d '=' -f 1)
        value=$(echo $entry | cut -d '=' -f 2)
        echo "constexpr int $key = $value;" >> "$CONSTANTS_FILE"
    done

    echo "#endif // CONSTANTS_H" >> "$CONSTANTS_FILE"

    # Recompile the program
    echo "Compiling with configuration $((idx + 1))"
    if ! cmake --build ../cmake-build-release --target parallel -- -j 14; then
        echo "Compilation failed for configuration $((idx + 1))"
        exit 1
    fi

    # Define a unique filename for the current configuration results
    result_config_folder="${results_folder}/results_config_$((idx + 1))"

    # Run the executable and save output to the result file
    echo "Running experiment with configuration $((idx + 1))"
    $EXECUTABLE $array_length $test_repetitions $sort_order $num_threads $skip_gpu "$result_config_folder"

    echo "Experiment completed for configuration $((idx + 1)), results saved in $result_config_folder"
done

echo "All experiments completed! Results are saved in $results_folder"
