#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_name>"
    echo "Example: $0 run_GPU_20241017_211341"
    exit 1
fi

# Define the path to the results folder and folder specified by the user
RESULTS_DIR="../results/$1"

# Check if the folder exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Folder $RESULTS_DIR not found!"
    exit 1
fi

# Define the output file path within the results folder
OUTPUT_FILE="$RESULTS_DIR/merged_sorting_results.csv"

# Check if the output file already exists and remove it to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing $OUTPUT_FILE..."
    rm "$OUTPUT_FILE"
fi

# Create an associative array to store filenames by their GPU thread count for sorting
declare -A file_map

# Loop over all the sorting result files in the results directory
for file in "$RESULTS_DIR"/array_length_*.csv; do
    echo "Processing $file..."

    # Extract GPU thread count from the filename, if present
    if [[ "$file" =~ _gpu_threads_([0-9]+)\.csv$ ]]; then
        threadsGPU="${BASH_REMATCH[1]}"
    else
        threadsGPU="0"  # Default to 0 if GPU threads are not present (CPU-only runs)
    fi

    # Use GPU thread count as the key to store each file
    file_map["$threadsGPU"]="$file"
done

# Sort keys by GPU thread count and process files in order
for threadsGPU in $(printf "%s\n" "${!file_map[@]}" | sort -n); do
    file="${file_map[$threadsGPU]}"

    # Extract array length from the filename
    array_length=$(grep "Array Length:" "$file" | cut -d ' ' -f 3)

    # Calculate the exponent of 2 (log base 2 of the array length)
    exponent=$(echo "l($array_length)/l(2)" | bc -l | awk '{printf "%d", $1}')

    # Extract CPU thread count from the filename
    if [[ "$file" =~ _cpu_threads_([0-9]+) ]]; then
        cpuThreads="${BASH_REMATCH[1]}"
    else
        cpuThreads="N/A"  # Default to "N/A" if CPU threads are not specified
    fi

    # Append header information for the current file to the output file
    echo "Array Length: 2^$exponent = $array_length" >> "$OUTPUT_FILE"
    echo "CPU Threads:,$cpuThreads" >> "$OUTPUT_FILE"
    if [ "$threadsGPU" -ne 0 ]; then
        echo "GPU Threads:,$threadsGPU" >> "$OUTPUT_FILE"
    fi

    # Append the relevant data, skipping headers and blank lines
    tail -n +7 "$file" | sed '/^Threads (GPU):/d' >> "$OUTPUT_FILE"

    # Add blank lines for separation between runs
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Data successfully merged into $OUTPUT_FILE"
