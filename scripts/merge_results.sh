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

# Find all unique array lengths based on file names like array_length_2^16_threads_1.csv
array_lengths=$(ls "$RESULTS_DIR"/array_length_2^*_threads_*.csv | sed -n 's/.*array_length_2^\([0-9]*\)_threads_.*\.csv/\1/p' | sort -n | uniq)

# Loop over each unique array length
for array_length in $array_lengths; do
    # Define the output file path for this array length
    OUTPUT_FILE="$RESULTS_DIR/merged_sorting_results_2^${array_length}.csv"
    
    # Check if the output file already exists and remove it to start fresh
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Removing existing $OUTPUT_FILE..."
        rm "$OUTPUT_FILE"
    fi

    echo "Processing results for array length 2^$array_length..."

    # Start by writing the header for this array length in the output file
    echo "Array Length: 2^$array_length" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Add a blank line for separation between runs

    # Get all files for the current array length, extract thread count, sort numerically, then process
    ls "$RESULTS_DIR"/array_length_2^${array_length}_threads_*.csv |
    while read file; do
        threads=$(echo "$file" | sed -n 's/.*_threads_\([0-9]*\)\.csv/\1/p')
        echo "$threads $file"
    done |
    sort -n |
    while read threads file; do
        echo "  Merging data for $threads threads..."

        # Append a section header for the current thread count
        echo "Threads:,$threads" >> "$OUTPUT_FILE"

        # Append only the relevant data (skip the first 6 lines and append the rest)
        # Use sed to remove any extra "Threads:" line that might be in the data
        tail -n +7 "$file" | sed '/^Threads:/d' >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"  # Add a blank line for separation between runs
        echo "" >> "$OUTPUT_FILE"  # Add a blank line for separation between runs
    done

    echo "Data merged into $OUTPUT_FILE"
    echo "----------------------------------------"
done

echo "All data successfully merged."