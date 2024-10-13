#!/bin/bash

# Define the path to the results folder
RESULTS_DIR="../results"

# Define the output file path within the results folder
OUTPUT_FILE="$RESULTS_DIR/merged_sorting_results.csv"

# Check if the output file already exists and remove it to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    echo "Removing existing $OUTPUT_FILE..."
    rm "$OUTPUT_FILE"
fi

# Loop over all the sorting result files in the results directory
for file in "$RESULTS_DIR"/sorting_results_*.csv; do
    echo "Processing $file..."

    # Extract the array length from the file (from the first line)
    array_length=$(head -n 1 "$file" | cut -d ' ' -f 3)

    # Calculate the exponent of 2 (log base 2 of the array length)
    exponent=$(echo "l($array_length)/l(2)" | bc -l | awk '{printf "%d", $1}')

    # Extract the number of threads from the file (search for "Threads" line)
    threads=$(grep "Threads" "$file" | cut -d ' ' -f 2)

    # Add the Array Length and Threads info in the correct format before appending the data
    echo "Array Length: 2^$exponent = $array_length" >> "$OUTPUT_FILE"
    
    # Append only the relevant data (skip the first 6 lines and append the rest)
    tail -n +7 "$file" >> "$OUTPUT_FILE"

    echo "" >> "$OUTPUT_FILE"  # Add a blank line for separation between runs
done

echo "Data successfully merged into $OUTPUT_FILE"

