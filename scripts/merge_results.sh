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

# Loop over all the sorting result files in the results directory
for file in "$RESULTS_DIR"/array_length_*.csv "$RESULTS_DIR"/array_size_*.csv; do
    echo "Processing $file..."

    # Determine if filename contains '2^' or 'MB' to extract the array size
    if [[ "$file" == *"2^"* ]]; then
        # Extract the power of 2 array length
        exponent=$(echo "$file" | grep -oP '(?<=array_length_2\^)\d+')
        array_length=$((2 ** exponent))
        echo "Array Length: 2^$exponent = $array_length" >> "$OUTPUT_FILE"
    elif [[ "$file" == *"MB"* ]]; then
        # Extract the size in MB
        size_in_mb=$(echo "$file" | grep -oP '(?<=array_size_)\d+(\.\d+)?(?=MB)')
        array_length=$(echo "$size_in_mb * 1000000 / 4" | bc) # Convert MB to number of 32-bit integers
        echo "Array Size: ${size_in_mb}MB, Array Length: $array_length" >> "$OUTPUT_FILE"
    else
        echo "Unknown file format: $file"
        continue
    fi

    # Extract the number of GPU threads from the file
    threads=$(grep "Threads (GPU):" "$file" | cut -d ' ' -f 3)

    # Add the Array Length and Threads (GPU) info in the correct format before appending the data
    echo "Threads (GPU):,$threads" >> "$OUTPUT_FILE"

    # Append only the relevant data
    tail -n +7 "$file" | sed '/^Threads (GPU):/d' >> "$OUTPUT_FILE"

    # Add blank lines for separation between runs
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Data successfully merged into $OUTPUT_FILE"
