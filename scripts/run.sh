#!/bin/bash

# Path to the parallel executable
EXECUTABLE="../cmake-build-debug/parallel"

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    exit 1
fi

# Run experiments for array lengths from 2^12 to 2^30 in steps of 2 (i.e., 2^12, 2^14, 2^16, ...)
for i in {12..30..2}
do
    array_length=$((2**i))
    echo "Running experiment with array length 2^$i = $array_length"
    $EXECUTABLE $array_length 5 1
    echo "Experiment completed for array length 2^$i"
    echo "----------------------------------------"
done

echo "All experiments completed!"

