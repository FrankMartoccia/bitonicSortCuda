#ifndef TIMERGPU_CUH
#define TIMERGPU_CUH

#include "cuda_runtime.h"

// The TimerGPU class provides functionality for measuring elapsed time on the GPU using CUDA events.
// It allows you to start, stop, and measure the elapsed time in milliseconds between the start and stop events.
class TimerGPU
{
public:
    // Constructor: Initializes the CUDA events used to mark the start and stop of the timer
    TimerGPU();

    // Starts the timer by recording the start event
    void start() const;

    // Stops the timer by recording the stop event
    void stop() const;

    // Returns the elapsed time between the start and stop events, in milliseconds
    float getElapsedMilliseconds() const;

private:
    cudaEvent_t m_Start{};  // CUDA event for the start time
    cudaEvent_t m_Stop{};   // CUDA event for the stop time
};

#endif
