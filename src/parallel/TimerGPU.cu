#include "TimerGPU.cuh"

#include "cuda_runtime.h"

// Constructor
TimerGPU::TimerGPU()
{
    cudaEventCreate(&m_Start);
    cudaEventCreate(&m_Stop);
}

// Starts the timer
void TimerGPU::start() const
{
    cudaEventRecord(m_Start);
}

// Stops the timer
void TimerGPU::stop() const
{
    cudaEventRecord(m_Stop);
}

// Returns the elapsed time in milliseconds
float TimerGPU::getElapsedMilliseconds() const
{
    cudaEventSynchronize(m_Stop);
    float time = 0.0f;
    cudaEventElapsedTime(&time, m_Start, m_Stop);
    return time;
}
