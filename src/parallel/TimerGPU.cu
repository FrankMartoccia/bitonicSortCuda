#include "TimerGPU.cuh"
#include "cuda_runtime.h"

/*
 * Constructor: Creates two CUDA events, m_Start and m_Stop, that will be used to mark
 * the start and stop of the GPU timing process. These events are created using cudaEventCreate.
 */
TimerGPU::TimerGPU()
{
    // Initialize CUDA events to mark the start and stop points
    cudaEventCreate(&m_Start);
    cudaEventCreate(&m_Stop);
}

/*
 * Starts the timer by recording the m_Start event. This marks the point from
 * which the time measurement will begin. The event is recorded asynchronously.
 */
void TimerGPU::start() const
{
    cudaEventRecord(m_Start);
}

/*
 * Stops the timer by recording the m_Stop event. This marks the point at which
 * the time measurement will end. The event is recorded asynchronously.
 */
void TimerGPU::stop() const
{
    cudaEventRecord(m_Stop);
}

/*
 * Returns the elapsed time in milliseconds between the start and stop events.
 * This function synchronizes the m_Stop event, ensuring that the stop event has been completed
 * before calculating the time difference using cudaEventElapsedTime.
 */
float TimerGPU::getElapsedMilliseconds() const
{
    // Ensure the stop event has been recorded before measuring the time
    cudaEventSynchronize(m_Stop);

    // Compute the elapsed time in milliseconds between the start and stop events
    float time = 0.0f;
    cudaEventElapsedTime(&time, m_Start, m_Stop);

    return time;
}
