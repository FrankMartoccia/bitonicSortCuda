#include "TimerCPU.h"

// Constructor
TimerCPU::TimerCPU()
{
    reset();
}

// Resets the timer
void TimerCPU::reset()
{
    m_StartingTime = std::chrono::high_resolution_clock::now();
}

// Returns the elapsed time in milliseconds
float TimerCPU::getElapsedMilliseconds() const
{
    return getElapsed() * 1000.0f;
}

// Returns the elapsed time in seconds
float TimerCPU::getElapsed() const
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - m_StartingTime)
               .count() * 0.001f * 0.001f * 0.001f;
}
