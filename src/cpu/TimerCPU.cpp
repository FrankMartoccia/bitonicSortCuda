#include "TimerCPU.h"

// Starts or resets the timer by recording the current time point
void TimerCPU::start()
{
    m_StartingTime = std::chrono::high_resolution_clock::now();
}

// Returns the elapsed time since the timer started, in milliseconds
// Multiplies the elapsed time in seconds by 1000 to convert to milliseconds
float TimerCPU::getElapsedMilliseconds() const
{
    return getElapsed() * 1000.0f;
}

// Returns the elapsed time since the timer started, in seconds
// The duration is cast from nanoseconds to seconds
float TimerCPU::getElapsed() const
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - m_StartingTime)
               .count() * 0.001f * 0.001f * 0.001f;
}
