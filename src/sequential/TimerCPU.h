#ifndef TIMERCPU_H
#define TIMERCPU_H

#include <chrono> // For high-resolution clock and time points

// TimerCPU class is used for measuring elapsed time using the CPU clock
class TimerCPU
{
public:
    // Starts or resets the timer
    void start();

    // Returns the elapsed time since the timer started, in milliseconds
    float getElapsedMilliseconds() const;

private:
    // Returns the elapsed time since the timer started, in seconds
    float getElapsed() const;

    // Time point that stores the start time of the timer
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartingTime;
};

#endif // TIMERCPU_H
