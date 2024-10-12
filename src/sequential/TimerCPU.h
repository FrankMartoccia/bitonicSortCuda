#ifndef TIMERCPU_H
#define TIMERCPU_H
#include <chrono>


class TimerCPU
{
public:

    void start();

    float getElapsedMilliseconds() const;

private:
    float getElapsed() const;

    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartingTime;
};


#endif //TIMERCPU_H
