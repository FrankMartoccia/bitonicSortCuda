#ifndef TIMERGPU_CUH
#define TIMERGPU_CUH

#include "cuda_runtime.h"

// GpuTimer class for measuring elapsed time on GPU using CUDA events
class TimerGPU
{
public:
    TimerGPU();

    void start() const;
    void stop() const;
    float getElapsedMilliseconds() const;

private:
    cudaEvent_t m_Start{};
    cudaEvent_t m_Stop{};
};

#endif
