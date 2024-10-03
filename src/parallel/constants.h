#ifndef CONSTANTS_H
#define CONSTANTS_H

constexpr int ORDER_ASC = 0;
constexpr int ORDER_DESC = 1;

constexpr int THREADS_BITONIC_SORT = 128;
constexpr int ELEMENTS_BITONIC_SORT = 4;

constexpr int THREADS_GLOBAL_MERGE = 256;
constexpr int ELEMENTS_GLOBAL_MERGE = 4;

constexpr int THREADS_LOCAL_MERGE = 256;
constexpr int ELEMENTS_LOCAL_MERGE = 8;


#endif