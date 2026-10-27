#pragma once

__device__ __forceinline__ void reset_linear_counters(int* counters, int count, int block, int thread, int blocks) {
    for (int index = block * 128 + thread; index < count; index += blocks * 128) counters[index] = 0;
}
