#include <algorithm>
#include <array>
#include <cassert>

#define __device__
#define __forceinline__ inline
#include "../../../zig/kernels/cuda/exl3_linear_reset.cuh"

int main() {
    constexpr int guard = 16;
    constexpr int capacity = 8192;
    std::array<int, capacity + 2 * guard> storage;
    for (int blocks : {1, 2, 5, 16, 64}) {
        for (int count : {0, 8, 512, 7760, 1, capacity, 8}) {
            storage.fill(0x5a5a5a5a);
            for (int block = blocks - 1; block >= 0; --block) {
                for (int thread = 127; thread >= 0; --thread) {
                    reset_linear_counters(storage.data() + guard, count, block, thread, blocks);
                }
            }
            assert(std::all_of(storage.begin(), storage.begin() + guard, [](int value) { return value == 0x5a5a5a5a; }));
            assert(std::all_of(storage.begin() + guard, storage.begin() + guard + count, [](int value) { return value == 0; }));
            assert(std::all_of(storage.begin() + guard + count, storage.end(), [](int value) { return value == 0x5a5a5a5a; }));
        }
    }
    reset_linear_counters(nullptr, 0, 0, 0, 1);
}
