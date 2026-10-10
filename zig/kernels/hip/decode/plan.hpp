#pragma once

#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

// A lane round's plan in device memory: no stream state is a launch argument, so one launch or graph serves any.
struct PlanArgs {
    const int* pos;                   // per row: its position
    const int* slot;                  // per row: its slot
    const int* first;                 // per slot: its first row
    const int* count;                 // per slot: its rows
    const unsigned long long* desc;   // per slot: the address of its descriptor
    const unsigned long long* snaps;  // per layer: the round's conv and DeltaNet snapshots, a row each
    unsigned pages;                   // words of a descriptor before its page table
    unsigned pool;                    // pages a head of a layer's pool holds
};

// Slot descriptor: [0] cached positions, [1] last kept final row, two cache addresses a layer, then the page table.
__device__ __forceinline__ const unsigned long long* plan_desc(const PlanArgs& p, int slot) {
    return reinterpret_cast<const unsigned long long*>(p.desc[slot]);
}

__device__ __forceinline__ int plan_first(int layer) { return 2 + 2 * layer; }

__device__ __forceinline__ int plan_second(int layer) { return 3 + 2 * layer; }

__device__ __forceinline__ const unsigned* plan_pages(const PlanArgs& p, const unsigned long long* desc) {
    return reinterpret_cast<const unsigned*>(desc + p.pages);
}
