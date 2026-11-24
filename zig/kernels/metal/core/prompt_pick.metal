// Kernel NAME: after a prompt's non-finite scan, the finite (grids 0-2) or masked (3-5) grids; clears the flag.
#define TF_PROMPT_PICK(NAME)                                                                                       \
  [[kernel]] void NAME(device atomic_uint* flag [[buffer(0)]], constant uint* grid [[buffer(1)]],                  \
      device uint* grids [[buffer(2)]]) {                                                                          \
    const bool masked = atomic_load_explicit(flag, memory_order_relaxed) != 0;                                     \
    atomic_store_explicit(flag, 0u, memory_order_relaxed);                                                         \
    for (int i = 0; i < 3; i++) { grids[i] = masked ? 0u : grid[i]; grids[3 + i] = masked ? grid[i] : 0u; }       \
  }
