// Extern "C" entry points for Hy4-flash fused iHC boundary kernels (ticket, cooperative, tiled, and 1-pass prefill).
#include "ihc_fused.cuh"

using hy4ihc::Args;

#define TICKET(name, POST, M)                                                                                   \
    extern "C" __global__ void __launch_bounds__(hy4ihc::NT, 1) name(const __grid_constant__ Args a) {          \
        hy4ihc::ticket_body<POST, M>(a);                                                                        \
    }
TICKET(hy4_ihc_fused_boundary, true, 8)
TICKET(hy4_ihc_entry, false, 8)
TICKET(hy4_ihc_epilogue_head, true, 4)
TICKET(hy4_ihc_head_only, false, 4)
TICKET(hy4_ihc_post_only, true, 0)
#undef TICKET

#define COOP(name, POST, M)                                                                                     \
    extern "C" __global__ void __launch_bounds__(hy4ihc::NT, 1) name(const __grid_constant__ Args a) {          \
        hy4ihc::coop_body<POST, M>(a);                                                                          \
    }
COOP(hy4_ihc_fused_boundary_coop, true, 8)
COOP(hy4_ihc_entry_coop, false, 8)
COOP(hy4_ihc_epilogue_head_coop, true, 4)
COOP(hy4_ihc_head_only_coop, false, 4)
#undef COOP

extern "C" __global__ void __launch_bounds__(hy4ihc::NT) hy4_ihc_post_vec(const __grid_constant__ Args a) {
    hy4ihc::post_vec_body(a);
}
extern "C" __global__ void __launch_bounds__(hy4ihc::NT, 4) hy4_ihc_pre_tiled_rb4(const __grid_constant__ Args a) {
    hy4ihc::pre_tiled_body<8, 4>(a);
}
extern "C" __global__ void __launch_bounds__(hy4ihc::NT, 4) hy4_ihc_pre_tiled_rb2(const __grid_constant__ Args a) {
    hy4ihc::pre_tiled_body<8, 2>(a);
}
extern "C" __global__ void __launch_bounds__(hy4ihc::NT, 4) hy4_ihc_head_tiled(const __grid_constant__ Args a) {
    hy4ihc::pre_tiled_body<4, 8>(a);
}
extern "C" __global__ void __launch_bounds__(hy4ihc::NT) hy4_ihc_fused_prefill_m8(const __grid_constant__ Args a) {
    hy4ihc::fused_prefill_body<8>(a);
}
extern "C" __global__ void __launch_bounds__(hy4ihc::NT) hy4_ihc_fused_prefill_m4(const __grid_constant__ Args a) {
    hy4ihc::fused_prefill_body<4>(a);
}
