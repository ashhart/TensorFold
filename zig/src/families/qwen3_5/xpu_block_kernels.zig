//! The kernel handles of the single-token Qwen decode ops (xpu_blocks.zig).
const rt = @import("xpu").rt;

pub const Kernels = struct {
    embed: rt.Kernel,
    embed16: rt.Kernel,
    mv16: rt.Kernel,
    rms: rt.Kernel,
    rms512: rt.Kernel,
    add_rms: rt.Kernel,
    add_rms32: rt.Kernel,
    add: rt.Kernel,
    round: rt.Kernel,
    qmv: rt.Kernel,
    gateup: rt.Kernel,
    head: rt.Kernel,
    swiglu: rt.Kernel,
    am_part: rt.Kernel,
    am_fin: rt.Kernel,
    g_prep: rt.Kernel,
    g_step: rt.Kernel,
    g_norm: rt.Kernel,
    a_prep: rt.Kernel,
    a_part: rt.Kernel,
    a_merge: rt.Kernel,
};
