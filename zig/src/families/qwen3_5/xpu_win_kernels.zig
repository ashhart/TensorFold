//! The kernel handles of the Qwen prompt and verification windows (xpu_win.zig).
const std = @import("std");
const rt = @import("xpu").rt;

/// Value rows a sub-group of the delta-rule window kernel (env ARC_GDN_ROWS = 1 | 2 | 4): same bits, different speed.
pub fn gdnRows() u32 {
    const v = std.c.getenv("ARC_GDN_ROWS") orelse return 4;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch return 1;
    return if (n == 2 or n == 4) n else 1;
}

pub fn gdnStepName() [*:0]const u8 {
    return switch (gdnRows()) {
        2 => "gdn_step_r2x2",
        4 => "gdn_step_r2x4",
        else => "gdn_step_r2",
    };
}

pub const Kernels = struct {
    embed4: rt.Kernel,
    embed16: rt.Kernel,
    swiglu: rt.Kernel,
    add: rt.Kernel,
    round: rt.Kernel,
    am_part: rt.Kernel,
    am_fin: rt.Kernel,
    add_rms: rt.Kernel,
    add_rms32: rt.Kernel,
    rms: rt.Kernel,
    rms32: rt.Kernel,
    g_prep: rt.Kernel,
    g_step: rt.Kernel,
    g_vconv: rt.Kernel,
    g_gates: rt.Kernel,
    g_norm: rt.Kernel,
    g_commit: rt.Kernel,
    mv16r: rt.Kernel,
    a_prep: rt.Kernel,
    a_prep32: rt.Kernel,
    a_part: rt.Kernel,
    a_merge: rt.Kernel,
};
