//! Inkling 2 Mini loop operator source (families/inkling2/metal_loop.zig). Register as
//! `pub const inkling2_loop_ops = @import("inkling2/loop_sources.zig").ops;` in kernels/metal/sources.zig.

pub const common = @embedFile("common.metal");
pub const loop = @embedFile("loop.metal");

/// The library the runtime compiles: common helpers, then the three kernels named inkling2_loop_*.
pub const ops = common ++ "\n" ++ loop;
