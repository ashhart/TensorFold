//! Mini loop source (metal_loop.zig); register as `inkling2_loop_ops = @import("inkling2/loop_sources.zig").ops`.

pub const common = @embedFile("common.metal");
pub const loop = @embedFile("loop.metal");

/// The library the runtime compiles: common helpers, then the three kernels named inkling2_loop_*.
pub const ops = common ++ "\n" ++ loop;
