//! Mini DFlash 2 drafter shader source.
const guard = @import("../threadgroup_source.zig");

pub const dflash2 = guard.normalize(@embedFile("dflash2.metal"));

pub const dflash2_fast = guard.normalize(@embedFile("dflash2_fast.metal"));
pub const dflash2_fast_pre_m5 = guard.normalize("#define TF_SIMD_FRAGS 1\n" ++ dflash2_fast);
