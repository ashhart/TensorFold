//! Mini DFlash 2 drafter shader source.
const guard = @import("../threadgroup_source.zig");

pub const dflash2 = guard.normalize(@embedFile("dflash2.metal"));
