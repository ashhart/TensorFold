//! Mini DFlash 2 drafter shader sources.

pub const dflash2 = @embedFile("dflash2.metal");
pub const dflash2_fast = @embedFile("dflash2_fast.metal");
pub const dflash2_fast_pre_m5 = "#define TF_SIMD_FRAGS 1\n" ++ dflash2_fast;
