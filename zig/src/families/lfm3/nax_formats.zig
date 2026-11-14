//! The weight formats LFM3's tensor-unit prompt route offers, and the one launch each of them compiles to.

pub const Format = struct { bits: u8, group: u16 };

/// Every format `prompt_nax` builds a pipeline for: each has a launch-limit record and a kernel-contract probe.
pub const formats = [_]Format{
    .{ .bits = 4, .group = 64 },
    .{ .bits = 6, .group = 64 },
    .{ .bits = 8, .group = 64 },
    .{ .bits = 4, .group = 32 },
    .{ .bits = 6, .group = 32 },
    .{ .bits = 8, .group = 32 },
};

/// `tf_qmmw_t_nax_win` runs four simdgroups of 32 for every format: prompt_nax's dispatch.
pub const threads = 128;

/// The kernel's two bf16 arrays, a 64 x 72 tile and a 16 x 64 scale window (prefill/qmmw_nax.metal), for every format.
pub const threadgroup_bytes = 2 * (64 * (64 + 8) + 16 * 64);

pub fn supported(bits: u8, group: u16) bool {
    for (formats) |f| if (f.bits == bits and f.group == group) return true;
    return false;
}
