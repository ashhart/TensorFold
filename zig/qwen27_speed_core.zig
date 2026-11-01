//! Shared host planning and profiling used by the recording Metal fixture.
pub const tree_slots = @import("src/core/tree_slots.zig");
pub const gpu_profile = @import("src/core/gpu_profile.zig");
pub const checkpoint_host = @import("src/core/checkpoint.zig");

pub const draft_ops = struct {
    pub const Ref = struct { buf: @import("metal").Buffer, off: usize = 0 };
};
