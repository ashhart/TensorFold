//! Data and kernel-call types shared by Qwen's CUDA MTP draft path.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Stream = lanes.Stream;

pub const head_dim = @import("cuda_geometry.zig").head_dim;
pub const theta = @import("cuda_geometry.zig").theta;
pub const max_rows = engine.most_verify;
pub const most_window = engine.most_window;
pub const max_segs = engine.max_segs;
pub const default_depth: u32 = 3;
pub const default_initial_prompt_rows: u32 = 256;
pub const max_drafts: u32 = @intCast(most_window - 1);
pub const max_lanes = most_window;
pub const ranks = lanes.shape.ranks;
pub const lp_words = 3 + 2 * 20;

pub const Fns = struct {
    norm_rope: cuda.Function,
    attn: cuda.Function,
    attn_tree: cuda.Function,
    attn_rows: cuda.Function,
};

pub const Ring = struct {
    gather_taps: cuda.Function,
    gather_rows: cuda.Function,
    ring_put: cuda.Function,
};

pub const TapRows = extern struct { n: c_int, rows: [most_window]c_int };

/// A stream's MTP state and drafts held from the newest target row.
pub const Ctx = struct {
    ring: DeviceBuffer,
    last: DeviceBuffer,
    step: DeviceBuffer,
    held_dev: DeviceBuffer,
    begin: u64 = 0,
    end: u64 = 0,
    cands: [ranks]u32 = @splat(0),
    held: [max_lanes]u32 = undefined,
    held_n: usize = 0,
};

pub const Pass = union(enum) { ring, tree: u64 };

pub const RingOf = struct {
    ctx: *anyopaque,
    area: *const fn (*anyopaque, *Stream) ?u64,
    all: bool = false,
};

test "dense MTP defaults to a 256-row initial prompt history" {
    try std.testing.expectEqual(@as(u32, 256), default_initial_prompt_rows);
}
