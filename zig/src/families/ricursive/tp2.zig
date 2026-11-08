//! Ripple (model_type ricursive) on two GB10s (sm_121, --tp 2): an ExLlamaV3 pack cut over both ranks.
const std = @import("std");
pub const config = @import("config.zig");
pub const kernels = @import("tp2_kernels.zig");
pub const weights = @import("tp2_weights.zig");
pub const forward = @import("tp2_forward.zig");
pub const Model = forward.Model;
pub const native = @import("tp2_native.zig");

/// Bytes this rank's weights take: its share of the cut matrices, plus the bf16 embedding every rank holds whole.
pub fn placementWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    _ = drafts;
    const total = try shardBytes(a, io, dir);
    try forward.admit(121, world, rank);
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    defer a.free(bytes);
    const c = try config.parse(a, bytes);
    return share(total, @as(u64, c.vocab) * c.hidden * 2, world);
}

/// The checkpoint's shard files' bytes.
pub fn shardBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const core = @import("core");
    const files = try core.checkpoint.shardFiles(a, io, dir);
    defer core.checkpoint.freeShardFiles(a, files);
    var total: u64 = 0;
    for (files) |f| total += (try std.Io.Dir.cwd().statFile(io, f, .{})).size;
    return total;
}

fn share(total: u64, whole: u64, world: usize) u64 {
    if (total < whole) return total;
    return (total - whole) / world + whole;
}

test "ricursive weight share: the cut part halves, the embedding stays whole" {
    try std.testing.expectEqual(@as(u64, 1000), share(1000, 2000, 2));
    try std.testing.expectEqual(@as(u64, 600), share(1000, 200, 2));
}

test {
    _ = config;
    _ = kernels;
    _ = weights;
    _ = forward;
    _ = native;
    _ = @import("tp2_lanes.zig");
}
