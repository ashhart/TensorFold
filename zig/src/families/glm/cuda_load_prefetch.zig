//! Read-only layer prefetch through core; uploads and page drops retain their loader's order.
const std = @import("std");
const core = @import("core");

/// The rank's cut of a layer's routed experts, in intermediate elements: cuda_shard.Span's pair without its import.
pub const Cut = struct { lo: usize, hi: usize };

/// A layer's pages read ahead, `count` threads over the byte ranges its EXL3 loader will take out of `cut`.
pub fn begin(ck: *const core.Checkpoint, prefix: []const u8, i: usize, count: usize, cut: Cut) !core.checkpoint_read_ahead.Scope {
    if (count == 0) return .{};
    var buf: [160]u8 = undefined;
    var own = cut;
    const keep: core.checkpoint_read_ahead.Keep = .{ .ctx = @ptrCast(&own), .ranges = ranges };
    return core.checkpoint_read_ahead.checkpoint(ck, try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i }), count, keep);
}

/// The rank's cut, in core's terms: every other name stays whole, so other GLM checkpoints prefetch as they did.
fn ranges(ctx: *anyopaque, gpa: std.mem.Allocator, name: []const u8, entry: core.safetensors.Entry, at: u64, out: *std.ArrayList([2]u64)) !void {
    const cut: *const Cut = @ptrCast(@alignCast(ctx));
    return core.exl3_load.keptRanges(cut.lo, cut.hi, name, entry, at, gpa, out);
}
