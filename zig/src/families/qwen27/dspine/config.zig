//! The 27B's DSpine drafter: core.dspine's contract pinned to tencent/Qwen3.8-27B-DSpine-b16's shape.
const std = @import("std");
const core = @import("core");
pub const Shape = core.dspine.Shape;
pub const repository = "tencent/Qwen3.8-27B-DSpine-b16";
pub const taps = [_]u32{ 1, 16, 31, 46, 61 };

/// A config.json's DSpine contract, refused unless it is the 27B drafter Metal runs.
pub fn parse(gpa: std.mem.Allocator, raw: []const u8) !Shape {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer doc.deinit();
    const s = try core.dspine.parse(doc.value);
    try pin(s);
    return s;
}

/// The one shape Metal's DSpine kernels are sized for; anything else is NotStockDSpine.
pub fn pin(s: Shape) !void {
    const ok = s.hidden == 5120 and s.intermediate == 17408 and s.layers == 5 and s.vocab == 248320 and s.heads == 32 and s.kv_heads == 8 and
        s.head_dim == 128 and s.block == 16 and s.mask == 248070 and s.window == 8192 and s.code == 1024 and s.rank == 512 and
        s.topk == core.dspine.head_topk and s.eps == 1e-6 and s.bridge_eps == 1e-6 and s.theta == 10000000 and s.target_layers == 64;
    if (!ok or !std.mem.eql(u32, s.tapIds(), &taps)) return error.NotStockDSpine;
}

/// A draft reads one target's taps and vocabulary: its width, vocabulary and depth.
pub fn bind(s: Shape, hidden: usize, vocab: usize, layers: usize) !void {
    if (hidden != s.hidden or vocab != s.vocab or layers != s.target_layers) return error.DraftForAnotherTarget;
}

pub fn qWidth(s: Shape) usize {
    return s.heads * s.head_dim;
}

pub fn kvWidth(s: Shape) usize {
    return s.kv_heads * s.head_dim;
}

pub fn tapWidth(s: Shape) usize {
    return s.hidden * s.n_taps;
}

const t = std.testing;
const fixture = @embedFile("testdata/config.json");

test "DSpine 27B config: Tencent's checkpoint pins; another width, tap or window is NotStockDSpine" {
    const s = try parse(t.allocator, fixture);
    try t.expectEqual(@as(usize, 25600), tapWidth(s));
    try bind(s, 5120, 248320, 64);
    try t.expectError(error.DraftForAnotherTarget, bind(s, 5120, 248320, 48));
    var other = s;
    other.window = 4096;
    try t.expectError(error.NotStockDSpine, pin(other));
    other = s;
    other.taps[0] = 5;
    try t.expectError(error.NotStockDSpine, pin(other));
    other = s;
    other.rank = 256;
    try t.expectError(error.NotStockDSpine, pin(other));
}
