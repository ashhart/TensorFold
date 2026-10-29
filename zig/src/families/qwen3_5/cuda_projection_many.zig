//! Grouped projections share input sums and preserve each caller's arithmetic role.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Proj = engine.Proj;

pub fn projectMany(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64, sums: bool, allow_fused_prompt: bool) !void {
    if (ls.len == 0 or ls.len != outs.len) return error.UnexpectedTensor;
    const A = cuda.qlinear.Affine4Kernels;
    const fused_prompt = allow_fused_prompt and prompt and rows > A.max_rows and qwen38PromptShape(ls, rows) and !e.int8.on and e.affine.pdl and blk: {
        const k = ls[0].inputs();
        for (ls) |l| if (l != .a4 or l.inputs() != k) break :blk false;
        for (outs, 0..) |out, i| for (outs[i + 1 ..]) |other| if (out == other) break :blk false;
        break :blk true;
    };
    if (fused_prompt) {
        var many: [A.segs_max]A.Out = undefined;
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out };
        return e.affine.promptMany(e.stream, x, many[0..ls.len], rows);
    }
    const segmented = e.affine.split != null and !(prompt and rows > 16) and ls.len <= A.segs_max and for (ls) |l| {
        if (l != .a4 or l.inputs() != ls[0].inputs()) break false;
    } else true;
    if (!segmented) {
        for (ls, outs, 0..) |l, out, i| try e.project(l, x, xs, out, rows, prompt, params, sums and i == 0);
        return;
    }
    const k = ls[0].inputs();
    if (sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    var many: [A.segs_max]A.Out = undefined;
    var at: usize = 0;
    while (at < rows) : (at += A.max_rows) {
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out + at * l.outputs() * 2 };
        try e.affine.decodeMany(e.stream, x + at * k * 2, xs + at * (k / 64) * 4, many[0..ls.len], @min(A.max_rows, rows - at));
    }
}

/// Fuses only Qwen3.8 short QKV and Delta shapes that passed same-stack A-B-A screening on GB10.
fn qwen38PromptShape(ls: []const Proj, rows: usize) bool {
    if (rows > 256 or (ls.len != 3 and ls.len != 4)) return false;
    for (ls) |l| if (l != .a4 or l.inputs() != 5120) return false;
    if (ls.len == 3)
        return ls[0].outputs() == 12288 and ls[1].outputs() == 1024 and ls[2].outputs() == 1024;
    return ls[0].outputs() == 10240 and ls[1].outputs() == 6144 and ls[2].outputs() == 48 and ls[3].outputs() == 48;
}

pub fn qwen38LargePromptShape(n: usize, k: usize, rows: usize) bool {
    if (rows < 2048 or k != 5120) return false;
    return switch (n) {
        48, 1024, 6144, 10240, 12288, 17408 => true,
        else => false,
    };
}

test "Qwen3.8 fused prompt admits only screened BF16 projection shapes" {
    const qkv = [_]Proj{ affine(12288), affine(1024), affine(1024) };
    const delta = [_]Proj{ affine(10240), affine(6144), affine(48), affine(48) };
    const mlp = [_]Proj{ affine(17408), affine(17408) };
    try std.testing.expect(qwen38PromptShape(&qkv, 256));
    try std.testing.expect(qwen38PromptShape(&delta, 17));
    try std.testing.expect(!qwen38PromptShape(&qkv, 257));
    try std.testing.expect(!qwen38PromptShape(&mlp, 128));
}

test "Qwen3.8 GB10 large prompt tile admits only screened projection shapes" {
    try std.testing.expect(qwen38LargePromptShape(12288, 5120, 2048));
    try std.testing.expect(qwen38LargePromptShape(17408, 5120, 4096));
    try std.testing.expect(!qwen38LargePromptShape(12288, 5120, 1024));
    try std.testing.expect(!qwen38LargePromptShape(5120, 17408, 2048));
}

fn affine(n: usize) Proj {
    return .{ .a4 = .{ .w = 0, .s = 0, .b = 0, .n = n, .k = 5120, .npad = n } };
}
