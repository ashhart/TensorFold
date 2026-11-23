//! Grouped projections share input sums and preserve each caller's arithmetic role.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const Proj = engine.Proj;

pub fn projectMany(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64, sums: bool, allow_fused_prompt: bool) !void {
    if (ls.len == 0 or ls.len != outs.len) return error.UnexpectedTensor;
    if (e.kit != null and !(prompt and rows > 16)) return exl3Group(e, ls, x, xs, outs, rows, prompt, params, sums);
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
    const paired = allow_fused_prompt and qwen38PairShape(ls, rows, prompt, e.affine.pdl, e.int8.on) and blk: {
        for (outs, 0..) |out, i| for (outs[i + 1 ..]) |other| if (out == other) break :blk false;
        break :blk true;
    };
    if (paired) {
        for (ls[0..2], outs[0..2], 0..) |l, out, i| try e.project(l, x, xs, out, rows, prompt, params, sums and i == 0);
        const pair = [_]A.Out{ .{ .q = ls[2].a4, .out = outs[2] }, .{ .q = ls[3].a4, .out = outs[3] } };
        return e.affine.promptMany(e.stream, x, &pair, rows);
    }
    const segmented = (e.affine.split != null or e.affine.pdl) and !(prompt and rows > 16) and !(e.affine.pdl and prompt and e.int8.on) and ls.len <= A.segs_max and for (ls) |l| {
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
    const tile = if (e.affine.pdl) e.affine.verifyRows() else A.max_rows;
    while (at < rows) : (at += tile) {
        for (ls, outs, many[0..ls.len]) |l, out, *m| m.* = .{ .q = l.a4, .out = out + at * l.outputs() * 2 };
        try e.affine.verifyMany(e.stream, x + at * k * 2, xs + at * (k / 64) * 4, many[0..ls.len], @min(tile, rows - at));
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

/// Large BF16 prompt DeltaNet inputs on the GB10's existing many tile; qkv and z keep their own launches.
fn qwen38PairShape(ls: []const Proj, rows: usize, prompt: bool, pdl: bool, int8: bool) bool {
    if (!prompt or !pdl or int8 or rows < 2048 or ls.len != 4) return false;
    for (ls, [_]usize{ 10240, 6144, 48, 48 }) |l, n| {
        if (l != .a4 or l.inputs() != 5120 or l.outputs() != n) return false;
    }
    return true;
}

test "Qwen3.8 pair shape: only large BF16 GDN prompts on GB10 admit the a and b pair" {
    const delta = [_]Proj{ affine(10240), affine(6144), affine(48), affine(48) };
    for ([_]usize{ 0, 16, 256, 2047, 2048, 4096 }) |rows| {
        for ([_]bool{ false, true }) |prompt| for ([_]bool{ false, true }) |pdl| for ([_]bool{ false, true }) |int8| {
            try std.testing.expectEqual(rows >= 2048 and prompt and pdl and !int8, qwen38PairShape(&delta, rows, prompt, pdl, int8));
        };
    }
    for (0..delta.len) |i| {
        var other = delta;
        other[i].a4.k = 4096;
        try std.testing.expect(!qwen38PairShape(&other, 2048, true, true, false));
        other = delta;
        other[i].a4.n += 1;
        try std.testing.expect(!qwen38PairShape(&other, 2048, true, true, false));
        other[i] = .{ .bf = .{ .w = 0, .n = delta[i].outputs(), .k = 5120 } };
        try std.testing.expect(!qwen38PairShape(&other, 2048, true, true, false));
    }
    try std.testing.expect(!qwen38PairShape(delta[0..3], 2048, true, true, false));
    try std.testing.expect(!qwen38PairShape(&.{ delta[0], delta[1], delta[2], delta[3], delta[3] }, 2048, true, true, false));
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

/// Decode rows of an EXL3 engine: groupable EXL3 members in one launch (each keeps its own bits), the rest one by one.
fn exl3Group(e: *Engine, ls: []const Proj, x: u64, xs: u64, outs: []const u64, rows: usize, prompt: bool, params: u64, sums: bool) !void {
    var ws: [3]cuda.exl3.Weight = undefined;
    var ys: [3]u64 = undefined;
    var n: usize = 0;
    for (ls, outs) |l, out| if (l == .x3 and n < ws.len) {
        ws[n] = l.x3;
        ys[n] = out;
        n += 1;
    };
    const grouped = n >= 2 and cuda.exl3.groupable(ws[0..n], rows);
    if (grouped) try e.kit.?.dense.decodeGroup(e.stream, ws[0..n], x, .bf16, rows, ys[0..n], .bf16);
    var first = true;
    var taken: usize = 0;
    for (ls, outs) |l, out| {
        if (grouped and l == .x3 and taken < n) {
            taken += 1;
            continue;
        }
        try e.project(l, x, xs, out, rows, prompt, params, sums and first);
        first = false;
    }
}
