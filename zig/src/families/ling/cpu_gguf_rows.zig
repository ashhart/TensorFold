//! GGUF CPU projection over the shared affine reader, with the existing eight-lane F32 sum order.
const std = @import("std");
const core = @import("core");
const g = core.gguf;
const rows = @import("cpu_rows.zig");
const q4k = @import("cpu_gguf_q4k.zig");
const q6k_q50 = @import("cpu_gguf_q6k_q50.zig");

fn supported(kind: g.TensorKind) !void {
    switch (kind) {
        .q4_k, .q6_k, .q5_0 => {},
        else => return error.UnsupportedLingGgufQuant,
    }
}
pub fn decodeBlock(kind: g.TensorKind, src: []const u8, output: []f32) !void {
    return switch (kind) {
        .q4_k => decodeKnown(.q4_k, src, output),
        .q6_k => decodeKnown(.q6_k, src, output),
        .q5_0 => decodeKnown(.q5_0, src, output),
        else => error.UnsupportedLingGgufQuant,
    };
}
fn decodeKnown(comptime kind: g.TensorKind, src: []const u8, output: []f32) !void {
    const block = comptime kind.block();
    if (src.len != block.bytes or output.len != block.values) return error.BadLingGgufBlock;
    const layout = comptime g.affine.layout(kind, kind.block().bytes) catch unreachable;
    var words: [48]u32 = undefined;
    var scales: [16]f32 = undefined;
    var biases: [16]f32 = undefined;
    try g.affine.into(kind, src, words[0..layout.words], scales[0..layout.groups], biases[0..layout.groups]);
    const bits = layout.format.bits;
    for (output, 0..) |*v, i| {
        const at = i * bits;
        const shift: u5 = @intCast(at % 32);
        var code = words[at / 32] >> shift;
        if (@as(usize, shift) + bits > 32) code |= words[at / 32 + 1] << @intCast(32 - @as(usize, shift));
        code &= (@as(u32, 1) << @intCast(bits)) - 1;
        const group = i / layout.format.group;
        const centered: i32 = @as(i32, @intCast(code)) - (if (kind == .q6_k) @as(i32, 32) else if (kind == .q5_0) @as(i32, 16) else 0);
        // Centered multiplication retains negative zero from a negative source scale.
        v.* = scales[group] * @as(f32, @floatFromInt(centered));
        if (kind == .q4_k) v.* += biases[group];
    }
}
pub fn decodeTensor(t: g.Tensor, output: []f32) !void {
    var elements: usize = 1;
    for (t.shape[0..t.rank]) |dim| elements = try std.math.mul(usize, elements, std.math.cast(usize, dim) orelse return error.BadLingGgufMatrix);
    if (elements != output.len) return error.BadLingGgufMatrix;
    if (t.kind == .f32) {
        if (t.bytes.len != try std.math.mul(usize, elements, 4)) return error.BadLingGgufMatrix;
        for (output, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little));
        return;
    }
    try supported(t.kind);
    const b = t.kind.block();
    if (elements % b.values != 0 or t.bytes.len != try std.math.mul(usize, elements / b.values, b.bytes)) return error.BadLingGgufMatrix;
    try g.affine.validate(t.kind, t.bytes);
    for (0..elements / b.values) |i| try decodeBlock(t.kind, t.bytes[i * b.bytes ..][0..b.bytes], output[i * b.values ..][0..b.values]);
}

pub fn projectReference(t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
    if (t.rank != 2 or count == 0) return error.BadLingGgufMatrix;
    const width = std.math.cast(usize, t.shape[0]) orelse return error.BadLingGgufMatrix;
    const n = std.math.cast(usize, t.shape[1]) orelse return error.BadLingGgufMatrix;
    if (input.len != try std.math.mul(usize, count, width) or output.len != try std.math.mul(usize, count, n)) return error.BadLingGgufMatrix;
    if (t.kind == .f32) return rows.project(.{ .dtype = .f32, .rank = 2, .shape = .{ n, width, 1, 1, 1 }, .bytes = t.bytes }, input, output, count, width, n, rounded);
    try supported(t.kind);
    const block = t.kind.block();
    if (width == 0 or n == 0 or width % block.values != 0 or t.bytes.len != try std.math.mul(usize, n, try std.math.mul(usize, width / block.values, block.bytes))) return error.BadLingGgufMatrix;
    try g.affine.validate(t.kind, t.bytes);
    for (0..n) |row| {
        var first: usize = 0;
        while (first < count) : (first += 8) {
            const batch = @min(8, count - first);
            var sums: [8]@Vector(8, f32) = @splat(@splat(0));
            for (0..width / block.values) |b| {
                var values: [256]f32 = undefined;
                const start = (row * (width / block.values) + b) * block.bytes;
                try decodeBlock(t.kind, t.bytes[start..][0..block.bytes], values[0..block.values]);
                for (0..batch) |r| {
                    var i: usize = 0;
                    while (i < block.values) : (i += 8) {
                        const x: @Vector(8, f32) = input[(first + r) * width + b * block.values + i ..][0..8].*;
                        const w: @Vector(8, f32) = values[i..][0..8].*;
                        sums[r] = @mulAdd(@Vector(8, f32), x, w, sums[r]);
                    }
                }
            }
            for (0..batch) |r| {
                const value = @reduce(.Add, sums[r]);
                output[(first + r) * n + row] = if (rounded) rows.round(value) else value;
            }
        }
    }
}

fn dispatchRange(kind: g.TensorKind, bytes: [*]const u8, input: [*]const f32, output: [*]f32, count: usize, width: usize, n: usize, rounded: bool, lo: usize, hi: usize) void {
    switch (kind) {
        .q4_k => q4k.projectQ4KRange(bytes, input, output, count, width, n, rounded, lo, hi),
        .q6_k => q6k_q50.projectQ6KRange(bytes, input, output, count, width, n, rounded, lo, hi),
        .q5_0 => q6k_q50.projectQ50Range(bytes, input, output, count, width, n, rounded, lo, hi),
        else => unreachable,
    }
}

/// Split output rows only: every row preserves the exact 8-lane FMA/reduction order at every thread count.
pub fn projectValidatedOn(t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool, threads: usize) !void {
    if (threads == 0 or threads > 8) return error.CpuThreadCount;
    if (t.rank != 2 or count == 0) return error.BadLingGgufMatrix;
    const width = std.math.cast(usize, t.shape[0]) orelse return error.BadLingGgufMatrix;
    const n = std.math.cast(usize, t.shape[1]) orelse return error.BadLingGgufMatrix;
    if (input.len != try std.math.mul(usize, count, width) or output.len != try std.math.mul(usize, count, n)) return error.BadLingGgufMatrix;
    if (t.kind == .f32) return rows.projectOn(.{ .dtype = .f32, .rank = 2, .shape = .{ n, width, 1, 1, 1 }, .bytes = t.bytes }, input, output, count, width, n, rounded, threads);
    try supported(t.kind);
    const block = t.kind.block();
    if (width == 0 or n == 0 or width % block.values != 0 or t.bytes.len != try std.math.mul(usize, n, try std.math.mul(usize, width / block.values, block.bytes))) return error.BadLingGgufMatrix;
    const active = @min(threads, n);
    if (active == 1) {
        dispatchRange(t.kind, t.bytes.ptr, input.ptr, output.ptr, count, width, n, rounded, 0, n);
        return;
    }
    var workers: [7]std.Thread = undefined;
    var made: usize = 0;
    defer for (workers[0..made]) |w| w.join();
    for (0..active - 1) |i| {
        workers[made] = try std.Thread.spawn(.{}, dispatchRange, .{ t.kind, t.bytes.ptr, input.ptr, output.ptr, count, width, n, rounded, i * n / active, (i + 1) * n / active });
        made += 1;
    }
    dispatchRange(t.kind, t.bytes.ptr, input.ptr, output.ptr, count, width, n, rounded, (active - 1) * n / active, n);
}

pub fn projectOn(t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool, threads: usize) !void {
    if (t.kind != .f32) {
        try supported(t.kind);
        if (t.rank != 2 or count == 0) return error.BadLingGgufMatrix;
        const width = std.math.cast(usize, t.shape[0]) orelse return error.BadLingGgufMatrix;
        const n = std.math.cast(usize, t.shape[1]) orelse return error.BadLingGgufMatrix;
        const block = t.kind.block();
        if (width == 0 or n == 0 or width % block.values != 0 or t.bytes.len != try std.math.mul(usize, n, try std.math.mul(usize, width / block.values, block.bytes))) return error.BadLingGgufMatrix;
        try g.affine.validate(t.kind, t.bytes);
    }
    try projectValidatedOn(t, input, output, count, rounded, threads);
}

pub fn project(t: g.Tensor, input: []const f32, output: []f32, count: usize, rounded: bool) !void {
    return projectOn(t, input, output, count, rounded, 1);
}
