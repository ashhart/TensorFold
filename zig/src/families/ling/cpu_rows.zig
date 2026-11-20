//! Portable BF16/F32 dense rows for the Ling CPU path; F32 GGUF tensors project through these.
const std = @import("std");
const st = @import("core").safetensors;
const rounding = @import("bf16_round");
pub const Tensor = st.Tensor;

pub fn value(t: st.Tensor, i: usize) f32 {
    return switch (t.dtype) {
        .bf16 => rounding.value(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)),
        .f32 => @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little)),
        else => unreachable,
    };
}

pub fn check(t: st.Tensor, shape: []const usize) !void {
    if (t.dtype != .bf16 and t.dtype != .f32) return error.UnsupportedCpuDType;
    if (t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return error.CpuTensorShape;
    var n: usize = t.dtype.size();
    for (shape) |d| n = try std.math.mul(usize, n, d);
    if (t.bytes.len != n) return error.CpuTensorShape;
}

pub fn round(x: f32) f32 {
    return rounding.round32(x);
}

pub fn project(t: st.Tensor, x: []const f32, y: []f32, rows: usize, width: usize, outputs: usize, rounded: bool) !void {
    try check(t, &.{ outputs, width });
    if (rows == 0 or width == 0 or outputs == 0 or x.len != try std.math.mul(usize, rows, width) or
        y.len != try std.math.mul(usize, rows, outputs)) return error.CpuRowShape;
    projectRange(t, x, y, rows, width, outputs, rounded, 0, outputs);
}

// Whole shapes and byte spans are checked by every caller before this dispatcher.
pub fn projectRange(t: st.Tensor, x: []const f32, y: []f32, rows: usize, width: usize, outputs: usize, rounded: bool, lo: usize, hi: usize) void {
    switch (t.dtype) {
        .bf16 => projectRangeTyped(u16, t.bytes.ptr, x, y, rows, width, outputs, rounded, lo, hi),
        .f32 => projectRangeTyped(u32, t.bytes.ptr, x, y, rows, width, outputs, rounded, lo, hi),
        else => unreachable,
    }
}
fn nativeLittle(comptime T: type, v: T) T {
    return if (@import("builtin").cpu.arch.endian() == .little) v else @byteSwap(v);
}
fn weights8(comptime W: type, p: [*]align(1) const W) @Vector(8, f32) {
    const loaded: @Vector(8, W) = @as(*align(1) const [8]W, @ptrCast(p)).*;
    const little = if (@import("builtin").cpu.arch.endian() == .little) loaded else @byteSwap(loaded);
    if (W == u16) {
        const widened: @Vector(8, u32) = @intCast(little);
        return @bitCast(widened << @splat(16));
    }
    return @bitCast(little);
}
fn projectRangeTyped(comptime W: type, bytes: [*]const u8, x: []const f32, y: []f32, rows: usize, width: usize, outputs: usize, rounded: bool, lo: usize, hi: usize) void {
    const weights: [*]align(1) const W = @ptrCast(bytes);
    const vector_end = width & ~@as(usize, 7);
    for (0..rows) |r| {
        const input = x.ptr + r * width;
        const output = y.ptr + r * outputs;
        for (lo..hi) |n| {
            const row = weights + n * width;
            var sum: @Vector(8, f32) = @splat(0);
            var i: usize = 0;
            // i cannot wrap: vector_end is within the validated row, rounded down to 8.
            while (i < vector_end) : (i +%= 8) {
                const w = weights8(W, row + i);
                const a: @Vector(8, f32) = @as(*align(1) const [8]f32, @ptrCast(input + i)).*;
                sum = @mulAdd(@Vector(8, f32), a, w, sum);
            }
            var acc: f32 = @reduce(.Add, sum);
            while (i < width) : (i += 1) {
                const bits = nativeLittle(W, row[i]);
                const v: f32 = if (W == u16) rounding.value(bits) else @bitCast(bits);
                acc = @mulAdd(f32, input[i], v, acc);
            }
            output[n] = if (rounded) round(acc) else acc;
        }
    }
}

/// Split output columns only: each dot retains the same reduction order at every thread count.
pub fn projectOn(t: st.Tensor, x: []const f32, y: []f32, rows: usize, width: usize, outputs: usize, rounded: bool, threads: usize) !void {
    if (threads == 0 or threads > 8) return error.CpuThreadCount;
    try check(t, &.{ outputs, width });
    if (rows == 0 or width == 0 or outputs == 0 or x.len != try std.math.mul(usize, rows, width) or y.len != try std.math.mul(usize, rows, outputs)) return error.CpuRowShape;
    const n = @min(threads, outputs);
    var workers: [7]std.Thread = undefined;
    var made: usize = 0;
    defer for (workers[0..made]) |worker| worker.join();
    for (0..n - 1) |i| {
        workers[made] = try std.Thread.spawn(.{}, projectRange, .{ t, x, y, rows, width, outputs, rounded, i * outputs / n, (i + 1) * outputs / n });
        made += 1;
    }
    projectRange(t, x, y, rows, width, outputs, rounded, (n - 1) * outputs / n, outputs);
}

test "dense rows reject malformed views and preserve nonfinite inputs" {
    const bytes = [_]u8{ 0x80, 0x3f, 0x00, 0x40 };
    const w = st.Tensor{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 2, 1, 1, 1 }, .bytes = &bytes };
    var out: [1]f32 = undefined;
    try project(w, &.{ 3, 4 }, &out, 1, 2, 1, false);
    try std.testing.expectEqual(@as(f32, 11), out[0]);
    try std.testing.expectError(error.CpuTensorShape, project(w, &.{3}, &out, 1, 1, 1, false));
    try project(w, &.{ std.math.nan(f32), 4 }, &out, 1, 2, 1, false);
    try std.testing.expect(std.math.isNan(out[0]));
}

test "unaligned BF16 vector loads preserve all encodings including signed zero and NaN payloads" {
    var bytes: [17]u8 = undefined;
    const tensor = Tensor{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 8, 1, 1, 1 }, .bytes = bytes[1..] };
    for (0..8192) |group| {
        for (0..8) |lane| std.mem.writeInt(u16, bytes[1 + lane * 2 ..][0..2], @intCast(group * 8 + lane), .little);
        const got: [8]f32 = weights8(u16, @ptrCast(bytes[1..].ptr));
        for (got, 0..) |v, lane| try std.testing.expectEqual(@as(u32, @bitCast(value(tensor, lane))), @as(u32, @bitCast(v)));
    }
}

test "unaligned F32 vector loads preserve exact payload bits" {
    const bits = [_]u32{ 0, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x7f800000, 0x7fc00123, 0x7f800001 };
    var bytes: [33]u8 = undefined;
    for (bits, 0..) |v, i| std.mem.writeInt(u32, bytes[1 + i * 4 ..][0..4], v, .little);
    const got: [8]f32 = weights8(u32, @ptrCast(bytes[1..].ptr));
    for (got, bits) |v, want| try std.testing.expectEqual(want, @as(u32, @bitCast(v)));
}
