//! A row's token from its logits: argmax for greedy, else the keyed draw over top_k + MARGIN candidates.

const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// exact_sampling.MARGIN: candidates past top_k, so ties resolve by id on the host.
pub const MARGIN = 8;

pub const Dtype = enum { f16, bf16 };

pub fn widen(row: []const u16, dtype: Dtype, i: usize) f32 {
    return switch (dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(row[i]))),
        .bf16 => @bitCast(@as(u32, row[i]) << 16),
    };
}

/// torch.argmax over the widened row: the first index of the largest value (a NaN wins, as torch's).
pub fn argmax(row: []const u16, dtype: Dtype) u32 {
    var best: usize = 0;
    var top = widen(row, dtype, 0);
    for (1..row.len) |i| {
        const v = widen(row, dtype, i);
        if (std.math.isNan(top)) break;
        if (std.math.isNan(v) or v > top) {
            best = i;
            top = v;
        }
    }
    return @intCast(best);
}

const Candidate = struct { value: f32, id: u32 };

fn larger(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// draw(): greedy (no sampling or temperature <= 0) is the argmax; else `choose` at `position` over the candidates.
pub fn draw(gpa: Allocator, row: []const u16, dtype: Dtype, sampling: ?lanes.Sampling, position: u64) !u32 {
    const s = sampling orelse return argmax(row, dtype);
    if (s.temperature <= 0.0) return argmax(row, dtype);
    const width = row.len;
    const count = if (s.top_k != 0) @min(width, @as(usize, s.top_k) + MARGIN) else width;
    const best = try gpa.alloc(Candidate, count);
    defer gpa.free(best);
    var filled: usize = 0;
    // the whole row when top_k is 0 (choose orders it), else the top `count` by value then id in one pass
    if (count == width) {
        for (best, 0..) |*c, i| c.* = .{ .value = widen(row, dtype, i), .id = @intCast(i) };
        filled = width;
    } else for (0..width) |i| {
        const c: Candidate = .{ .value = widen(row, dtype, i), .id = @intCast(i) };
        if (filled == count and !larger({}, c, best[count - 1])) continue;
        var at = if (filled < count) filled else count - 1;
        if (filled < count) filled += 1;
        while (at > 0 and larger({}, c, best[at - 1])) : (at -= 1) best[at] = best[at - 1];
        best[at] = c;
    }
    const values = try gpa.alloc(f64, count);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, count);
    defer gpa.free(ids);
    for (best, values, ids) |c, *v, *id| {
        v.* = c.value;
        id.* = c.id;
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, s));
}

/// The softmax probability of `id` in the widened row (the MTP chain's confidence cut), in double.
pub fn probability(row: []const u16, dtype: Dtype, id: u32) f64 {
    var top: f64 = -std.math.inf(f64);
    for (0..row.len) |i| top = @max(top, widen(row, dtype, i));
    var total: f64 = 0;
    for (0..row.len) |i| total += @exp(@as(f64, widen(row, dtype, i)) - top);
    return @exp(@as(f64, widen(row, dtype, id)) - top) / total;
}

test "argmax keeps the first of equal values; bf16 and fp16 widen exactly" {
    const one_bf16: u16 = 0x3f80;
    const row = [_]u16{ 0, one_bf16, one_bf16, 0 };
    try std.testing.expectEqual(@as(u32, 1), argmax(&row, .bf16));
    const one_f16: u16 = 0x3c00;
    try std.testing.expectEqual(@as(f32, 1.0), widen(&.{one_f16}, .f16, 0));
}

test "greedy draws the argmax" {
    const row = [_]u16{ 0x3c00, 0x4000, 0x3e00 };
    try std.testing.expectEqual(@as(u32, 1), try draw(std.testing.allocator, &row, .f16, null, 0));
}
