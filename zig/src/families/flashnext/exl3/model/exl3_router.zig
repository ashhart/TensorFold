//! Rank 0 of the Flash Next router.

const std = @import("std");
const route = @import("exl3_route.zig");

pub const gate_rows: usize = 512;
pub const gate_cols: usize = 2560;
pub const gate_bytes: usize = gate_rows * gate_cols * @sizeOf(u16);

pub const Error = error{
    NotRank0,
    EmptyGate,
    GateBytes,
    MissingSharedGate,
    Length,
};

comptime {
    std.debug.assert(gate_bytes == 2_621_440);
    std.debug.assert((gate_rows + 1) * gate_cols * @sizeOf(u16) == 2_626_560);
}

/// Storage type of the two replicated gate tensors.
pub const Dtype = enum { bf16, f16 };

fn dot(dtype: Dtype, weight: []const u8, x: []const f32) f32 {
    var sum: f32 = 0;
    for (x, 0..) |v, i| {
        const h = std.mem.readInt(u16, weight[i * 2 ..][0..2], .little);
        const w: f32 = switch (dtype) {
            .bf16 => @bitCast(@as(u32, h) << 16),
            .f16 => @floatCast(@as(f16, @bitCast(h))),
        };
        sum += w * v;
    }
    return sum;
}

/// `gate` is row-major bf16, exactly `gate_bytes`, times `hidden` (`gate_cols`).
pub fn launch(
    rank: usize,
    gate: []const u8,
    hidden: []const f32,
    shared_gate: []const u8,
    shared_x: []const f32,
    topk: usize,
    logits: []f32,
    picks: []u16,
    weights: []f32,
    scratch: []f32,
) Error!void {
    return launchDtype(.bf16, rank, gate, hidden, shared_gate, shared_x, topk, logits, picks, weights, scratch);
}

/// `launch` with the gate tensors' storage type named.
pub fn launchDtype(
    dtype: Dtype,
    rank: usize,
    gate: []const u8,
    hidden: []const f32,
    shared_gate: []const u8,
    shared_x: []const f32,
    topk: usize,
    logits: []f32,
    picks: []u16,
    weights: []f32,
    scratch: []f32,
) Error!void {
    if (rank != 0) return error.NotRank0;
    if (gate.len == 0) return error.EmptyGate;
    if (gate.len != gate_bytes) return error.GateBytes;
    if (shared_gate.len == 0) return error.MissingSharedGate;
    if (shared_gate.len % 2 != 0) return error.Length;
    const shared_n = shared_gate.len / 2;
    if (shared_x.len != shared_n) return error.Length;
    if (hidden.len != gate_cols) return error.Length;
    if (logits.len != gate_rows + 1) return error.Length;
    if (topk == 0 or topk > gate_rows) return error.Length;
    if (picks.len != topk + 1 or weights.len != topk + 1 or scratch.len != gate_rows) return error.Length;

    const row_bytes = gate_cols * @sizeOf(u16);
    for (0..gate_rows) |r| {
        const row = gate[r * row_bytes ..][0..row_bytes];
        logits[r] = dot(dtype, row, hidden);
    }
    logits[gate_rows] = dot(dtype, shared_gate, shared_x);
    try route.select(logits, topk, picks, weights, scratch);
}

const one_bf16: u16 = 0x3f80;
const winner: usize = 42;
const shared_span: usize = 3;

fn paint(gate: []u8) void {
    @memset(gate, 0);
    const at = winner * gate_cols * @sizeOf(u16);
    std.mem.writeInt(u16, gate[at..][0..2], one_bf16, .little);
}

fn sharedGate() [shared_span * 2]u8 {
    var bytes: [shared_span * 2]u8 = @splat(0);
    for (0..shared_span) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], one_bf16, .little);
    return bytes;
}

fn hiddenVec() [gate_cols]f32 {
    var hidden: [gate_cols]f32 = @splat(0);
    hidden[0] = 1;
    return hidden;
}

fn expectUntouched(logits: []const f32, picks: []const u16, weights: []const f32) !void {
    try std.testing.expectEqual(@as(f32, 8), logits[0]);
    try std.testing.expectEqual(@as(f32, 8), logits[gate_rows]);
    try std.testing.expectEqualSlices(u16, &.{ 9, 9 }, picks);
    try std.testing.expectEqualSlices(f32, &.{ 3, 3 }, weights);
}

test "one nonzero gate row and a shared span of 3 select id 512" {
    try std.testing.expect(shared_span != gate_cols);
    try std.testing.expect(shared_span != 640);
    try std.testing.expect(shared_span != 1280);
    try std.testing.expect(shared_span != 61_440);
    try std.testing.expectEqual(@as(usize, 2_621_440), gate_bytes);

    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, gate_bytes);
    defer gpa.free(gate);
    paint(gate);

    const hidden = hiddenVec();
    const shared_gate = sharedGate();
    const shared_x = [_]f32{ 0, 5, 4 };
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = undefined;
    try launch(0, gate, &hidden, &shared_gate, &shared_x, 1, &logits, &picks, &weights, &scratch);

    try std.testing.expectEqual(@as(f32, 0), logits[0]);
    try std.testing.expectEqual(@as(f32, 1), logits[winner]);
    try std.testing.expectEqual(@as(f32, 0), logits[gate_rows - 1]);
    try std.testing.expectEqual(@as(f32, 9), logits[gate_rows]);
    try std.testing.expectEqual(@as(u16, winner), picks[0]);
    try std.testing.expectEqual(@as(u16, 512), picks[1]);
    try std.testing.expectEqual(@as(f32, 1), weights[0]);
}

test "rank 1 leaves the pick buffer untouched" {
    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, gate_bytes);
    defer gpa.free(gate);
    paint(gate);

    const hidden = hiddenVec();
    const shared_gate = sharedGate();
    const shared_x = [_]f32{ 0, 5, 4 };
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = @splat(4);
    try std.testing.expectError(error.NotRank0, launch(1, gate, &hidden, &shared_gate, &shared_x, 1, &logits, &picks, &weights, &scratch));
    try expectUntouched(&logits, &picks, &weights);
    try std.testing.expectEqual(@as(f32, 4), scratch[0]);
}

test "a missing shared gate does not call select" {
    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, gate_bytes);
    defer gpa.free(gate);
    paint(gate);

    const hidden = hiddenVec();
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = @splat(4);
    try std.testing.expectError(error.MissingSharedGate, launch(0, gate, &hidden, &.{}, &.{}, 1, &logits, &picks, &weights, &scratch));
    try expectUntouched(&logits, &picks, &weights);
    try std.testing.expectEqual(@as(f32, 4), scratch[0]);
}

test "a 513-row gate buffer is refused" {
    const gpa = std.testing.allocator;
    const wide_bytes = (gate_rows + 1) * gate_cols * @sizeOf(u16);
    try std.testing.expectEqual(@as(usize, 2_626_560), wide_bytes);
    const gate = try gpa.alloc(u8, wide_bytes);
    defer gpa.free(gate);

    const hidden = hiddenVec();
    const shared_gate = sharedGate();
    const shared_x = [_]f32{ 0, 5, 4 };
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = @splat(4);
    try std.testing.expectError(error.GateBytes, launch(0, gate, &hidden, &shared_gate, &shared_x, 1, &logits, &picks, &weights, &scratch));
    try expectUntouched(&logits, &picks, &weights);
}

test "an empty gate does not launch" {
    const hidden = hiddenVec();
    const shared_gate = sharedGate();
    const shared_x = [_]f32{ 0, 5, 4 };
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = @splat(4);
    try std.testing.expectError(error.EmptyGate, launch(0, &.{}, &hidden, &shared_gate, &shared_x, 1, &logits, &picks, &weights, &scratch));
    try expectUntouched(&logits, &picks, &weights);
    try std.testing.expectEqual(@as(f32, 4), scratch[0]);
}

test "a shared-span mismatch stores nothing" {
    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, gate_bytes);
    defer gpa.free(gate);
    @memset(gate, 0);

    const hidden = hiddenVec();
    const shared_gate = sharedGate();
    const short_x = [_]f32{1};
    var logits: [gate_rows + 1]f32 = @splat(8);
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 3, 3 };
    var scratch: [gate_rows]f32 = @splat(4);
    try std.testing.expectError(error.Length, launch(0, gate, &hidden, &shared_gate, &short_x, 1, &logits, &picks, &weights, &scratch));
    try expectUntouched(&logits, &picks, &weights);
}

test "the same bits read as F16 are not the bf16 value" {
    const gpa = std.testing.allocator;
    const gate = try gpa.alloc(u8, gate_bytes);
    defer gpa.free(gate);
    @memset(gate, 0);
    const one_f16: u16 = 0x3c00;
    std.mem.writeInt(u16, gate[7 * gate_cols * 2 ..][0..2], one_f16, .little);
    var shared: [gate_cols * 2]u8 = @splat(0);
    std.mem.writeInt(u16, shared[0..2], one_f16, .little);
    var hidden: [gate_cols]f32 = @splat(0);
    hidden[0] = 1;
    var logits: [gate_rows + 1]f32 = undefined;
    var picks = [_]u16{ 9, 9 };
    var weights = [_]f32{ 0, 0 };
    var scratch: [gate_rows]f32 = undefined;
    try launchDtype(.f16, 0, gate, &hidden, &shared, &hidden, 1, &logits, &picks, &weights, &scratch);
    try std.testing.expectEqual(@as(f32, 1), logits[7]);
    try std.testing.expectEqual(@as(f32, 1), logits[gate_rows]);
    try std.testing.expectEqual(@as(u16, 7), picks[0]);
    try launchDtype(.bf16, 0, gate, &hidden, &shared, &hidden, 1, &logits, &picks, &weights, &scratch);
    try std.testing.expect(logits[7] != 1);
}
