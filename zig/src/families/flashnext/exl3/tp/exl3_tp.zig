//! Two-rank cuts for the Qwen3.8-Flash-Next EXL3 3.05 pack.

const std = @import("std");
const shape = @import("../pack/exl3_shape.zig");

pub const Axis = enum { replicate, split_k, split_n, expert_parallel };

pub const Role = enum { gate, up, down, q, k, v, o, head, shared, routed_set, hc_lowrank };

pub const Cut = struct {
    role: Role,
    axis: Axis,
    k: usize,
    n: usize,
    part: usize,
};

pub const Span = struct { lo: usize, hi: usize };

/// True when `axis_len` can be cut into `ranks` EXL3 pieces.
pub fn legal(axis_len: usize, ranks: usize) bool {
    if (ranks == 0) return false;
    return axis_len % (shape.hadamard * ranks) == 0;
}

/// The 3.05 cut for `role` on `rank`. Ranks are 1 (whole tensor) or 2.
pub fn cut(role: Role, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!Cut {
    if (ranks != 1 and ranks != 2) return error.IllegalCut;
    if (rank >= ranks) return error.RankOutOfRange;
    if (ranks == 1) return unsplit(role);
    return switch (role) {
        .gate, .up => .{ .role = role, .axis = .split_k, .k = shape.hidden, .n = shape.expert_intermediate, .part = shape.hidden / 2 },
        .down => .{ .role = role, .axis = .split_n, .k = shape.expert_intermediate, .n = shape.hidden, .part = shape.hidden / 2 },
        .shared => .{ .role = role, .axis = .replicate, .k = shape.hidden, .n = shape.expert_intermediate, .part = 0 },
        .routed_set => .{ .role = role, .axis = .expert_parallel, .k = 0, .n = 0, .part = 0 },
        .q => .{ .role = role, .axis = .split_n, .k = shape.hidden, .n = shape.qOut(), .part = shape.qOut() / 2 },
        .k, .v => .{ .role = role, .axis = .split_n, .k = shape.hidden, .n = shape.kvOut(), .part = shape.kvOut() / 2 },
        // o keeps the heads this rank produced in q, so the split is along K.
        .o => .{ .role = role, .axis = .split_k, .k = shape.qOut(), .n = shape.hidden, .part = shape.qOut() / 2 },
        .head => .{ .role = role, .axis = .split_n, .k = shape.hidden, .n = shape.vocab, .part = shape.vocab / 2 },
        // Unquantized. The 320-wide low-rank axis is split in half. 128 alignment does not apply.
        .hc_lowrank => .{ .role = role, .axis = .split_n, .k = shape.hc_lowrank, .n = shape.hidden, .part = shape.hc_lowrank / 2 },
    };
}

fn unsplit(role: Role) Cut {
    return switch (role) {
        .gate, .up, .shared => .{ .role = role, .axis = .replicate, .k = shape.hidden, .n = shape.expert_intermediate, .part = shape.hidden },
        .down => .{ .role = role, .axis = .replicate, .k = shape.expert_intermediate, .n = shape.hidden, .part = shape.hidden },
        .routed_set => .{ .role = role, .axis = .expert_parallel, .k = 0, .n = 0, .part = 0 },
        .q => .{ .role = role, .axis = .replicate, .k = shape.hidden, .n = shape.qOut(), .part = shape.qOut() },
        .k, .v => .{ .role = role, .axis = .replicate, .k = shape.hidden, .n = shape.kvOut(), .part = shape.kvOut() },
        .o => .{ .role = role, .axis = .replicate, .k = shape.qOut(), .n = shape.hidden, .part = shape.qOut() },
        .head => .{ .role = role, .axis = .replicate, .k = shape.hidden, .n = shape.vocab, .part = shape.vocab },
        .hc_lowrank => .{ .role = role, .axis = .replicate, .k = shape.hc_lowrank, .n = shape.hidden, .part = shape.hc_lowrank },
    };
}

/// `[lo, hi)` on the split axis. Expert-parallel spans are expert ids.
pub fn range(role: Role, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!Span {
    const c = try cut(role, rank, ranks);
    if (role == .routed_set) {
        if (shape.experts % ranks != 0) return error.IllegalCut;
        const part = shape.experts / ranks;
        return .{ .lo = rank * part, .hi = (rank + 1) * part };
    }
    const along: usize = switch (role) {
        .gate, .up, .o => c.k,
        .hc_lowrank => c.k,
        else => c.n,
    };
    if (c.axis == .replicate or ranks == 1) return .{ .lo = 0, .hi = along };
    const part = along / ranks;
    return .{ .lo = rank * part, .hi = (rank + 1) * part };
}

/// True when an EXL3 N-split at `ranks` would break a Hadamard block.
pub fn refusesColumnSplit(k: usize, n: usize, ranks: usize) bool {
    _ = k;
    return !legal(n, ranks);
}

/// `axis_len` is this tensor's header axis, not a class total. `hc_lowrank` is not a Hadamard cut.
pub fn share(role: Role, span: u64, axis_len: usize, rank: usize, ranks: usize) error{ IllegalCut, RankOutOfRange }!u64 {
    const c = try cut(role, rank, ranks);
    if (span == 0 or c.axis == .replicate or c.axis == .expert_parallel) return span;
    if (role == .hc_lowrank) {
        if (axis_len == 0 or axis_len % ranks != 0) return span;
        return span / @as(u64, ranks);
    }
    if (axis_len == 0 or !legal(axis_len, ranks)) return span;
    return span / @as(u64, ranks);
}

test "gate is K-split and down is N-split" {
    const gate = try cut(.gate, 0, 2);
    try std.testing.expectEqual(Axis.split_k, gate.axis);
    try std.testing.expectEqual(@as(usize, 1280), gate.part);
    try std.testing.expect(refusesColumnSplit(2560, 640, 2));
    try std.testing.expect(!legal(640, 2));
    try std.testing.expect(legal(2560, 2));
    try std.testing.expect(!legal(320, 2));

    const down = try cut(.down, 1, 2);
    try std.testing.expectEqual(Axis.split_n, down.axis);
    try std.testing.expect(down.axis != .split_k);
    const span = try range(.down, 1, 2);
    try std.testing.expectEqual(@as(usize, 1280), span.lo);
    try std.testing.expectEqual(@as(usize, 2560), span.hi);
}

test "expert ids abut and q, k, hc take their parts" {
    const a = try range(.routed_set, 0, 2);
    const b = try range(.routed_set, 1, 2);
    try std.testing.expectEqual(@as(usize, 0), a.lo);
    try std.testing.expectEqual(@as(usize, 256), a.hi);
    try std.testing.expectEqual(a.hi, b.lo);
    try std.testing.expectEqual(@as(usize, 512), b.hi);

    const q = try cut(.q, 0, 2);
    try std.testing.expectEqual(@as(usize, 3072), q.part);
    try std.testing.expect(q.part % 128 == 0);
    try std.testing.expectEqual(@as(usize, 256), (try cut(.k, 0, 2)).part);
    try std.testing.expectEqual(@as(usize, 160), (try cut(.hc_lowrank, 0, 2)).part);
    try std.testing.expect((try cut(.hc_lowrank, 0, 2)).part % 128 != 0);
    try std.testing.expectError(error.RankOutOfRange, cut(.gate, 2, 2));
}

test "one rank keeps the whole tensor and both ranks of q stay 128-aligned" {
    try std.testing.expectEqual(Axis.replicate, (try cut(.gate, 0, 1)).axis);
    try std.testing.expectEqual(Axis.expert_parallel, (try cut(.routed_set, 0, 1)).axis);
    const all = try range(.routed_set, 0, 1);
    try std.testing.expectEqual(@as(usize, 0), all.lo);
    try std.testing.expectEqual(@as(usize, 512), all.hi);
    try std.testing.expectEqual(@as(usize, 124160), (try cut(.head, 0, 2)).part);
    try std.testing.expect((try cut(.head, 0, 2)).part % 128 == 0);
}

test "share follows the header axis and hc_lowrank still halves 320" {
    try std.testing.expectEqual(@as(u64, 500), try share(.q, 1000, shape.qOut(), 0, 2));
    try std.testing.expectEqual(@as(u64, 1000), try share(.q, 1000, 100, 0, 2));
    try std.testing.expect(!legal(shape.hc_lowrank, 2));
    try std.testing.expectEqual(@as(u64, 160), try share(.hc_lowrank, 320, shape.hc_lowrank, 1, 2));
    try std.testing.expectEqual(@as(u64, 640), try share(.shared, 640, shape.expert_intermediate, 0, 2));
    try std.testing.expectEqual(@as(u64, 1000), try share(.q, 1000, shape.qOut(), 0, 1));
    try std.testing.expectError(error.IllegalCut, share(.q, 1000, shape.qOut(), 0, 0));
    try std.testing.expectError(error.IllegalCut, share(.gate, 1000, shape.hidden, 0, 4));
}
