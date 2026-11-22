//! The attention projection stage writes K/V interleaved; the consumers read that layout directly.
pub fn project(k: anytype, e: anytype, weights: anytype, f: anytype, sums: anytype, rows: u32) !void {
    try k.quant(e, weights.q, f.get(.input), sums, f.get(.dims), f.get(.q_gate), rows);
    try k.quant(e, weights.kv, f.get(.input), sums, f.get(.dims), f.get(.mixed), rows);
    e.barrier();
}

/// The same stage with main's prompt-tail Q projection; K/V still cover every row in the mixed buffer.
pub fn projectTail(k: anytype, e: anytype, weights: anytype, f: anytype, sums: anytype, rows: u32, from: u32) !void {
    if (from == 0) return project(k, e, weights, f, sums, rows);
    const Ref = @import("projection.zig").Ref;
    var q_sums = sums;
    q_sums.ref.offset += @as(usize, from) * 4;
    const q_in: Ref = .{ .buffer = f.get(.input).buffer, .offset = f.get(.input).offset + @as(usize, from) * weights.q.k * 2 };
    const q_out: Ref = .{ .buffer = f.get(.q_gate).buffer, .offset = f.get(.q_gate).offset + @as(usize, from) * weights.q.n * 2 };
    try k.quant(e, weights.q, q_in, q_sums, f.get(.dims), q_out, rows - from);
    try k.quant(e, weights.kv, f.get(.input), sums, f.get(.dims), f.get(.mixed), rows);
    e.barrier();
}

const std = @import("std");
const Field = enum { input, dims, q_gate, mixed, key, value };
const Recording = struct {
    calls: [6][]const u8 = undefined,
    n: usize = 0,
    fn add(r: *Recording, name: []const u8) void {
        r.calls[r.n] = name;
        r.n += 1;
    }
    pub fn quant(r: *Recording, _: *@This(), weight: Field, _: Field, _: ?Field, _: Field, out: Field, _: u32) !void {
        try std.testing.expectEqual(weight, out);
        r.add(if (weight == .q_gate) "q" else "kv");
    }
    pub fn unstack(r: *Recording, _: *@This(), _: Field, _: Field) !void {
        r.add("q27_slice");
    }
    pub fn barrier(r: *Recording) void {
        r.add("barrier");
    }
    pub fn get(_: *const Recording, f: Field) Field {
        return f;
    }
};
test "qwen27 mixed: attentionForward projection stage has no q27_slice" {
    for ([_]u32{ 1, 16, 2048 }) |rows| {
        var record = Recording{};
        try project(&record, &record, .{ .q = Field.q_gate, .kv = Field.mixed }, &record, @as(?Field, null), rows);
        try std.testing.expectEqual(@as(usize, 3), record.n);
        for (record.calls[0..record.n], [_][]const u8{ "q", "kv", "barrier" }) |got, want|
            try std.testing.expectEqualStrings(want, got);
    }
}
