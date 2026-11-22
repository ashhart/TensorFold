//! The attention projection stage writes K/V interleaved; the consumers read that layout directly.
pub fn project(k: anytype, e: anytype, weights: anytype, f: anytype, sums: anytype, rows: u32, t0: u32) !void {
    var q_sums = sums;
    q_sums.ref.offset += @as(usize, t0) * 4;
    const q_in = .{ .buffer = f.get(.input).buffer, .offset = f.get(.input).offset + @as(usize, t0) * weights.q.k * 2 };
    const q_out = .{ .buffer = f.get(.q_gate).buffer, .offset = f.get(.q_gate).offset + @as(usize, t0) * weights.q.n * 2 };
    try k.quant(e, weights.q, q_in, q_sums, f.get(.dims), q_out, rows - t0);
    try k.quant(e, weights.kv, f.get(.input), sums, f.get(.dims), f.get(.mixed), rows);
    e.barrier();
}

const std = @import("std");
const Field = enum { input, dims, q_gate, mixed, key, value, sums };
const Ref = struct { buffer: Field, offset: usize = 0 };
const Sums = struct { ref: Ref, stride: u32 = 0 };
const Weight = struct { n: u32 = 1, k: u32 = 1 };
const Recording = struct {
    calls: [6][]const u8 = undefined,
    n: usize = 0,
    fn add(r: *Recording, name: []const u8) void {
        r.calls[r.n] = name;
        r.n += 1;
    }
    pub fn quant(r: *Recording, _: *@This(), weight: anytype, _: Ref, _: Sums, _: Ref, out: Ref, _: u32) !void {
        _ = weight;
        r.add(if (out.buffer == .q_gate) "q" else "kv");
    }
    pub fn unstack(r: *Recording, _: *@This(), _: Field, _: Field) !void {
        r.add("q27_slice");
    }
    pub fn barrier(r: *Recording) void {
        r.add("barrier");
    }
    pub fn get(_: *const Recording, f: Field) Ref {
        return .{ .buffer = f };
    }
};
test "qwen27 mixed: attentionForward projection stage has no q27_slice" {
    for ([_]u32{ 1, 16, 2048 }) |rows| {
        var record = Recording{};
        try project(&record, &record, .{ .q = Weight{}, .kv = Weight{} }, &record, Sums{ .ref = .{ .buffer = .sums } }, rows, 0);
        try std.testing.expectEqual(@as(usize, 3), record.n);
        for (record.calls[0..record.n], [_][]const u8{ "q", "kv", "barrier" }) |got, want|
            try std.testing.expectEqualStrings(want, got);
    }
}
