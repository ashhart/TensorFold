//! The attention projection stage writes K/V interleaved; the consumers read that layout directly.
pub fn project(k: anytype, e: anytype, weights: anytype, f: anytype, sums: anytype, rows: u32, t0: u32) !void {
    var q_sums = sums;
    switch (@typeInfo(@TypeOf(sums))) {
        .optional => if (q_sums) |*s| s.offset += @as(usize, t0) * 4,
        else => q_sums.ref.offset += @as(usize, t0) * 4,
    }
    var q_in: @TypeOf(f.get(.input)) = f.get(.input);
    q_in.offset += @as(usize, t0) * weights.q.k * 2;
    var q_out: @TypeOf(f.get(.q_gate)) = f.get(.q_gate);
    q_out.offset += @as(usize, t0) * weights.q.n * 2;
    try k.quant(e, weights.q, q_in, q_sums, f.get(.dims), q_out, rows - t0);
    try k.quant(e, weights.kv, f.get(.input), sums, f.get(.dims), f.get(.mixed), rows);
    e.barrier();
}

const std = @import("std");
const Ref = struct { field: Field, offset: usize = 0 };
const Weight = struct { field: Field, k: usize = 1, n: usize = 1 };
const Recording = struct {
    calls: [6][]const u8 = undefined,
    n: usize = 0,
    fn add(r: *Recording, name: []const u8) void {
        r.calls[r.n] = name;
        r.n += 1;
    }
    pub fn quant(r: *Recording, _: *@This(), weight: Weight, _: Ref, _: ?Ref, _: Ref, out: Ref, _: u32) !void {
        try std.testing.expectEqual(weight.field, out.field);
        r.add(if (weight.field == .q_gate) "q" else "kv");
    }
    pub fn unstack(r: *Recording, _: *@This(), _: Field, _: Field) !void {
        r.add("q27_slice");
    }
    pub fn barrier(r: *Recording) void {
        r.add("barrier");
    }
    pub fn get(_: *const Recording, f: Field) Ref {
        return .{ .field = f };
    }
};
test "qwen27 mixed: attentionForward projection stage has no q27_slice, including the prompt tail" {
    for ([_]u32{ 1, 16, 2048 }) |rows| for ([_]u32{ 0, 1 }) |q_from| {
        var record = Recording{};
        try project(&record, &record, .{ .q = Weight{ .field = .q_gate }, .kv = Weight{ .field = .mixed } }, &record, @as(?Ref, .{ .field = .sums }), rows, q_from);
        try std.testing.expectEqual(@as(usize, 3), record.n);
        for (record.calls[0..record.n], [_][]const u8{ "q", "kv", "barrier" }) |got, want|
            try std.testing.expectEqualStrings(want, got);
    };
}
