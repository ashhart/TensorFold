//! The identical device gate runs under a host numeric instrument; planted logit and pick faults must refuse.
const std = @import("std");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const D = @import("numeric_lanes_test.zig").D;
const f = @import("device_fixture.zig");
fn read(_: *n.B, buffer: n.B.Buffer, out: []u8) !void {
    @memcpy(out, buffer[0..out.len]);
}
test "Qwen4 device gate: same capture, oracle, state seams and lane checks on the host instrument" {
    var b: n.B = .{};
    var d: D = .{};
    try @import("device_gate.zig").Gate(n.G, D, @import("core").lanes, read).runAll(std.testing.allocator, std.testing.io, &b, n.Loader{}, &d);
    try std.testing.expectEqual(@as(usize, 0), b.held);
}
test "Qwen4 device gate: one wrong logit or GPU token cannot pass" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const doc = try std.json.parseFromSlice(f.Oracle, a, f.reference, .{});
    const logits = try a.alloc(u16, doc.value.logits.len * doc.value.logits[0].len);
    var at: usize = 0;
    for (doc.value.logits) |row| for (row) |value| {
        logits[at] = @import("core").quant.convert.bf16(@floatCast(value));
        at += 1;
    };
    const picks = try a.dupe(u32, doc.value.picks);
    try f.oracle(doc.value, logits, picks);
    logits[logits.len - 1] ^= 1;
    try std.testing.expectError(error.Qwen4OracleMismatch, f.oracle(doc.value, logits, picks));
    logits[logits.len - 1] ^= 1;
    picks[picks.len - 1] ^= 1;
    try std.testing.expectError(error.Qwen4DeviceExactness, f.oracle(doc.value, logits, picks));
}
fn corrupt(_: *n.B, buffer: n.B.Buffer, out: []u8) !void {
    @memcpy(out, buffer[0..out.len]);
    out[out.len - 1] ^= 1;
}
test "Qwen4 device gate: a damaged queued capture refuses the complete gate and releases its owners" {
    var b: n.B = .{};
    var d: D = .{};
    try std.testing.expectError(error.Qwen4OracleMismatch, @import("device_gate.zig").Gate(n.G, D, @import("core").lanes, corrupt).runAll(std.testing.allocator, std.testing.io, &b, n.Loader{}, &d));
    try std.testing.expectEqual(@as(usize, 0), b.held);
}
