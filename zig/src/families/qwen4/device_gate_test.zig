//! The identical device gate runs under a host numeric instrument; planted logit and pick faults must refuse.
const std = @import("std");
const n = @import("../../../tests/qwen4/numeric_gpu.zig");
const D = @import("numeric_lanes_test.zig").D;
const f = @import("device_fixture.zig");
fn read(_: *n.B, buffer: n.B.Buffer, out: []u8) !void {
    @memcpy(out, buffer[0..out.len]);
}
/// Sets what the gate prints (device_fixture.say) and returns what it was.
fn swap(to: @TypeOf(f.say)) @TypeOf(f.say) {
    defer f.say = to;
    return f.say;
}
test "Qwen4 device gate: same capture, oracle, state seams and lane checks on the host instrument" {
    const was = swap(.failures);
    defer f.say = was;
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
    const was = swap(.nothing);
    defer f.say = was;
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
    const was = swap(.nothing);
    defer f.say = was;
    var b: n.B = .{};
    var d: D = .{};
    try std.testing.expectError(error.Qwen4OracleMismatch, @import("device_gate.zig").Gate(n.G, D, @import("core").lanes, corrupt).runAll(std.testing.allocator, std.testing.io, &b, n.Loader{}, &d));
    try std.testing.expectEqual(@as(usize, 0), b.held);
}

test "Qwen4 device gate: Metal probability nodes retain exact binary64 logit comparison" {
    const was = swap(.failures);
    defer f.say = was;
    var b: n.B = .{};
    var d: D = .{};
    try @import("device_gate.zig").Gate(n.G, D, @import("core").lanes, read).runAll(std.testing.allocator, std.testing.io, &b, n.MetalLoader{}, &d);
    try std.testing.expectEqual(@as(usize, 0), b.held);
}

test "Qwen4 device gate: two-row node ledger covers every binary64 storage node" {
    var b: n.B = .{};
    const gate = @import("node_gate.zig");
    const summary = try gate.run(n.G, read, std.testing.allocator, std.testing.io, &b, n.MetalLoader{}, .first);
    try std.testing.expectEqual(@as(usize, 0), summary.different);
    try std.testing.expect(summary.first[0] == null and summary.first[1] == null);
    try std.testing.expectEqual(@as(usize, 0), b.held);
    const io = std.testing.io;
    const ids = [_]u32{ 3, 5 };
    const legacy_ref = gate.legacy_reference;
    const a = std.testing.allocator;
    const legacy = try gate.runRows(n.G, read, a, io, &b, n.MetalLoader{}, &ids, legacy_ref, .nothing);
    try std.testing.expect(legacy.first[0] == null);
    try std.testing.expectEqual([2]usize{ 53, 24 }, [2]usize{ legacy.first[1] orelse 0, legacy.values[1] });
    try std.testing.expectEqual(@as(usize, 0), b.held);
}

test "Qwen4 device gate: full Metal storage-node diagnostic" {
    var b: n.B = .{};
    const doc = try std.json.parseFromSlice(f.Oracle, std.testing.allocator, f.metal_reference, .{});
    defer doc.deinit();
    const nodes = @embedFile("fixtures/native-metal-storage-nodes.json");
    const gate = @import("node_gate.zig");
    const io = std.testing.io;
    const ids = doc.value.tokens;
    const summary = try gate.runRows(n.G, read, std.testing.allocator, io, &b, n.MetalLoader{}, ids, nodes, .first);
    try std.testing.expectEqual(@as(usize, 0), summary.different);
    try std.testing.expectEqual(@as(usize, 0), b.held);
}

test "Qwen4 device gate: Metal profile preserves the reported probability precision counterexample" {
    const a = std.testing.allocator;
    const old = try std.json.parseFromSlice(f.Oracle, a, f.legacy_reference, .{});
    defer old.deinit();
    const metal = try std.json.parseFromSlice(f.Oracle, a, f.metal_reference, .{});
    defer metal.deinit();
    const bf = @import("core").quant.convert.bf16;
    const columns = [_]usize{ 1, 2, 4, 5, 6, 7, 10, 11 };
    const bits = [_]u16{ 0x3f7d, 0xc03d, 0xbfb7, 0x3e4e, 0xc018, 0xbdf9, 0xbee1, 0xbfa6 };
    for (columns, bits) |col, want| {
        try std.testing.expectEqual(want, bf(@floatCast(metal.value.logits[1][col])));
        try std.testing.expect(want != bf(@floatCast(old.value.logits[1][col])));
    }
    var different: usize = 0;
    for (old.value.logits, metal.value.logits) |before, after| for (before, after) |x, y| {
        different += @intFromBool(bf(@floatCast(x)) != bf(@floatCast(y)));
    };
    try std.testing.expectEqual(@as(usize, 219), different);
}
