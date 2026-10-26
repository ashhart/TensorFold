//! A two-row fresh pass dumps every observable storage node before the unchanged complete device exactness gate.
const std = @import("std");
const fixture = @import("device_fixture.zig");
const capture = @import("node_capture.zig");
pub const cuda_reference = capture.cuda_reference;
pub fn run(comptime G: type, comptime read: anytype, a: std.mem.Allocator, io: std.Io, b: *G.Storage, loader: anytype, verbose: bool) !capture.Summary {
    const nodes = if (@hasDecl(@TypeOf(loader), "node_reference")) @TypeOf(loader).node_reference else capture.reference;
    return runRows(G, read, a, io, b, loader, &.{ 3, 5 }, nodes, verbose);
}
pub fn runRows(comptime G: type, comptime read: anytype, a: std.mem.Allocator, io: std.Io, b: *G.Storage, loader: anytype, ids: []const u32, reference: []const u8, verbose: bool) !capture.Summary {
    const E = @import("engine.zig").Engine(G);
    const file = try fixture.Fixture.open(a, io);
    defer file.close();
    var p: @import("config.zig").Problem = .{};
    const e = try E.open(a, io, file.path, b, .{ .capacity = 32, .window = 16, .streams = 2, .budget = std.math.maxInt(u64) }, &p, loader);
    defer e.close() catch {};
    var snapshots: capture.Capture(G.Storage, read) = .{ .a = a, .backend = b };
    defer snapshots.close() catch {};
    const doc = try std.json.parseFromSlice(capture.Oracle, a, reference, .{ .ignore_unknown_fields = true });
    defer doc.deinit();
    const slot = try e.lease();
    e.trace = snapshots.sink();
    e.gpu.trace = snapshots.sink();
    try e.forward(&.{.{ .state = &slot.state, .tokens = ids }}, E.NoDraw{}, &p);
    try b.drain();
    e.trace = null;
    e.gpu.trace = null;
    if (ids.len == 2 and ids[0] == 3 and ids[1] == 5) {
        const legacy = try std.json.parseFromSlice(capture.Oracle, a, @embedFile("fixtures/native-legacy-nodes.json"), .{ .ignore_unknown_fields = true });
        defer legacy.deinit();
        if (verbose) std.debug.print("QWEN4_REFERENCE legacy BF16 probabilities: diagnostic before correction\n", .{});
        const before = try snapshots.report(legacy.value, false);
        if (verbose) std.debug.print("QWEN4_REFERENCE legacy: {d} differing BF16 node values\n", .{before.different});
    }
    const profile = if (@hasDecl(@TypeOf(loader), "node_profile")) @TypeOf(loader).node_profile else "Metal scalar nodes and FP32 probabilities";
    if (verbose) std.debug.print("QWEN4_REFERENCE {s}: strict storage nodes\n", .{profile});
    const result = try snapshots.report(doc.value, verbose);
    try e.keep(slot, ids.len);
    try e.release(slot);
    return result;
}
