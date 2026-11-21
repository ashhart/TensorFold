//! Versioned phase records keep legacy dumps readable and reject layer or phase disagreement.
const std = @import("std");
const c = @import("tp_check.zig");
fn fixture(rank: u32, version: u32) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();
    try c.writeHeader(&w.writer, .{ .format_version = version, .rank = rank, .world = 2, .label = "phase host fixture" });
    var send: [4]u8 = undefined;
    var recv: [4]u8 = undefined;
    std.mem.writeInt(u32, &send, @bitCast(@as(f32, if (rank == 0) 1 else 2)), .little);
    std.mem.writeInt(u32, &recv, @bitCast(@as(f32, 3)), .little);
    for ([_]c.Phase{ .mixer, .ffn }, 0..) |phase, seq| try c.writeCollectiveVersion(&w.writer, version, .{
        .phase = if (version == 1) .legacy else phase,
        .seq = seq,
        .layer = 0,
        .call = .all_reduce,
        .dtype = .f32,
        .op = .sum,
        .peer = 1 - rank,
        .send = &send,
        .recv = &recv,
    });
    try c.writeLogits(&w.writer, 0, 0, 1, &.{ 0.5, 1.0 });
    return w.toOwnedSlice();
}
test "phase dump versions roundtrip and negative layer and phase controls fail rule1 and rule2" {
    const a = std.testing.allocator;
    for ([_]u32{ 1, 2 }) |version| {
        const b0 = try fixture(0, version);
        defer a.free(b0);
        const b1 = try fixture(1, version);
        defer a.free(b1);
        var d0 = try c.parse(a, b0);
        defer d0.deinit(a);
        var d1 = try c.parse(a, b1);
        defer d1.deinit(a);
        try std.testing.expectEqual(version, d0.header.format_version);
        try std.testing.expectEqual(if (version == 1) c.Phase.legacy else c.Phase.mixer, d0.collectives[0].phase);
        try std.testing.expectEqual(@as(usize, 0), (try c.checkReference(a, &.{ d0, d1 })).mismatches);
        try std.testing.expect(c.repeatPassed(c.checkRepeat(d0, d1)) == false);
        d1.collectives[0].layer = 1;
        try std.testing.expect((try c.checkReference(a, &.{ d0, d1 })).mismatches > 0);
        d1.collectives[0].layer = 0;
        if (version == 2) {
            d1.collectives[0].phase = .ffn;
            try std.testing.expect((try c.checkReference(a, &.{ d0, d1 })).mismatches > 0);
            var clone = try c.parse(a, b0);
            defer clone.deinit(a);
            clone.collectives[0].phase = .ffn;
            try std.testing.expect(!c.repeatPassed(c.checkRepeat(d0, clone)));
        }
    }
}
test "version1 rejects a phase byte and version2 rejects an unknown phase" {
    const a = std.testing.allocator;
    for ([_]u32{ 1, 2 }) |version| {
        const bytes = try fixture(0, version);
        defer a.free(bytes);
        const label = "phase host fixture";
        const record = 40 + label.len;
        bytes[record + 19] = if (version == 1) 1 else 255;
        try std.testing.expectError(error.BadEnum, c.parse(a, bytes));
    }
}

test "version2 phase collectives retain rule4 compatibility with legacy solo logits" {
    const a = std.testing.allocator;
    const bytes = try fixture(0, 2);
    defer a.free(bytes);
    var mixed = try c.parse(a, bytes);
    defer mixed.deinit(a);
    var w: std.Io.Writer.Allocating = .init(a);
    defer w.deinit();
    try c.writeHeader(&w.writer, .{ .rank = 0, .world = 1, .label = "legacy solo" });
    try c.writeLogits(&w.writer, 0, 0, 1, &.{ 0.5, 1.0 });
    var solo = try c.parse(a, w.written());
    defer solo.deinit(a);
    var report = try c.checkAgree(a, mixed, solo);
    defer report.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), report.greedy_same);
    try std.testing.expectEqual(@as(usize, 1), report.top1_same);
    try std.testing.expectEqual(@as(usize, 0), report.splits.len);
}

test "legacy writer rejects phase emission before writing bytes" {
    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();
    const record: c.Collective = .{ .phase = .mixer, .seq = 0, .layer = 0, .call = .all_reduce, .dtype = .f32, .op = .sum, .peer = 1, .send = &.{}, .recv = &.{} };
    try std.testing.expectError(error.UnsupportedPhaseVersion, c.writeCollective(&w.writer, record));
    try std.testing.expectError(error.UnsupportedPhaseVersion, c.writeCollectiveVersion(&w.writer, 1, record));
    try std.testing.expectEqual(@as(usize, 0), w.written().len);
}

const frozen_v1 =
    "\x54\x46\x54\x50\x44\x4d\x50\x31\x01\x00\x00\x00\x00\x00\x00\x00" ++
    "\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00" ++
    "\x10\x00\x00\x00\x66\x72\x6f\x7a\x65\x6e\x20\x6c\x65\x67\x61\x63" ++
    "\x79\x20\x76\x31\x00\x00\x00\x00\x01\x00\x00\x00\x09\x00\x00\x00" ++
    "\x00\x00\x00\x00\x07\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00" ++
    "\x04\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00" ++
    "\x00\x00\x40\x40\x00\x00\x40\x40\x02\x00\x00\x00\x00\x00\x00\x00" ++
    "\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00" ++
    "\x00\x00\x00\x3f\x00\x00\x80\x3f";

test "independently frozen version1 bytes remain readable" {
    const a = std.testing.allocator;
    var d = try c.parse(a, frozen_v1);
    defer d.deinit(a);
    try std.testing.expectEqual(@as(u32, 1), d.header.format_version);
    try std.testing.expectEqual(c.Phase.legacy, d.collectives[0].phase);
    try std.testing.expectEqual(@as(u64, 9), d.collectives[0].seq);
    try std.testing.expectEqual(@as(u32, 7), d.collectives[0].layer);
    try std.testing.expectEqual(@as(usize, 0), (try c.checkReference(a, &.{d})).mismatches);
    try std.testing.expectEqual(@as(f32, 1.0), d.logits[0].at(1));
}

test "version3 bootstrap is explicit and cannot be emitted as version2" {
    const a = std.testing.allocator;
    var w: std.Io.Writer.Allocating = .init(a);
    defer w.deinit();
    const row: c.Collective = .{ .phase = .bootstrap, .seq = 0, .layer = 64, .call = .exchange, .dtype = .u8, .op = .sum, .peer = 1, .send = "H", .recv = "H" };
    try std.testing.expectError(error.UnsupportedPhaseVersion, c.writeCollectiveVersion(&w.writer, 2, row));
    try std.testing.expectEqual(@as(usize, 0), w.written().len);
    try c.writeHeader(&w.writer, .{ .format_version = 3, .rank = 0, .world = 2, .label = "MCT3" });
    try c.writeCollectiveVersion(&w.writer, 3, row);
    var d = try c.parse(a, w.written());
    defer d.deinit(a);
    try std.testing.expectEqual(c.Phase.bootstrap, d.collectives[0].phase);
    try std.testing.expectEqual(@as(u32, 64), d.collectives[0].layer);
    var bytes = try a.dupe(u8, w.written());
    defer a.free(bytes);
    // Header is 44 bytes for label MCT3 and empty provenance; collective phase follows kind/seq/layer/three enums.
    bytes[44 + 4 + 8 + 4 + 3] = 4;
    try std.testing.expectError(error.BadEnum, c.parse(a, bytes));
}
