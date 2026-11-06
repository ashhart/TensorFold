//! The dump parser refuses what it cannot vouch for: foreign bytes, unknown versions, cut files and illegal phases.
const std = @import("std");
const c = @import("tp_check.zig");
const testing = std.testing;

fn dumpOf(a: std.mem.Allocator, version: u32, phase: c.Phase) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    defer w.deinit();
    try c.writeHeader(&w.writer, .{ .format_version = version, .rank = 0, .world = 2, .label = "reject fixture" });
    const row = [_]u8{ 0, 0, 128, 63 };
    try c.writeCollectiveVersion(&w.writer, version, .{ .seq = 0, .layer = 0, .phase = phase, .call = .all_reduce, .dtype = .f32, .op = .sum, .peer = 0, .send = &row, .recv = &row });
    return w.toOwnedSlice();
}

/// Where the phase byte of the first collective record sits: the only byte that differs between these two dumps.
fn phaseAt(legacy: []const u8, mixer: []const u8) !usize {
    try testing.expectEqual(legacy.len, mixer.len);
    for (legacy, mixer, 0..) |x, y, i| if (x != y) return i;
    return error.NoPhaseByte;
}

test "tp check: foreign magic is refused before anything else is read" {
    const a = testing.allocator;
    const bytes = try dumpOf(a, c.canonical_version, .bootstrap);
    defer a.free(bytes);
    bytes[0] ^= 0xff;
    try testing.expectError(error.BadMagic, c.parse(a, bytes));
}

test "tp check: an unsupported version is refused, whatever follows it" {
    const a = testing.allocator;
    const bytes = try dumpOf(a, c.canonical_version, .bootstrap);
    defer a.free(bytes);
    for ([_]u32{ 0, 4, 5, 255, 1 << 16, std.math.maxInt(u32) }) |v| {
        std.mem.writeInt(u32, bytes[c.magic.len..][0..4], v, .little);
        try testing.expectError(error.BadVersion, c.parse(a, bytes));
    }
}

test "tp check: every cut of a valid dump is refused or parses, and never reads past its end" {
    const a = testing.allocator;
    for ([_]u32{ c.version, c.phase_version, c.canonical_version }) |v| {
        const bytes = try dumpOf(a, v, if (v == c.version) .legacy else .mixer);
        defer a.free(bytes);
        var refused: usize = 0;
        for (0..bytes.len) |n| {
            if (c.parse(a, bytes[0..n])) |d| {
                var kept = d;
                kept.deinit(a);
            } else |err| {
                try testing.expect(err == error.Truncated or err == error.BadMagic);
                refused += 1;
            }
        }
        try testing.expect(refused > 0);
    }
}

test "tp check: a phase byte its version does not allow is refused" {
    const a = testing.allocator;
    const legacy = try dumpOf(a, c.phase_version, .legacy);
    defer a.free(legacy);
    const mixer = try dumpOf(a, c.phase_version, .mixer);
    defer a.free(mixer);
    const at = try phaseAt(legacy, mixer);
    const cases = [_]struct { version: u32, byte: u8 }{
        .{ .version = c.version, .byte = 1 },
        .{ .version = c.version, .byte = 3 },
        .{ .version = c.phase_version, .byte = 3 },
        .{ .version = c.phase_version, .byte = 4 },
        .{ .version = c.canonical_version, .byte = 4 },
        .{ .version = c.canonical_version, .byte = 255 },
    };
    for (cases) |t| {
        const bytes = try a.dupe(u8, legacy);
        defer a.free(bytes);
        std.mem.writeInt(u32, bytes[c.magic.len..][0..4], t.version, .little);
        bytes[at] = t.byte;
        try testing.expectError(error.BadEnum, c.parse(a, bytes));
    }
}

test "tp check: the writer refuses a phase or version the format does not have" {
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    const row = [_]u8{ 0, 0, 128, 63 };
    const rec: c.Collective = .{ .seq = 0, .layer = 0, .phase = .bootstrap, .call = .all_reduce, .dtype = .f32, .op = .sum, .peer = 0, .send = &row, .recv = &row };
    try testing.expectError(error.UnsupportedPhaseVersion, c.writeCollectiveVersion(&w.writer, c.version, rec));
    try testing.expectError(error.UnsupportedPhaseVersion, c.writeCollectiveVersion(&w.writer, c.phase_version, rec));
    try testing.expectError(error.UnsupportedPhaseVersion, c.writeCollectiveVersion(&w.writer, 4, rec));
    try testing.expectEqual(@as(usize, 0), w.written().len);
}
