//! A saved bank is checked before use: its geometry, its data's CRC and its files' owner-only modes.
const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const store = @import("store.zig");
const t = std.testing;

fn saved(tmp: *std.testing.TmpDir, mock: *safety.Mock) ![:0]u8 {
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    errdefer t.allocator.free(dir);
    const b = try safety.taught(t.allocator, dir, mock, true);
    b.deinit();
    return dir;
}

test "bank validate: a bank written for another geometry of the same sizes is refused by name" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try saved(&tmp, &mock);
    defer t.allocator.free(dir);
    var other: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    other.arch.rope_theta = 1_000_000.0; // every tensor the same size, every stored K rotated for another model
    const b = try Bank.init(t.allocator, t.io, dir, other.backend());
    defer b.deinit();
    try t.expectError(error.LivingWeightsArchMismatch, b.loadIfPresent());
    try t.expect(!b.hasUnits());
}

test "bank validate: a flipped byte in a saved tensor is refused by name before any value is used" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try saved(&tmp, &mock);
    defer t.allocator.free(dir);
    {
        const f = try tmp.dir.openFile(t.io, store.file_name, .{ .mode = .read_write });
        defer f.close(t.io);
        var len: [8]u8 = undefined;
        _ = try f.readPositionalAll(t.io, &len, 0);
        const data = 8 + std.mem.readInt(u64, &len, .little);
        var byte: [1]u8 = undefined;
        _ = try f.readPositionalAll(t.io, &byte, data + 64);
        byte[0] ^= 0x10;
        try f.writePositionalAll(t.io, &byte, data + 64);
    }
    const b = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer b.deinit();
    try t.expectError(error.LivingWeightsCorrupt, b.loadIfPresent());
}

test "bank validate: the base file and the journal are created readable by their owner alone" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const more = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true });
    _ = try b.save(); // into the journal
    for ([_][]const u8{ store.file_name, store.delta_file_name }) |name| {
        const st = try tmp.dir.statFile(t.io, name, .{});
        try t.expectEqual(@as(std.posix.mode_t, 0o600), st.permissions.toMode() & 0o777);
    }
}
