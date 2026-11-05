//! Real conversation files for host namespace regressions; the eight-byte image uses no device.
const std = @import("std");
const pc = @import("engine_api").prompt_cache;
const rows = pc.rows;
const a = std.testing.allocator;
const value: u64 = 0x123456789abcdef0;
const tokens = [_]u32{ 5, 7, 11 };

fn bytes(_: *anyopaque, _: u32) u64 {
    return 8;
}
fn save(_: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved {
    const p = try a.create(u64);
    p.* = value;
    return p;
}
fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
fn drop(_: *anyopaque, saved: pc.Saved) void {
    a.destroy(@as(*u64, @ptrCast(@alignCast(saved))));
}
fn parts(_: *anyopaque, _: u32, out: *std.ArrayList(rows.image.Part), allocator: std.mem.Allocator) !void {
    try out.append(allocator, .{ .bytes = 8 });
}
fn blank(ptr: *anyopaque, at: u32) !pc.Saved {
    const saved = try save(ptr, null, at);
    @as(*u64, @ptrCast(@alignCast(saved))).* = 0;
    return saved;
}
fn copyOut(_: *anyopaque, saved: pc.Saved, ranges: []const rows.image.Range, out: []u8) !void {
    const p: *u64 = @ptrCast(@alignCast(saved));
    var at: usize = 0;
    for (ranges) |range| {
        const n: usize = @intCast(range.len);
        @memcpy(out[at..][0..n], std.mem.asBytes(p)[@intCast(range.off)..][0..n]);
        at += n;
    }
}
fn copyIn(_: *anyopaque, saved: pc.Saved, ranges: []const rows.image.Range, in: []const u8) !void {
    const p: *u64 = @ptrCast(@alignCast(saved));
    var at: usize = 0;
    for (ranges) |range| {
        const n: usize = @intCast(range.len);
        @memcpy(std.mem.asBytes(p)[@intCast(range.off)..][0..n], in[at..][0..n]);
        at += n;
    }
}
const image: rows.Image = .{ .parts = parts, .gather = copyOut, .blank = blank, .scatter = copyIn };
const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &image };

/// Plant a file, reopen and read it, then reject changed keys without destroying the matching state.
pub fn misses(before: u64, matching: u64, changed: []const u64) !void {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(root);
    var context: u8 = 0;
    const family: pc.Snapshots = .{ .ptr = &context, .vtable = &vtable };
    var options: rows.Options = .{ .root = root, .identity = before, .cap = 1 << 20, .floor = 0, .daily = 1 << 20, .lookahead = 1, .sync_writes = true };
    {
        const disk = try rows.Rows.open(a, family, options);
        defer disk.deinit();
        var store = pc.Store.init(a, family, .{ .lookahead = 1 }, 1 << 20);
        defer store.deinit();
        store.disk = disk;
        try std.testing.expect(store.keep(&tokens, 2, null, &.{}, &.{}));
        while (rows.write.step(disk)) {}
        try std.testing.expectEqual(@as(u64, 1), disk.counts.writes);
    }
    options.identity = matching;
    const same = try rows.Rows.open(a, family, options);
    defer same.deinit();
    const i = same.best(&tokens, &.{}, &.{}, 0) orelse return error.ExpectedMatchingDiskState;
    const saved = try rows.read.read(same, i);
    defer drop(&context, saved);
    try std.testing.expectEqual(value, @as(*u64, @ptrCast(@alignCast(saved))).*);
    for (changed) |identity| {
        options.identity = identity;
        const stale = try rows.Rows.open(a, family, options);
        defer stale.deinit();
        try std.testing.expect(stale.best(&tokens, &.{}, &.{}, 0) == null);
        try std.testing.expect(same.best(&tokens, &.{}, &.{}, 0) != null);
    }
}
