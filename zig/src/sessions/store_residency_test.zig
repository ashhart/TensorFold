//! A policy-side-effect fixture uses the real Disk registry but does not simulate successful filesystem I/O.
const std = @import("std");
const sm = @import("store.zig");
const pm = @import("pool.zig");
const dm = @import("disk.zig");
const ps = @import("pagestore.zig");
const prefix = @import("prefix.zig");
const a = std.testing.allocator;
const fams = [_]pm.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 8 }};
const Bound = struct {
    released: usize = 0,
    side: ?*Effect = null,
    fn nbytes(_: *anyopaque) u64 {
        return 8;
    }
    fn blobs(_: *anyopaque, _: std.mem.Allocator) anyerror![]dm.Blob {
        return error.Unused;
    }
    fn release(ptr: *anyopaque) void {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        b.released += 1;
        if (b.side) |e| {
            b.side = null;
            e.take();
        }
    }
    fn value(b: *@This()) sm.Bounded {
        return .{ .ptr = b, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
    }
};
const Effect = struct {
    s: *sm.Store,
    d: *dm.Disk,
    f: *std.testing.FailingAllocator,
    id: u32,
    evict: bool,
    called: bool = false,
    published: bool = false,
    fn take(e: *Effect) void {
        e.called = true;
        e.published = e.s.entry(e.id).ram;
        if (e.evict) {
            _ = e.d.by_key.remove(e.s.entry(e.id).key);
            e.s.entry(e.id).extra.file = false;
        }
        e.f.fail_index = e.f.alloc_index;
    }
};
fn read(_: *anyopaque, _: u32, _: []const u32, _: []u8) anyerror!void {
    return error.Unused;
}
fn write(_: *anyopaque, _: u32, _: []const u32, _: []const u8) anyerror!void {
    return error.Unused;
}
fn copy(_: *anyopaque, _: u32, _: u32, _: u32) anyerror!void {}
fn fence(_: *anyopaque) anyerror!u64 {
    return 0;
}
fn wait(_: *anyopaque, _: u64) anyerror!void {}
fn publish(_: *anyopaque) anyerror!void {}
fn run(evict: bool) !void {
    var pool = try pm.Pool.init(a, .{ .families = &fams, .page = 4 }, 64, 1, 0);
    defer pool.deinit();
    var dummy: u8 = 0;
    const bytes: ps.PageStore = .{ .ptr = &dummy, .vtable = &.{ .read = read, .write = write, .copyPage = copy, .fence = fence, .waitFence = wait, .publish = publish } };
    var s = try sm.Store.init(a, std.testing.io, &pool, bytes, null, .{});
    defer s.deinit();
    var registry: dm.Disk = undefined;
    registry.by_key = .empty;
    registry.opts = .{ .root = "unused", .compat = @splat(0), .min_tokens = std.math.maxInt(u64) };
    defer registry.by_key.deinit(a);
    var old: [2]Bound = .{ .{}, .{} };
    var b: Bound = .{};
    var retry: Bound = .{};
    const slot = try pool.newSlot(8);
    defer _ = slot.releaseAll() catch {};
    for (&old, 0..) |*o, i| {
        try slot.ensure(5);
        _ = try s.save(slot, o.value(), &.{ 9, 8, 7, 6, 5 }, @intCast(10 + i), 0);
        _ = try slot.releaseAll();
    }
    const ids = [_]i32{ 1, 2, 3, 4, 5 };
    var dig: [1]prefix.Digest = undefined;
    prefix.chain(7, &ids, 4, &dig);
    const id = (try s.index.insert(7, &ids, &dig, false, .{ .file = true, .kind = 19, .used = 42 })).id;
    const key = s.entry(id).key;
    try registry.by_key.put(a, key, 0);
    s.disk = &registry;
    try slot.ensure(5);
    var f = std.testing.FailingAllocator.init(a, .{});
    s.gpa = f.allocator();
    s.index.gpa = f.allocator();
    var effect: Effect = .{ .s = &s, .d = &registry, .f = &f, .id = id, .evict = evict };
    old[0].side = &effect;
    s.policy = .{ .ram_bytes = 16, .chats = 0 };
    const result = s.save(slot, b.value(), &ids, 7, 23);
    f.fail_index = std.math.maxInt(usize);
    defer b.value().vtable.release(&b);
    try std.testing.expectError(error.OutOfMemory, result);
    try std.testing.expect(effect.called and effect.published);
    try std.testing.expectEqual(@as(usize, 0), b.released);
    try std.testing.expectEqual(@as(usize, 1), old[0].released);
    try std.testing.expectEqual(@as(usize, 0), old[1].released);
    try std.testing.expectEqual(@as(u32, 1), s.ram_count);
    try std.testing.expectEqual(@as(u64, 8), s.bounded_total);
    for (slot.mapped()) |pg| try std.testing.expectEqual(@as(u16, 1), pool.holders(pg));
    for (s.index.entries.items) |e| if (e.live and !e.ram and e.extra.file) {
        try std.testing.expect(registry.by_key.contains(e.key));
    };
    if (evict) {
        try std.testing.expect(s.index.lookup(key) == null);
        try std.testing.expectEqual(@as(?sm.Hit, null), try s.find(7, &.{ 1, 2, 3, 4, 5, 6 }));
    } else {
        try std.testing.expectEqual(id, s.index.lookup(key).?);
        try std.testing.expectEqual(@as(u32, 19), s.entry(id).extra.kind);
        try std.testing.expectEqual(@as(u64, 42), s.entry(id).extra.used);
    }
    s.policy = .{};
    const again = (try s.save(slot, retry.value(), &ids, 7, 23)).?;
    try std.testing.expect(s.entry(again).ram);
    try s.drop(again);
    try std.testing.expectEqual(@as(usize, 1), retry.released);
}
test "Residency rollback: promotion rollback respects a file evicted by a committed policy side effect" {
    try run(true);
}
test "Residency rollback: promotion rollback preserves a file still in the Disk registry" {
    try run(false);
}
