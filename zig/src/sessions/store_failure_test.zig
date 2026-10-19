//! Exercise the real RAM store with failing policy allocations and failed partial-page transfers.
const std = @import("std");
const pm = @import("pool.zig");
const sm = @import("store.zig");
const prefix = @import("prefix.zig");
const dm = @import("disk.zig");
const PageStore = @import("pagestore.zig").PageStore;
const a = std.testing.allocator;
const fams = [_]pm.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 8 }};
const Bound = struct {
    released: usize = 0,
    failing: ?*std.testing.FailingAllocator = null,
    offset: usize = 0,
    fn nbytes(ptr: *anyopaque) u64 {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        if (b.failing) |f| {
            f.fail_index = f.alloc_index + b.offset;
            b.failing = null;
        }
        return 8;
    }
    fn blobs(_: *anyopaque, _: std.mem.Allocator) anyerror![]dm.Blob {
        return error.Unused;
    }
    fn release(ptr: *anyopaque) void {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        b.released += 1;
    }
    fn bounded(b: *@This()) sm.Bounded {
        return .{ .ptr = b, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
    }
};
const Bytes = struct {
    fail: bool = false,
    copies: usize = 0,
    fn read(_: *anyopaque, _: u32, _: []const u32, _: []u8) anyerror!void {
        return error.Unused;
    }
    fn write(_: *anyopaque, _: u32, _: []const u32, _: []const u8) anyerror!void {
        return error.Unused;
    }
    fn copy(ptr: *anyopaque, _: u32, _: u32, _: u32) anyerror!void {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        b.copies += 1;
        if (b.fail) return error.CopyFailed;
    }
    fn fence(_: *anyopaque) anyerror!u64 {
        return 0;
    }
    fn wait(_: *anyopaque, _: u64) anyerror!void {}
    fn publish(_: *anyopaque) anyerror!void {}
    fn store(b: *@This()) PageStore {
        return .{ .ptr = b, .vtable = &.{ .read = read, .write = write, .copyPage = copy, .fence = fence, .waitFence = wait, .publish = publish } };
    }
};
fn consistent(s: *sm.Store) !void {
    var count: u32 = 0;
    var bytes: u64 = 0;
    var prev: u32 = sm.none;
    var id: u32 = s.oldest;
    while (id != sm.none) {
        try std.testing.expect(count < s.index.entries.items.len);
        const e = s.entry(id);
        try std.testing.expect(e.live and e.ram);
        try std.testing.expectEqual(prev, e.extra.older);
        bytes += e.extra.bounded.?.vtable.nbytes(e.extra.bounded.?.ptr);
        prev = id;
        id = e.extra.newer;
        count += 1;
    }
    try std.testing.expectEqual(s.newest, prev);
    try std.testing.expectEqual(s.ram_count, count);
    try std.testing.expectEqual(s.bounded_total, bytes);
    var live: u32 = 0;
    for (s.index.entries.items) |e| if (e.live and e.ram) {
        live += 1;
    };
    try std.testing.expectEqual(count, live);
}
fn saveFailure(existing: bool, trim: bool, offset: usize) !bool {
    var pool = try pm.Pool.init(a, .{ .families = &fams, .page = 4 }, 64, 1, 0);
    defer pool.deinit();
    var bytes: Bytes = .{};
    var s = try sm.Store.init(a, std.testing.io, &pool, bytes.store(), null, .{});
    var old: [2]Bound = .{ .{}, .{} };
    var b: Bound = .{};
    var retry: Bound = .{};
    var closed = false;
    defer if (!closed) s.deinit();
    const idle = try pool.newSlot(8);
    const slot = try pool.newSlot(8);
    defer _ = slot.releaseAll() catch {};
    for (&old, 0..) |*o, i| {
        try idle.ensure(5);
        _ = try s.save(idle, o.bounded(), &.{ 9, 8, 7, 6, 5 }, @intCast(i + 10), 0);
        _ = try idle.releaseAll();
    }
    const ids = [_]i32{ 1, 2, 3, 4, 5 };
    try slot.ensure(ids.len);
    var disk_id: ?u32 = null;
    var disk_key: prefix.Digest = undefined;
    if (existing) {
        var d: [1]prefix.Digest = undefined;
        prefix.chain(7, &ids, 4, &d);
        const ins = try s.index.insert(7, &ids, &d, false, .{ .file = true, .kind = 19, .used = 42 });
        disk_id = ins.id;
        disk_key = s.entry(ins.id).key;
    }
    var f = std.testing.FailingAllocator.init(a, .{});
    s.gpa = f.allocator();
    s.index.gpa = f.allocator();
    s.policy = .{ .chats = 0, .ram_bytes = if (trim) 16 else std.math.maxInt(u64) };
    b.failing = &f;
    b.offset = offset;
    var failed = false;
    const saved = s.save(slot, b.bounded(), &ids, 7, 23) catch |err| blk: {
        failed = true;
        f.fail_index = std.math.maxInt(usize);
        // Keep the pre-fix red run's deinit from following its already-freed pages array.
        defer if (existing and s.entry(disk_id.?).ram) {
            s.entry(disk_id.?).extra.pages = &.{};
            s.entry(disk_id.?).extra.bounded = null;
        };
        defer b.bounded().vtable.release(&b);
        try std.testing.expectEqual(error.OutOfMemory, err);
        try std.testing.expectEqual(@as(usize, 0), b.released);
        for (slot.mapped()) |pg| try std.testing.expectEqual(@as(u16, 1), pool.holders(pg));
        try consistent(&s);
        if (disk_id) |id| {
            const e = s.entry(id);
            try std.testing.expect(e.live and !e.ram and e.extra.file);
            try std.testing.expectEqualSlices(u8, &disk_key, &e.key);
            try std.testing.expectEqual(@as(u32, 19), e.extra.kind);
            try std.testing.expectEqual(@as(u64, 42), e.extra.used);
            try std.testing.expectEqual(@as(usize, 0), e.extra.pages.len);
        } else {
            try std.testing.expectEqual(@as(?sm.Hit, null), try s.find(7, &.{ 1, 2, 3, 4, 5, 6 }));
        }
        const id = (try s.save(slot, retry.bounded(), &ids, 7, 23)).?;
        if (disk_id) |want| try std.testing.expectEqual(want, id);
        try consistent(&s);
        // Successful retry owns its bounded state until this explicit drop.
        try s.drop(id);
        try std.testing.expectEqual(@as(usize, 1), retry.released);
        break :blk null;
    };
    f.fail_index = std.math.maxInt(usize);
    if (saved) |id| {
        try consistent(&s);
        try s.drop(id);
    }
    try std.testing.expectEqual(@as(usize, 1), b.released);
    try consistent(&s);
    s.deinit();
    closed = true;
    for (old) |o| try std.testing.expectEqual(@as(usize, 1), o.released);
    for (pool.refs, pool.slot_refs, 0..) |refs, slot_refs, pg| {
        if (pg != pool.null_page) try std.testing.expectEqual(slot_refs, refs);
    }
    return failed;
}
fn sweep(existing: bool, trim: bool) !void {
    var failures: usize = 0;
    for (0..8) |offset| if (try saveFailure(existing, trim, offset)) {
        failures += 1;
    };
    try std.testing.expect(failures > 0);
}
test "Ownership rollback: new save rolls back published owners after policy OOM" {
    try sweep(false, false);
}
test "Ownership rollback: disk promotion rollback preserves identity and owners" {
    try sweep(true, false);
}
test "Ownership rollback: save rollback remains valid after earlier trim eviction" {
    try sweep(false, true);
}

fn restoreFailure(copy: bool) !void {
    var pool = try pm.Pool.init(a, .{ .families = &fams, .page = 4 }, 64, 1, 0);
    defer pool.deinit();
    var bytes: Bytes = .{};
    var s = try sm.Store.init(a, std.testing.io, &pool, bytes.store(), null, .{});
    var b: Bound = .{};
    defer s.deinit();
    const src = try pool.newSlot(8);
    const dst = try pool.newSlot(if (copy) 8 else 4);
    const retry = try pool.newSlot(8);
    try src.ensure(5);
    const id = (try s.save(src, b.bounded(), &.{ 1, 2, 3, 4, 5 }, 1, 0)).?;
    _ = try src.releaseAll();
    const free = pool.freePages();
    const available = pool.available();
    bytes.fail = copy;
    const result = s.restoreRam(id, dst);
    if (copy) try std.testing.expectError(error.CopyFailed, result) else try std.testing.expectError(error.PastCapacity, result);
    try std.testing.expectEqual(@as(u32, 0), dst.len);
    try std.testing.expectEqual(free, pool.freePages());
    try std.testing.expectEqual(available, pool.available());
    for (s.entry(id).extra.pages) |pg| try std.testing.expectEqual(@as(u16, 1), pool.holders(pg));
    bytes.fail = false;
    _ = try s.restoreRam(id, retry);
    try std.testing.expectEqual(@as(u32, 2), retry.len);
    _ = try retry.releaseAll();
    try std.testing.expectEqual(free, pool.freePages());
    try std.testing.expectEqual(@as(usize, 0), b.released);
    try s.drop(id);
    try std.testing.expectEqual(@as(usize, 1), b.released);
}
test "Ownership rollback: restore copy failure returns its temporary page" {
    try restoreFailure(true);
}
test "Ownership rollback: restore push failure returns its temporary page" {
    try restoreFailure(false);
}
