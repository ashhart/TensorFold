//! Real synchronous park/trim during promotion. This test runs only on Linux and uses actual files.
const std = @import("std");
const pm = @import("pool.zig");
const ps = @import("pagestore.zig");
const dm = @import("disk.zig");
const sm = @import("store.zig");
const a = std.testing.allocator;
const Bound = struct {
    released: usize = 0,
    failing: ?*std.testing.FailingAllocator = null,
    offset: usize = 0,
    fn nbytes(ptr: *anyopaque) u64 {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        if (b.failing) |f| {
            b.failing = null;
            f.fail_index = f.alloc_index + b.offset;
        }
        return 8;
    }
    fn blobs(_: *anyopaque, g: std.mem.Allocator) anyerror![]dm.Blob {
        return g.alloc(dm.Blob, 0);
    }
    fn release(ptr: *anyopaque) void {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        b.released += 1;
    }
    fn value(b: *@This()) sm.Bounded {
        return .{ .ptr = b, .vtable = &.{ .nbytes = nbytes, .blobs = blobs, .release = release } };
    }
};
fn run(offset: usize) !bool {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    const fams = [_]pm.Family{.{ .name = "kv", .ratio = 1, .row_bytes = 8 }};
    var pool = try pm.Pool.init(a, .{ .families = &fams, .page = 4 }, 64, 1, 0);
    defer pool.deinit();
    var host = try ps.HostPool.init(a, &pool);
    defer host.deinit();
    const d = try dm.Disk.open(a, io, .{ .root = root, .compat = @splat(19), .min_tokens = 1, .direct = false, .chunk = 64 << 10, .staging = 2, .lanes = 1 });
    defer d.close();
    var s = try sm.Store.init(a, io, &pool, host.store(), d, .{});
    defer s.deinit();
    const slot = try pool.newSlot(8);
    defer _ = slot.releaseAll() catch {};
    var old: [3]Bound = .{ .{}, .{}, .{} };
    const ids = [_]i32{ 1, 2, 3, 4, 5 };
    try slot.ensure(5);
    const e = (try s.save(slot, old[0].value(), &ids, 7, 19)).?;
    const key = s.entry(e).key;
    _ = try slot.releaseAll();
    _ = try s.evict(e);
    try std.testing.expect(d.by_key.contains(key));
    var ab: [2]u32 = undefined;
    for (&ab, 0..) |*id, i| {
        try slot.ensure(5);
        id.* = (try s.save(slot, old[i + 1].value(), &.{ 9, 8, 7, 6, 5 }, @intCast(10 + i), 0)).?;
        _ = try slot.releaseAll();
    }
    d.opts.budget = d.used;
    try slot.ensure(5);
    var f = std.testing.FailingAllocator.init(a, .{});
    s.gpa = f.allocator();
    s.index.gpa = f.allocator();
    var promoted: Bound = .{ .failing = &f, .offset = offset };
    var retry: Bound = .{};
    s.policy = .{ .ram_bytes = 16, .chats = 0 };
    const result = s.save(slot, promoted.value(), &ids, 7, 23);
    f.fail_index = std.math.maxInt(usize);
    if (result) |saved| {
        if (saved) |id| try s.drop(id);
        return false;
    } else |err| {
        defer promoted.value().vtable.release(&promoted);
        try std.testing.expectEqual(error.OutOfMemory, err);
        if (d.stats.evicted == 0 or d.by_key.contains(key) or s.entry(ab[0]).ram or !s.entry(ab[1]).ram) return false;
        for (s.index.entries.items) |entry| if (entry.live and !entry.ram and entry.extra.file) {
            try std.testing.expect(d.by_key.contains(entry.key));
        };
        try std.testing.expect(s.index.lookup(key) == null);
        try std.testing.expectEqual(@as(usize, 0), promoted.released);
        for (slot.mapped()) |pg| try std.testing.expectEqual(@as(u16, 1), pool.holders(pg));
        s.policy = .{};
        const id = (try s.save(slot, retry.value(), &ids, 7, 23)).?;
        try s.drop(id);
        try std.testing.expectEqual(@as(usize, 1), retry.released);
        return true;
    }
}
test "Residency rollback: Linux promoting E parks A, evicts E on disk, then policy OOM leaves B and no stale file flag" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    var exercised = false;
    for (0..48) |offset| if (try run(offset)) {
        exercised = true;
    };
    try std.testing.expect(exercised);
}
