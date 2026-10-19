//! Production start owner, with injected completion instead of Linux syscalls.
const std = @import("std");
const t = std.testing;
const io = @import("engram_io.zig");
const format = @import("engram_rows.zig");
const Fake = struct {
    gpa: std.mem.Allocator,
    tables: []io.Table,
    batches: std.ArrayList(io.Batch) = .empty,
    pool: ?*io.Pool = null,
    stats: struct { reads: u64 = 0, bytes: u64 = 0 } = .{},
    mode: enum { ok, serial, first, later, drain } = .ok,
    pub const slotFor = io.slotFor;
    pub const queue = io.queue;
    pub fn submitBatch(r: *Fake, bi: u32) !void {
        const b = &r.batches.items[bi];
        if (r.mode == .serial or r.mode == .first) return error.Injected;
        b.left = 1;
        if (r.mode == .later or r.mode == .drain) return error.Injected;
        try r.drain(bi);
    }
    pub fn drain(r: *Fake, bi: u32) !void {
        if (r.mode == .drain) return error.DrainFailed;
        const b = &r.batches.items[bi];
        // An accepted request writes its destination before releasing it.
        if (b.left > 0) @memset(b.buf, 0xa5);
        b.left = 0;
    }
    fn close(r: *Fake) !void {
        for (r.batches.items, 0..) |*b, i| if (b.in_use) {
            try r.drain(@intCast(i));
            for (b.rows) |x| r.tables[b.table].drop(x);
            r.gpa.free(b.rows);
            r.gpa.free(b.ext);
            r.gpa.free(b.buf);
            b.in_use = false;
        };
        r.batches.deinit(r.gpa);
    }
};
fn table(a: std.mem.Allocator) !io.Table {
    return io.tableStorage(a, 0, false, .{ .layer = 1, .lo = 0, .hi = 100, .total = 100, .row_bytes = format.row_bytes }, 4096 + 26400, 2);
}
fn freeTable(a: std.mem.Allocator, tb: *io.Table) void {
    tb.map.deinit(a);
    a.free(tb.recs);
    a.free(tb.ids);
    a.free(tb.prev);
    a.free(tb.next);
}
test "production batch start unwinds serial and first and later submit errors, then retries" {
    inline for (.{ .serial, .first, .later }) |mode| {
        var tabs = [_]io.Table{try table(t.allocator)};
        defer freeTable(t.allocator, &tabs[0]);
        var r: Fake = .{ .gpa = t.allocator, .tables = &tabs, .mode = mode };
        defer r.close() catch unreachable;
        try t.expectError(error.Injected, io.start(&r, 0, &.{ 0, 99 }, false));
        try t.expectEqual(@as(u32, 0), tabs[0].map.count());
        for (r.batches.items) |b| try t.expect(!b.in_use);
        r.mode = .ok;
        _ = try io.start(&r, 0, &.{ 0, 99 }, false);
        try t.expectEqual(@as(u32, 2), tabs[0].map.count());
    }
}
test "production batch start retains full live owner after failed drain" {
    var tabs = [_]io.Table{try table(t.allocator)};
    defer freeTable(t.allocator, &tabs[0]);
    var r: Fake = .{ .gpa = t.allocator, .tables = &tabs, .mode = .drain };
    try t.expectError(error.Injected, io.start(&r, 0, &.{ 0, 99 }, false));
    try t.expectError(error.DrainFailed, r.close());
    try t.expect(r.batches.items[0].in_use);
    try t.expectEqual(@as(u32, 2), tabs[0].map.count());
    r.mode = .ok;
    try r.close();
}
test "production batch map, batch list, pool queue allocations and table constructor unwind" {
    try sweep(t.allocator, false);
    try sweep(t.allocator, true);
}

fn allocationCase(a: std.mem.Allocator, pool: bool) !void {
    var tabs = [_]io.Table{try table(a)};
    defer freeTable(a, &tabs[0]);
    var p: io.Pool = .{};
    defer p.jobs.deinit(a);
    var r: Fake = .{ .gpa = a, .tables = &tabs, .pool = if (pool) &p else null };
    defer r.close() catch unreachable;
    _ = try io.start(&r, 0, &.{ 0, 99 }, false);
}

fn sweep(a: std.mem.Allocator, pool: bool) !void {
    var stable = std.testing.FailingAllocator.init(a, .{ .resize_fail_index = 0 });
    try t.checkAllAllocationFailures(stable.allocator(), allocationCase, .{pool});
}

const VaryRemap = struct {
    backing: std.mem.Allocator,
    cases: usize = 0,
    remaps: usize = 0,
    remap_successes: usize = 0,

    fn allocator(v: *VaryRemap) std.mem.Allocator {
        return .{ .ptr = v, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const v: *VaryRemap = @ptrCast(@alignCast(ctx));
        if (len == 2 * format.row_bytes) v.cases += 1;
        return v.backing.rawAlloc(len, alignment, ra);
    }
    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const v: *VaryRemap = @ptrCast(@alignCast(ctx));
        v.remaps += 1;
        if (v.cases <= 1) return null;
        const ptr = v.backing.rawRemap(memory, alignment, len, ra) orelse return null;
        v.remap_successes += 1;
        return ptr;
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const v: *VaryRemap = @ptrCast(@alignCast(ctx));
        v.backing.rawFree(memory, alignment, ra);
    }
};

test "engram alloc: both complete failure sweeps ignore changing remap success" {
    for ([_]bool{ false, true }) |pool| {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        var varying: VaryRemap = .{ .backing = arena.allocator() };
        try sweep(varying.allocator(), pool);
        try t.expect(varying.cases > 1);
        try t.expectEqual(@as(usize, 0), varying.remaps);
    }
}

test "engram alloc: successful owner paths still exercise refused and successful remaps" {
    for ([_]bool{ false, true }) |pool| {
        var arena: std.heap.ArenaAllocator = .init(t.allocator);
        defer arena.deinit();
        var varying: VaryRemap = .{ .backing = arena.allocator() };
        try allocationCase(varying.allocator(), pool);
        try t.expect(varying.remaps > 0 and varying.remap_successes == 0);
        try allocationCase(varying.allocator(), pool);
        try t.expect(varying.remap_successes > 0);
    }
}
