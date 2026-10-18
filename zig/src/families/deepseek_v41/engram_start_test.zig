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
    const C = struct {
        fn run(a: std.mem.Allocator, pool: bool) !void {
            var tabs = [_]io.Table{try table(a)};
            defer freeTable(a, &tabs[0]);
            var p: io.Pool = .{};
            defer p.jobs.deinit(a);
            var r: Fake = .{ .gpa = a, .tables = &tabs, .pool = if (pool) &p else null };
            defer r.close() catch unreachable;
            _ = try io.start(&r, 0, &.{ 0, 99 }, false);
        }
    };
    try t.checkAllAllocationFailures(t.allocator, C.run, .{false});
    try t.checkAllAllocationFailures(t.allocator, C.run, .{true});
}
