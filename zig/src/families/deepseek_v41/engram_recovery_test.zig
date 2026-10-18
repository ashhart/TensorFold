//! The real Linux Rows.rows algorithm with injected disk completion, portable on every host.
const std = @import("std");
const t = std.testing;
const io = @import("engram_io.zig");
const format = @import("engram_rows.zig");
const production = @import("engram_rows_linux.zig");
const Fake = struct {
    gpa: std.mem.Allocator,
    tables: []io.Table,
    batches: std.ArrayList(io.Batch) = .empty,
    pool: ?*io.Pool = null,
    stats: production.Stats = .{},
    uniq: std.ArrayList(u64) = .empty,
    recs: std.ArrayList([*]const u8) = .empty,
    want: std.ArrayList(u32) = .empty,
    lut: *[65536]u16,
    cap: u32 = 16,
    fail_drain: bool = true,
    submitted: usize = 0,
    pub const rows = production.Rows.rows;
    pub const unique = production.Rows.unique;
    pub const settle = production.Rows.settle;
    pub const start = io.start;
    pub const slotFor = io.slotFor;
    pub const queue = io.queue;
    pub fn submitBatch(r: *Fake, bi: u32) !void {
        r.submitted += 1;
        r.batches.items[bi].left = @intCast(r.batches.items[bi].ext.len);
        if (r.fail_drain) return error.Injected;
        try r.drain(bi);
    }
    pub fn drain(r: *Fake, bi: u32) !void {
        if (r.fail_drain) return error.DrainFailed;
        const b = &r.batches.items[bi];
        for (b.rows, 0..) |id, k| {
            const rec = @constCast(b.record(k, 0));
            const code: u8 = switch (id) {
                0 => 0x38,
                50 => 0x40,
                99 => 0x44,
                else => unreachable,
            };
            @memset(rec[0..256], code);
            @memset(rec[256..], 127);
        }
        b.left = 0;
    }
    pub fn wait(r: *Fake, bi: u32) !void {
        try r.drain(bi);
        if (r.batches.items[bi].failed) return error.EngramReadFailed;
    }
    fn close(r: *Fake) void {
        r.fail_drain = false;
        for (r.batches.items, 0..) |b, i| if (b.in_use) {
            r.drain(@intCast(i)) catch unreachable;
            io.discard(r, @intCast(i));
        };
        r.batches.deinit(r.gpa);
        r.uniq.deinit(r.gpa);
        r.recs.deinit(r.gpa);
        r.want.deinit(r.gpa);
    }
};
test "Rows.rows recovers retained 0/99 before classifying 0/50/99 and reusing batch zero" {
    const a = t.allocator;
    var tables = [_]io.Table{try io.tableStorage(a, 0, false, .{ .layer = 1, .lo = 0, .hi = 100, .total = 100, .row_bytes = format.row_bytes }, 4096 + 26400, 16)};
    defer {
        tables[0].map.deinit(a);
        a.free(tables[0].recs);
        a.free(tables[0].ids);
        a.free(tables[0].prev);
        a.free(tables[0].next);
    }
    const lut = try a.create([65536]u16);
    defer a.destroy(lut);
    format.fillTable(lut);
    var r: Fake = .{ .gpa = a, .tables = &tables, .lut = lut };
    defer r.close();
    var initial: [64]u16 = undefined;
    try t.expectError(error.Injected, r.rows(0, &.{ 0, 99 }, 32, &initial));
    try t.expectEqualSlices(u64, &.{ 0, 99 }, r.batches.items[0].rows);
    try t.expect(r.batches.items[0].in_use and r.batches.items[0].failed and r.batches.items[0].left > 0);
    try t.expectEqual(@as(u32, 0), tables[0].map.get(0).?.pending);
    try t.expectEqual(@as(u32, 0), tables[0].map.get(99).?.pending);
    const held = r.batches.items[0].buf.ptr;
    var out: [96]u16 = undefined;
    try t.expectError(error.DrainFailed, r.rows(0, &.{ 0, 50, 99 }, 32, &out));
    try t.expectEqual(held, r.batches.items[0].buf.ptr);
    try t.expectEqual(@as(usize, 0), r.want.items.len);
    try t.expectEqual(@as(u32, 0), r.batches.items[0].taken);
    try t.expectEqual(@as(usize, 1), r.submitted);
    r.fail_drain = false;
    try r.rows(0, &.{ 0, 50, 99 }, 32, &out);
    for ([_]f32{ 1, 2, 3 }, 0..) |value, k| for (out[k * 32 ..][0..32]) |bits| try t.expectEqual(format.bf16(value), bits);
    try t.expectEqual(@as(usize, 1), r.batches.items.len);
    try t.expectEqual(@as(usize, 2), r.submitted);
    try t.expectEqual(@as(u32, 3), tables[0].map.count());
    var cached: [128]u16 = undefined;
    try r.rows(0, &.{ 99, 0, 50, 99 }, 32, &cached);
    for ([_]f32{ 3, 1, 2, 3 }, 0..) |value, k| for (cached[k * 32 ..][0..32]) |bits| try t.expectEqual(format.bf16(value), bits);
    try t.expectEqual(@as(usize, 2), r.submitted);
}
