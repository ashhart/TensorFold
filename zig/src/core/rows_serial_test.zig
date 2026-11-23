//! States read back from disk are numbered like kept ones: a lent one is borrowed again, never another.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };

fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 171, .cap = 1 << 26, .floor = 0, .daily = 1 << 28, .lookahead = 1 };
}

test "rows serial: two states read back from disk carry their own serials, and a reborrow takes exactly the one lent" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    var pa: [2000]u32 = undefined;
    for (&pa, 0..) |*token, i| token.* = @intCast(17 + i * 3);
    var pb: [2000]u32 = undefined;
    for (&pb, 0..) |*token, i| token.* = @intCast(29 + i * 7);
    var f: fake.Fake = .{ .gpa = a, .prompt = &pa, .at = 1400 };
    { // both prompts' states on disk, from a store that has gone
        const r = try rows.Rows.open(a, f.snapshots(), options(dir));
        defer r.deinit();
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        for ([_][]const u32{ &pa, &pb }) |p| {
            f.prompt = p;
            try t.expect(s.keep(p, 1400, null, &.{}, &.{}));
            while (rows.write.step(r)) {}
        }
    }
    const r = try rows.Rows.open(a, f.snapshots(), options(dir));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    var read: [2]*pc.Entry = undefined;
    for ([_][]const u32{ &pa, &pb }, &read) |p, *e| { // each read back by its own lookup, then lent (released)
        f.prompt = p;
        const l = try s.lookup(a, p, 0, &.{}, &.{}, &.{});
        defer a.free(l.marks);
        e.* = l.entry orelse return error.NotReadBack;
        s.release(e.*);
    }
    try t.expect(read[0].serial != 0 and read[1].serial != 0 and read[0].serial != read[1].serial);
    try t.expect(s.reborrow(0) == null); // no entry is unnumbered, and none is lent as one
    try t.expectEqual(read[1], s.reborrow(read[1].serial).?);
    s.release(read[1]);
    // the lent one evicted while the other stays: its reborrow finds nothing, never the other
    const serial = read[1].serial;
    s.remove(std.mem.indexOfScalar(*pc.Entry, s.entries.items, read[1]).?);
    try t.expect(s.reborrow(serial) == null);
    try t.expectEqual(@as(u32, 0), read[0].borrowers);
}
