//! A wrong family form is rejected before blank allocation, block reads or read-ahead payloads.
const std = @import("std");
const rows = @import("rows_store.zig");
const prefetch = @import("rows_prefetch.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;

fn whole(bytes: []const u8) !void {
    var form: u64 = undefined;
    @memcpy(std.mem.asBytes(&form), bytes);
    if (form != 1) return error.DiskKdaForm;
}

test "rows header: wrong form refuses before foreground blank and before read-ahead blocks" {
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(path);
    const dir = try a.dupeSentinel(u8, path, 0);
    defer a.free(dir);
    const name: rows.chain.Name = @splat(7);
    const block: rows.chain.Name = @splat(9);
    var writer = try rows.file.Writer.open(dir, &rows.chain.hex(name), ".st");
    errdefer writer.abort();
    const form = [_]u64{2};
    try writer.append(std.mem.asBytes(&form));
    _ = try writer.finish(.{ .kind = @backingInt(rows.file.Kind.state), .payload = 0, .payload_hash = 0, .name = name, .at = 32, .block = 256 }, &.{}, false);
    var family: fake.Fake = .{ .gpa = a };
    const snaps = family.snapshots();
    var img = snaps.vtable.rows.?.*;
    img.header = .{ .bytes = 8, .check = whole };
    var r: rows.Rows = .{ .gpa = a, .family = snaps, .img = &img, .root = path, .dir = dir, .keying = .{ .identity = 7, .block = 256, .lookahead = 0 }, .cap = 1 << 20, .daily = 1 << 20 };
    defer r.states.deinit(a);
    try r.states.append(a, .{ .name = name, .at = 32, .tokens = &.{}, .spans = &.{}, .starts = &.{}, .own = 8, .rows = 0, .used = 0, .shared = false });
    try t.expectError(error.DiskKdaForm, rows.read.read(&r, 0));
    try t.expectEqual(@as(usize, 0), family.blanks);
    var f: prefetch.Prefetch = .{ .gpa = a, .io = t.io, .limits = .{}, .dir = dir, .block = 256, .header = img.header };
    var pieces = [_]prefetch.Piece{
        .{ .name = block, .kind = .block, .at = 0, .len = 8, .first = 0, .end = 0, .off = 0 },
        .{ .name = name, .kind = .state, .at = 32, .len = 8, .first = 0, .end = 0, .off = 0 },
    };
    var rd: prefetch.Read = .{ .name = name, .since = 0, .plan = .{ .pieces = &pieces } };
    try t.expectError(error.DiskKdaForm, prefetch.readFiles(&f, &rd));
    f.header = null;
    try t.expectError(error.RowsMissing, prefetch.readFiles(&f, &rd));
}
