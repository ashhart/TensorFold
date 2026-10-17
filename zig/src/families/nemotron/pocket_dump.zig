//! A lesson's read rows written out for offline study (TF_SLIDE_POCKET_DUMP=dir): inputs, labels and examples.
const std = @import("std");
const platform = @import("sys");

/// `bytes` written to `path`.
fn put(path: [:0]const u8, bytes: []const u8) !void {
    const f = platform.c.fopen(path, "wb") orelse return error.DumpOpen;
    defer _ = platform.c.fclose(f);
    if (platform.c.fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.DumpWrite;
}

/// Pocket `p`'s read rows under `dir` for `tag`: bf16 [read, slots, D], then their labels and examples as text.
pub fn write(p: anytype, dir: []const u8, tag: usize, d: usize) !void {
    const a = p.gpa;
    var path: [1024]u8 = undefined;
    try put(try std.mem.printSentinel(&path, "{s}/fact-{d}-rows.bf16", .{ dir, tag }, 0), std.mem.sliceAsBytes(p.rows.items));
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.print(a, "{d} {d}", .{ p.moe.items.len, d });
    for (p.moe.items) |l| try text.print(a, " {d}", .{l});
    try text.append(a, '\n');
    var exs: usize = 0;
    for (p.labels.items) |l| {
        const ex = p.pickRead(l.ex).?.ex;
        const key: i32 = if (l.key) |k| @backingInt(k) else -1;
        try text.print(a, "{d} {d} {d} {d} {d} {d}\n", .{ @backingInt(l.kind), key, l.ex, l.row, ex.ids[l.row], if (l.row > 0) ex.ids[l.row - 1] else 0 });
        exs = @max(exs, l.ex + 1);
    }
    try put(try std.mem.printSentinel(&path, "{s}/fact-{d}-labels.txt", .{ dir, tag }, 0), text.items);
    text.clearRetainingCapacity();
    for (0..exs) |i| {
        const got = p.pickRead(i).?;
        const ex = got.ex;
        try text.print(a, "{d} {d} {d} {d} {d} {d}", .{ i, @backingInt(got.kind), ex.start, ex.decide, ex.subject, ex.value });
        for (ex.ids) |id| try text.print(a, " {d}", .{id});
        try text.append(a, '\n');
    }
    try put(try std.mem.printSentinel(&path, "{s}/fact-{d}-examples.txt", .{ dir, tag }, 0), text.items);
    std.log.info("slide pocket: the lesson's {d} read rows written to {s} for study", .{ p.labels.items.len, dir });
}
