//! Persisted selections restore the exact host image and retain RAM on read failure.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };
fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 169, .cap = 1 << 26, .floor = 0, .daily = 1 << 28, .lookahead = 1 };
}
fn seed(root: []const u8, prompt: []const u32) !void {
    var f: fake.Fake = .{ .gpa = a, .prompt = prompt };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    for ([_]u32{ 1400, 2600 }) |at| {
        f.at = at;
        try t.expect(s.keep(prompt, at, null, &.{}, &.{}));
        while (rows.write.step(r)) {}
    }
    try t.expectEqual(@as(usize, 2), r.states.items.len);
}
test "restore selection: a restart under a reduced budget reads the smaller state and equals fresh" {
    for ([_]bool{ false, true }) |tight| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        defer a.free(dir);
        var prompt: [3000]u32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(17 + i * 3);
        try seed(dir, &prompt);
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
        const family = f.snapshots();
        const r = try rows.Rows.open(a, family, options(dir));
        defer r.deinit();
        var s = pc.Store.init(a, family, rules, if (tight) family.vtable.bytes(family.ptr, 1400) else 1 << 30);
        defer s.deinit();
        s.disk = r;
        const status = rows.read.resumable(&s, &prompt, &.{}, &.{});
        const wanted: u32 = if (tight) 1400 else 2600;
        try t.expect(status.from == .disk and status.disk == wanted);
        const plan = try s.begin(a, &prompt, 0, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try t.expectEqual(wanted, plan.from);
        try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
        try t.expect(f.restored_ok and s.held <= s.budget);
        try t.expectEqual(@as(u64, 1), r.counts.reads);
    }
}
test "restore selection: corrupt files and allocation refusal keep the valid RAM state" {
    for ([_]bool{ false, true }) |corrupt| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        defer a.free(dir);
        var prompt: [3000]u32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(23 + i * 11);
        try seed(dir, &prompt);
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt, .at = 1400 };
        const family = f.snapshots();
        const r = try rows.Rows.open(a, family, options(dir));
        defer r.deinit();
        var s = pc.Store.init(a, family, rules, 1 << 30);
        defer s.deinit();
        try t.expect(s.keep(&prompt, 1400, null, &.{}, &.{}));
        s.disk = r;
        var image = r.img.*;
        if (corrupt) {
            const index = r.best(&prompt, &.{}, &.{}, 0).?;
            var buf: [1200]u8 = undefined;
            const path = try rows.file.path(&buf, r.dir, &rows.chain.hex(r.states.items[index].name), ".st", false);
            const fd = platform.c.open(path, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
            try t.expect(fd >= 0);
            defer _ = platform.c.close(fd);
            var byte: [1]u8 = undefined;
            try t.expectEqual(@as(isize, 1), platform.c.pread(fd, &byte, 1, 0));
            byte[0] ^= 128;
            try t.expectEqual(@as(isize, 1), platform.c.pwrite(fd, &byte, 1, 0));
        } else {
            image.blank = struct {
                fn fail(_: *anyopaque, _: u32) anyerror!pc.Saved {
                    return error.OutOfMemory;
                }
            }.fail;
            r.img = &image;
        }
        try t.expect(rows.read.resumable(&s, &prompt, &.{}, &.{}).from == .disk);
        const plan = try s.begin(a, &prompt, 0, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try t.expectEqual(@as(u32, 1400), plan.from);
        try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
        try t.expect(f.restored_ok);
        try t.expectEqual(@as(u64, 0), r.counts.reads);
        try t.expectEqual(@as(u64, @intFromBool(corrupt)), r.counts.failed);
        try t.expectEqual(@as(usize, 1), f.live);
        try t.expectEqual(@as(u32, 0), s.entries.items[0].borrowers);
    }
}
test "restore selection: a state whose directory another identity's cap removed evicts no kept state" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    var prompt: [3000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(29 + i * 7);
    try seed(dir, &prompt);
    var kept: [1200]u32 = undefined;
    for (&kept, 0..) |*token, i| token.* = @intCast(31 + i * 5);
    var f: fake.Fake = .{ .gpa = a, .prompt = &kept, .at = 1000 };
    const family = f.snapshots();
    const r = try rows.Rows.open(a, family, options(dir));
    defer r.deinit();
    var s = pc.Store.init(a, family, rules, family.vtable.bytes(family.ptr, 2600));
    defer s.deinit();
    try t.expect(s.keep(&kept, 1000, null, &.{}, &.{}));
    s.disk = r;
    f.prompt = &prompt;
    const status = rows.read.resumable(&s, &prompt, &.{}, &.{});
    try t.expect(status.from == .disk and status.disk == 2600);
    var o = options(dir);
    o.identity = 170;
    const other = try rows.Rows.open(a, family, o);
    defer other.deinit();
    _ = other.room(o.cap);
    try t.expect(rows.read.recall(&s, &prompt, prompt.len, &.{}, &.{}, null) == null);
    try t.expectEqual(@as(u64, 0), s.counts.evicted);
    try t.expectEqual(@as(usize, 1), s.entries.items.len);
    try t.expectEqual(@as(usize, 1), f.live);
    try t.expectEqual(@as(u64, 1), r.counts.failed);
}
