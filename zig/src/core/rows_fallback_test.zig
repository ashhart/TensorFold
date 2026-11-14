//! A multi-turn chain's disk states, with its longest files damaged after catalog opening.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const note = @import("prompt_cache_note.zig");
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };
fn options(root: []const u8) rows.Options {
    return .{ .sync_writes = true, .root = root, .identity = 322, .cap = 1 << 26, .floor = 0, .daily = 1 << 28, .lookahead = 1 };
}
fn seed(root: []const u8, prompt: []const u32) !void {
    var f: fake.Fake = .{ .gpa = a, .prompt = prompt };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    for ([_]u32{ 1000, 2000, 3000, 4000 }) |at| {
        f.at = at;
        try t.expect(s.keep(prompt, at, null, &.{}, &.{}));
        while (rows.write.step(r)) {}
    }
    try t.expectEqual(@as(usize, 4), r.states.items.len);
}
fn damage(r: *rows.Rows, at: u32) !void {
    const st = for (r.states.items) |s| {
        if (s.at == at) break s;
    } else return error.NoState;
    var buf: [1200]u8 = undefined;
    const path = try rows.file.path(&buf, r.dir, &rows.chain.hex(st.name), ".st", false);
    const fd = platform.c.open(path, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try t.expect(fd >= 0);
    defer _ = platform.c.close(fd);
    var byte: [1]u8 = undefined;
    try t.expectEqual(@as(isize, 1), platform.c.pread(fd, &byte, 1, 0));
    byte[0] ^= 128;
    try t.expectEqual(@as(isize, 1), platform.c.pwrite(fd, &byte, 1, 0));
}
test "restore fallback: one shorter disk read, then zero even when a third state remains" {
    for (0..3) |bad| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        defer a.free(root);
        var prompt: [5000]u32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(17 + i % 997);
        try seed(root, &prompt);
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
        const r = try rows.Rows.open(a, f.snapshots(), options(root));
        defer r.deinit();
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        var eligible: usize = 0;
        for (r.states.items) |*st| eligible += @intFromBool(r.matches(st, &prompt, &.{}, &.{}));
        for (0..bad) |i| try damage(r, @intCast(4000 - 1000 * i));
        const plan = try s.begin(a, &prompt, 4500, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
        try t.expectEqual(@as(usize, 4), eligible);
        try t.expectEqual(@as(u32, if (bad == 0) 4000 else if (bad == 1) 3000 else 0), plan.from);
        try t.expectEqual(@as(u64, if (bad < 2) 1 else 0), r.counts.reads);
        try t.expectEqual(@as(u64, @intCast(bad)), r.counts.failed);
        try t.expect(f.restored_ok);
        for (s.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers);
    }
}

test "restore fallback: a size refusal tries only the next position, with files retained" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [5000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(23 + i % 991);
    try seed(root, &prompt);
    for ([_]bool{ false, true }) |both| {
        var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
        const r = try rows.Rows.open(a, f.snapshots(), options(root));
        defer r.deinit();
        var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
        defer s.deinit();
        s.disk = r;
        const Refuse = struct {
            var limit: u32 = 4000;
            var attempted: [2]u32 = undefined;
            var n: usize = 0;
            fn blank(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
                if (n == attempted.len) return error.UnboundedRestoreRetry;
                attempted[n] = at;
                n += 1;
                if (at >= limit) return error.SnapshotSize;
                return fake.Fake.snapshots(@ptrCast(@alignCast(ptr))).vtable.rows.?.blank(ptr, at);
            }
        };
        Refuse.limit = if (both) 3000 else 4000;
        Refuse.n = 0;
        var image = r.img.*;
        image.blank = Refuse.blank;
        r.img = &image;
        const plan = try s.begin(a, &prompt, 0, &.{}, &.{}, null, &.{});
        defer a.free(plan.marks);
        try t.expectEqual(@as(u32, if (both) 0 else 3000), plan.from);
        try t.expectEqualSlices(u32, &.{ 4000, 3000 }, &Refuse.attempted);
        try t.expectEqual(@as(usize, 2), Refuse.n);
        try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
        try t.expectEqual(@as(usize, 4), r.states.items.len);
        try t.expectEqual(@as(u64, 0), r.counts.failed);
    }
}

test "restore fallback: the next compatible position wins over a cheaper shorter read" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [5000]u32 = @splat(11);
    try seed(root, &prompt);
    var f: fake.Fake = .{ .gpa = a, .prompt = &prompt };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    r.prefill_rate = 1000;
    r.read_rate = 1e9;
    for (r.states.items) |*state| if (state.at == 3000) {
        state.own = 8_000_000_000;
    };
    const choice = @import("rows_select.zig").shorter(&s, &prompt, &.{}, &.{}, null, 4000);
    try t.expectEqual(@as(u32, 3000), r.states.items[choice.restore.?].at);
    prompt[0] = 12;
    try t.expect(@import("rows_select.zig").shorter(&s, &prompt, &.{}, &.{}, null, 4000).restore == null);
}

test "restore fallback: two failed disk reads leave the lookup the shorter state kept in RAM" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [5000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(31 + i % 983);
    try seed(root, &prompt);
    var f: fake.Fake = .{ .gpa = a, .prompt = &prompt, .at = 500 };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    try t.expect(s.keep(&prompt, 500, null, &.{}, &.{})); // in RAM only, shorter than every state on disk
    s.disk = r;
    try damage(r, 4000);
    try damage(r, 3000);
    var heard: note.Heard = .{};
    note.sink = &heard;
    defer note.sink = null;
    const plan = try s.begin(a, &prompt, 4500, &.{}, &.{}, null, &.{});
    defer a.free(plan.marks);
    try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
    try t.expectEqual(@as(u32, 500), plan.from); // not 0: the RAM state still stands after the bounded retry
    try t.expectEqual(@as(u64, 2), r.counts.failed);
    try t.expectEqual(@as(u64, 0), r.counts.reads);
    try t.expect(std.mem.indexOf(u8, heard.text(), "disk restore of 3000 tokens failed (") != null);
    try t.expect(std.mem.endsWith(u8, heard.text(), "); resuming the shorter state kept in RAM\n"));
    try t.expect(f.restored_ok);
    for (s.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers);
}

test "restore fallback: two state files gone under a tight budget leave the lookup the shorter state kept in RAM" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var prompt: [5000]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(37 + i % 977);
    var other: [3000]u32 = @splat(5);
    try seed(root, &prompt);
    var f: fake.Fake = .{ .gpa = a, .prompt = &other, .at = 2000 };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    try t.expect(s.keep(&other, 2000, null, &.{}, &.{})); // another prompt's state: room a restore could evict into
    f.prompt = &prompt;
    f.at = 500;
    try t.expect(s.keep(&prompt, 500, null, &.{}, &.{})); // in RAM only, shorter than every state on disk
    s.budget = s.held + r.blankBytes(3000) - 1; // each restore fits after an eviction, so ready() is false for both
    s.disk = r;
    for (r.states.items) |st| if (st.at >= 3000) rows.unlinkName(r.dir, st.name, ".st");
    var heard: note.Heard = .{};
    note.sink = &heard;
    defer note.sink = null;
    const plan = try s.begin(a, &prompt, 4500, &.{}, &.{}, null, &.{});
    defer a.free(plan.marks);
    try t.expectEqual(@as(u32, 500), plan.from); // not 0: the RAM state still stands after the bounded retry
    try t.expectEqual(@as(usize, 0), f.blanks); // no read began: both failures came from present()
    try t.expectEqual(@as(u64, 2), r.counts.failed);
    try t.expectEqual(@as(u64, 0), r.counts.reads);
    try t.expectEqual(@as(usize, 2), r.states.items.len);
    try t.expect(std.mem.indexOf(u8, heard.text(), "a disk state failed its check") != null);
    try t.expect(std.mem.indexOf(u8, heard.text(), "disk restore of 4000 tokens failed (") != null);
    try t.expect(std.mem.indexOf(u8, heard.text(), "disk restore of 3000 tokens failed (") != null);
    try t.expect(std.mem.endsWith(u8, heard.text(), "); resuming the shorter state kept in RAM\n"));
    try t.expectEqual(fake.fresh(&prompt), f.pass(&s, plan));
    try t.expect(f.restored_ok);
    for (s.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers);
}
