//! Forgetting one conversation: its states leave RAM and disk; others' and shared ones stay; unknown drops nothing.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const fixture = @import("rows_async_fixture.zig");
const Ask = @import("rows_ask.zig").Ask;
const t = std.testing;
const a = t.allocator;
const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .lookahead = 1 };

fn options(root: []const u8) rows.Options {
    return .{
        .sync_writes = true,
        .root = root,
        .identity = 854,
        .cap = 1 << 27,
        .floor = 0,
        .daily = 1 << 28,
        .lookahead = 1,
    };
}

fn keep(s: *pc.Store, f: *fake.Fake, r: *rows.Rows, prompt: []const u32, at: u32) !void {
    f.prompt = prompt;
    f.at = at;
    try t.expect(s.keep(prompt, at, null, &.{}, &.{}));
    while (rows.write.step(r)) {}
}

/// RAM states of `prompt`'s conversation (its prefixes), shared ones or not.
fn ram(st: *const pc.Store, prompt: []const u32, shared: bool) usize {
    var n: usize = 0;
    for (st.entries.items) |e| {
        if (e.shared == shared and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) n += 1;
    }
    return n;
}

fn count(states: []const rows.State, prompt: []const u32) usize {
    var n: usize = 0;
    for (states) |st| n += @intFromBool(std.mem.eql(u32, st.tokens, prompt[0..st.tokens.len]));
    return n;
}

fn drop(s: *pc.Store, prompt: []const u32) Ask {
    var named: Ask = .{ .op = .drop, .prompt = prompt };
    named.answer(s, .{});
    return named;
}

test "drop: a conversation's states leave RAM and disk; another's and a shared cut stay; an unknown drops none" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var x: [3000]u32 = undefined;
    var y: [3000]u32 = undefined;
    var z: [3000]u32 = undefined;
    for (&x, &y, &z, 0..) |*p, *q, *w, i| {
        p.* = @intCast(31 + i % 977);
        q.* = @intCast(5000 + i % 977); // another conversation from its first token
        w.* = @intCast(9000 + i % 977); // one the server never saw
    }
    var f: fake.Fake = .{ .gpa = a, .prompt = &x };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    try keep(&s, &f, r, &x, 1000);
    try keep(&s, &f, r, &x, 2000);
    try keep(&s, &f, r, &y, 1000);
    try keep(&s, &f, r, &y, 2000);
    try t.expectEqual(@as(usize, 4), r.states.items.len);
    const ram_y = ram(&s, &y, false);
    try t.expect(ram_y > 0);
    // x's first state stands for a shared system cut: other conversations resume it, so it stays
    for (s.entries.items) |e| if (e.at == 1000 and e.tokens[0] == x[0]) {
        e.shared = true;
    };
    for (r.states.items) |*st| if (st.at == 1000 and st.tokens[0] == x[0]) {
        st.shared = true;
    };
    const unknown = drop(&s, &z);
    try t.expect(unknown.done.load(.acquire));
    try t.expectEqual(@as(u32, 0), unknown.dropped + unknown.disk_dropped + unknown.shared);
    try t.expectEqual(@as(u64, 0), unknown.freed);
    try t.expectEqual(@as(usize, 4), r.states.items.len);
    const ram_all = s.entries.items.len;
    const ram_x = ram(&s, &x, false);
    try t.expect(ram_x > 0);
    // x named by a later turn's prompt (its states are prefixes of it)
    const later = try std.mem.concat(a, u32, &.{ &x, &.{ 1, 2, 3 } });
    defer a.free(later);
    const named = drop(&s, later);
    try t.expectEqual(@as(u32, @intCast(ram_x)), named.dropped); // its own states; the shared cut stays
    try t.expect(named.ram_freed > 0);
    try t.expectEqual(@as(usize, 0), ram(&s, &x, false));
    try t.expectEqual(ram_y, ram(&s, &y, false)); // the other conversation untouched in RAM
    try t.expectEqual(@as(u32, 1), named.disk_dropped);
    try t.expectEqual(@as(u32, 2), named.shared); // the shared cut, in RAM and on disk
    try t.expect(named.freed > 0);
    try t.expectEqual(ram_all - ram_x, s.entries.items.len);
    try t.expectEqual(@as(usize, 3), r.states.items.len);
    try t.expectEqual(@as(usize, 2), count(r.states.items, &y)); // the other conversation untouched on disk
    try t.expectEqual(@as(usize, 1), count(r.states.items, &x)); // only x's shared cut
    try t.expectEqual(@as(u32, 1000), named.out.ram); // x would resume the shared cut now, no further
    // naming a prefix of y's states (its first turn) drops the states that extend it
    const first = drop(&s, y[0..1500]);
    try t.expectEqual(@as(u32, @intCast(ram_y)), first.dropped);
    try t.expectEqual(@as(u32, 2), first.disk_dropped);
    try t.expectEqual(@as(usize, 0), count(r.states.items, &y));
    // the bodiless route's ask keeps today's meaning: every disk state goes, RAM keeps what it holds
    const left = s.entries.items.len;
    var all: Ask = .{ .op = .forget };
    all.answer(&s, .{});
    try t.expect(all.freed > 0);
    try t.expectEqual(@as(usize, 0), r.states.items.len);
    try t.expectEqual(left, s.entries.items.len);
}

test "drop: a prefix another conversation continues stays until that conversation is forgotten too" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(root);
    var x: [3000]u32 = undefined;
    var w: [3000]u32 = undefined;
    for (&x, &w, 0..) |*p, *q, i| {
        p.* = @intCast(31 + i % 977);
        q.* = if (i < 1200) p.* else @intCast(7000 + i % 977); // the same first 1,200 tokens (a long shared document)
    }
    var f: fake.Fake = .{ .gpa = a, .prompt = &x };
    const r = try rows.Rows.open(a, f.snapshots(), options(root));
    defer r.deinit();
    var s = pc.Store.init(a, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    try keep(&s, &f, r, &x, 1000);
    try keep(&s, &f, r, &x, 2000);
    try keep(&s, &f, r, &w, 1500);
    try t.expectEqual(@as(usize, 3), r.states.items.len);
    // x forgotten: its state past the common part goes; the one at 1,000, which w's continues, stays
    const one = drop(&s, &x);
    try t.expectEqual(@as(u32, 1), one.dropped);
    try t.expectEqual(@as(u32, 1), one.disk_dropped);
    try t.expectEqual(@as(u32, 2), one.shared);
    try t.expectEqual(@as(usize, 2), r.states.items.len);
    try t.expectEqual(@as(u32, 1500), s.find(&w, &.{}, &.{}).?.at); // w still resumes where it did
    try t.expectEqual(@as(u32, 1000), one.out.ram);
    // w forgotten as well: nothing builds on the common state now, so it goes with w's own
    const two = drop(&s, &w);
    try t.expectEqual(@as(u32, 2), two.dropped);
    try t.expectEqual(@as(u32, 2), two.disk_dropped);
    try t.expectEqual(@as(u32, 0), two.shared);
    try t.expectEqual(@as(usize, 0), r.states.items.len);
    try t.expectEqual(@as(usize, 0), s.entries.items.len);
    try t.expectEqual(@as(usize, 0), fixture.files(r.dir, ".st"));
    try t.expectEqual(@as(usize, 0), fixture.files(r.dir, ".blk"));
}

fn open(g: *fixture.Gate) void {
    std.Io.sleep(t.io, .fromMilliseconds(30), .awake) catch {};
    g.release.store(true, .release);
}

test "drop: a write the I/O thread holds lands first, so another conversation's save stays and its own goes" {
    const x = try fixture.Fixture.open("ask-drop", .off, false, 1 << 30);
    defer x.close();
    var hold: fixture.Gate = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const mine = fixture.prompt(21);
    const theirs = fixture.prompt(22);
    try x.keep(&theirs, 600); // its write waits at the gate while the drop runs
    try fixture.Gate.wait(&hold.entered);
    try x.keep(&mine, 600);
    const opener = try std.Thread.spawn(.{}, open, .{&hold});
    const named = drop(&x.s, &mine);
    opener.join();
    try x.settle();
    try t.expectEqual(@as(u32, 1), named.dropped);
    try t.expectEqual(@as(usize, 1), x.r.states.items.len); // a fence would have cancelled the other's write too
    try t.expectEqualSlices(u32, theirs[0..601], x.r.states.items[0].tokens);
    try t.expectEqual(@as(usize, 1), fixture.files(x.r.dir, ".st"));
    try t.expectEqual(@as(usize, 1), x.s.entries.items.len);
}
