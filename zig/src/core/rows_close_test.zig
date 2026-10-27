//! Close-time spill uses the eviction image, capacity rules and bounded writer with RAM snapshots still alive.
const std = @import("std");
const rows = @import("rows_store.zig");
const fx = @import("rows_async_fixture.zig");
const Fixture = fx.Fixture;

fn drain(x: *Fixture) usize {
    return rows.write.drain(x.r, rows.write.nowNs() + 3 * std.time.ns_per_s);
}

test "rows close: spill-only flushes more than eight RAM states newest first without ordinary idle writes" {
    const x = try Fixture.open("close-order", .evicted, false, 1 << 20);
    defer x.close();
    var tokens: [12][1100]u32 = undefined;
    for (&tokens, 0..) |*p, i| {
        p.* = fx.prompt(@intCast(100 + i));
        try x.keep(p, 600);
    }
    try std.testing.expect(!rows.write.pending(x.r) and !rows.write.ready(x.r) and !rows.write.step(x.r));
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), drain(x));
    try std.testing.expectEqual(@as(usize, 12), x.r.states.items.len);
    try std.testing.expectEqual(@as(u64, 0), x.r.counts.skipped);
    for (x.r.states.items, 0..) |s, i| try std.testing.expectEqualSlices(u32, tokens[11 - i][0..601], s.tokens);
}

test "rows close: planned snapshots have eviction file bytes and restore the same image" {
    const closed = try Fixture.open("close-bytes", .evicted, false, 1 << 20);
    defer closed.close();
    const evicted = try Fixture.open("close-eviction", .evicted, false, 1 << 20);
    defer evicted.close();
    const tokens = fx.prompt(120);
    for ([_]*Fixture{ closed, evicted }) |x| {
        x.f.planned = true;
        x.f.starts = &.{ 0, 256, 512, 768, 900 };
        x.f.spans = &.{.{ 400, 450 }};
        x.s.rules.planned = true;
        x.r.keying.planned = true;
        try x.keep(&tokens, 900);
    }
    evicted.s.remove(0);
    try evicted.settle();
    try std.testing.expectEqual(@as(usize, 0), drain(closed));
    try std.testing.expectEqual(@as(usize, 1), closed.r.states.items.len);
    const name = closed.r.states.items[0].name;
    const state = try std.fmt.allocPrint(fx.a, "{s}.st", .{rows.chain.hex(name)});
    defer fx.a.free(state);
    try std.testing.expectEqual(try fx.digest(evicted.r.dir, state), try fx.digest(closed.r.dir, state));
    var it = closed.r.blocks.keyIterator();
    while (it.next()) |stem| {
        const block = try std.fmt.allocPrint(fx.a, "{s}.blk", .{rows.chain.hex(stem.*)});
        defer fx.a.free(block);
        try std.testing.expectEqual(try fx.digest(evicted.r.dir, block), try fx.digest(closed.r.dir, block));
    }
    const saved = try rows.read.read(closed.r, 0);
    defer closed.f.snapshots().vtable.drop(&closed.f, saved);
    try closed.f.snapshots().vtable.restore(&closed.f, null, saved);
    try std.testing.expect(closed.f.restored_ok);
}

test "rows close: the cap keeps the newest RAM state instead of replacing it with an older flush" {
    try capped(.evicted, false);
    try capped(.finish, false);
    try capped(.evicted, true);
    try capped(.finish, true);
}

fn capped(spill: rows.Spill, synchronous: bool) !void {
    const x = try Fixture.open("close-cap", spill, synchronous, 1 << 20);
    defer x.close();
    const old = fx.prompt(130);
    try x.keep(&old, 600);
    x.s.remove(0);
    try x.settle();
    x.r.cap = x.r.held();
    const middle = fx.prompt(131);
    const newest = fx.prompt(132);
    for ([_]*const [1100]u32{ &middle, &newest }) |p| {
        x.f.prompt = p;
        try std.testing.expect(x.s.keep(p, 600, null, &.{}, &.{}));
    }
    _ = drain(x);
    try std.testing.expect(x.r.held() <= x.r.cap);
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expect(x.r.best(&newest, &.{}, &.{}, 0) != null);
    try std.testing.expect(x.r.best(&middle, &.{}, &.{}, 0) == null);
    try std.testing.expect(x.r.best(&old, &.{}, &.{}, 0) == null);
    const reopened = try rows.Rows.open(fx.a, x.f.snapshots(), .{ .root = x.root, .identity = 7, .cap = x.r.cap, .daily = 1 << 30, .floor = 0, .lookahead = 1, .spill = .evicted });
    defer reopened.deinit();
    try std.testing.expect(reopened.held() <= x.r.cap);
    try std.testing.expectEqual(@as(usize, 1), reopened.states.items.len);
    try std.testing.expect(reopened.best(&newest, &.{}, &.{}, 0) != null);
}

test "rows close: finishing an earlier eviction leaves cap space available to the newer RAM conversation" {
    const x = try Fixture.open("close-pending", .evicted, false, 1 << 20);
    defer x.close();
    const old = fx.prompt(135);
    try x.keep(&old, 600);
    x.s.remove(0);
    const w = x.r.worker.?;
    const end = rows.write.nowNs() + 3 * std.time.ns_per_s;
    while (!w.ready() and rows.write.nowNs() < end) rows.write.io.nap();
    try std.testing.expect(w.ready());
    x.r.cap = w.reserved();
    const newest = fx.prompt(136);
    try x.keep(&newest, 600);
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), drain(x));
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expect(x.r.best(&newest, &.{}, &.{}, 0) != null);
    try std.testing.expect(x.r.held() <= x.r.cap);
}

test "rows close: a blocked spill-only flush cannot hold shutdown past its deadline" {
    const x = try Fixture.open("close-deadline", .evicted, false, 1 << 20);
    defer x.close();
    var gate: fx.Gate = .{};
    defer gate.release.store(true, .release);
    x.r.file_io = gate.io();
    const tokens = fx.prompt(140);
    try x.keep(&tokens, 600);
    const next = fx.prompt(141);
    try x.keep(&next, 600);
    const began = rows.write.nowNs();
    const left = rows.write.drain(x.r, began + 20 * std.time.ns_per_ms);
    const elapsed = rows.write.nowNs() - began;
    const entered = gate.entered.load(.acquire);
    gate.release.store(true, .release);
    if (entered) try fx.Gate.wait(&gate.left);
    try std.testing.expect(entered);
    try std.testing.expectEqual(@as(usize, 2), left);
    try std.testing.expect(elapsed < std.time.ns_per_s);
    try std.testing.expect(x.r.stopped and x.r.worker == null);
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(u64, 0), x.r.admission.reserved);
}

test "rows close: an expired deadline starts no spill work" {
    const x = try Fixture.open("close-expired", .evicted, false, 1 << 20);
    defer x.close();
    const tokens = fx.prompt(150);
    try x.keep(&tokens, 600);
    try std.testing.expectEqual(@as(usize, 1), rows.write.drain(x.r, rows.write.nowNs()));
    try std.testing.expect(x.r.worker == null);
    try std.testing.expectEqual(@as(usize, 0), fx.files(x.r.dir, ".st"));
}

test "rows close: without spill the learned-state queue and eviction behavior are unchanged" {
    const x = try Fixture.open("close-off", .off, false, 1 << 20);
    defer x.close();
    const old = fx.prompt(160);
    x.f.prompt = &old;
    x.f.at = 600;
    try std.testing.expect(x.s.keep(&old, 600, null, &.{}, &.{}));
    x.s.remove(0);
    try std.testing.expect(x.r.worker == null and x.r.queue.items.len == 0);
    const kept = fx.prompt(161);
    x.f.prompt = &kept;
    try std.testing.expect(x.s.keep(&kept, 600, null, &.{}, &.{}));
    try std.testing.expect(rows.write.pending(x.r));
    try std.testing.expectEqual(@as(usize, 0), drain(x));
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expect(x.r.best(&kept, &.{}, &.{}, 0) != null);
    try std.testing.expect(x.r.best(&old, &.{}, &.{}, 0) == null);
}
