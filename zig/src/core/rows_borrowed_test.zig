//! Borrowed keeps retain live rows across permit waits and stream images larger than the whole permit.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fixture = @import("rows_async_fixture.zig");
const fake = @import("rows_fake.zig");
const waits = @import("rows_test_wait.zig");
const t = std.testing;
const notes = @import("prompt_cache_note.zig");

fn transient(_: *anyopaque, _: pc.Saved) bool {
    return true;
}

fn restore(x: *fixture.Fixture, tokens: []const u32, at: u32) !void {
    return restored(x, tokens, at) catch |err| {
        std.debug.print("borrowed restore at {d}: {s}; {d} states, {d} writes, {d} skips\n", .{ at, @errorName(err), x.r.states.items.len, x.r.counts.writes, x.r.counts.skipped });
        return err;
    };
}

fn restored(x: *fixture.Fixture, tokens: []const u32, at: u32) !void {
    for (x.s.entries.items) |e| {
        x.s.expired(e.saved);
        const k: *fake.Kept = @ptrCast(@alignCast(e.saved));
        @memset(k.img, 0xa5);
    }
    x.s.collectExpired();
    try waits.published(x.r);
    const best = x.r.best(tokens, &.{}, &.{}, 0) orelse return error.MissingNewest;
    try t.expectEqual(at, x.r.states.items[best].at);
    x.f.gathers = 0;
    const plan = try x.s.begin(fixture.a, tokens, 0, &.{}, &.{}, null, &.{});
    defer fixture.a.free(plan.marks);
    try t.expectEqual(at, plan.from);
    _ = x.f.pass(&x.s, plan);
    try t.expect(x.f.restored_ok);
    try t.expectEqual(fake.fresh(tokens), x.f.sum);
}

const Keep = struct {
    x: *fixture.Fixture,
    tokens: []const u32,
    at: u32,
    idle_step: bool,
    started: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    err: ?anyerror = null,
    heard: notes.Heard = .{},

    fn run(k: *@This()) void {
        notes.sink = &k.heard;
        defer notes.sink = null;
        k.started.store(true, .release);
        defer k.done.store(true, .release);
        if (k.idle_step) {
            k.x.keep(k.tokens, k.at) catch |err| {
                k.err = err;
            };
            return;
        }
        // The transient image is admitted by Store.keep; an idle step could steal the newest save's queue entry.
        k.x.f.prompt = k.tokens;
        k.x.f.at = k.at;
        std.testing.expect(k.x.s.keep(k.tokens, k.at, null, k.x.f.starts, k.x.f.spans)) catch |err| {
            k.err = err;
        };
    }
};

test "borrowed spill: a newest keep waits for the earlier write and restores after its rows expire" {
    for ([_]bool{ false, true }) |oversized| for ([_]rows.Spill{ .off, .finish, .evicted }) |spill| {
        const x = try fixture.Fixture.open("borrowed-busy", spill, false, 9 << 20);
        defer x.close();
        var image = x.r.img.*;
        image.transient = transient;
        x.r.img = &image;
        var hold: fixture.Gate = .{ .hold_ns = 10 * std.time.ns_per_s };
        defer hold.release.store(true, .release);
        x.r.file_io = hold.io();
        const tokens = fixture.prompt(73);
        var earlier: Keep = .{ .x = x, .tokens = &tokens, .at = 900, .idle_step = false };
        const earlier_thread = try std.Thread.spawn(.{}, Keep.run, .{&earlier});
        defer {
            hold.release.store(true, .release);
            earlier_thread.join();
        }
        const earlier_started = fixture.Gate.wait(&earlier.started);
        try fixture.Gate.wait(&hold.entered);
        const used = x.r.worker.?.bytes();
        try t.expect(used > 0);
        try t.expect(x.r.worker.?.pending() > 0);
        try t.expect(!hold.left.load(.acquire));
        // One image fills the permit; a shared prefix can stage later, while a cold image needs the fallback.
        const limit = used;
        x.r.worker.?.limit = limit;
        const next = if (oversized) fixture.prompt(76) else tokens;
        var keep: Keep = .{ .x = x, .tokens = &next, .at = 1000, .idle_step = true };
        const thread = try std.Thread.spawn(.{}, Keep.run, .{&keep});
        defer {
            hold.release.store(true, .release);
            thread.join();
        }
        const started = fixture.Gate.wait(&keep.started);
        // Only a broken keep can return while its predecessor still owns the permit.
        const until = rows.write.nowNs() + 100 * std.time.ns_per_ms;
        while (!keep.done.load(.acquire) and rows.write.nowNs() < until) rows.write.io.nap();
        const premature = keep.done.load(.acquire);
        try t.expect(!hold.left.load(.acquire));
        hold.release.store(true, .release);
        try fixture.Gate.wait(&earlier.done);
        try fixture.Gate.wait(&keep.done);
        try earlier_started;
        try started;
        if (earlier.err) |err| return err;
        if (keep.err) |err| return err;
        try t.expectEqual(false, premature);
        const reason = if (oversized) "StagingFull" else "StagingBusy";
        try t.expectEqual(@as(usize, 1), std.mem.count(u8, keep.heard.text(), reason));
        try t.expect(std.mem.indexOf(u8, keep.heard.text(), "1000 borrowed tokens") != null);
        const action: []const u8 = if (oversized) "streaming bounded pieces" else "waiting for staging permit";
        try t.expect(std.mem.indexOf(u8, keep.heard.text(), action) != null);
        try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
        try t.expectEqual(oversized, x.r.stage.len > 0);
        try restore(x, &next, 1000);
        try t.expectEqual(@as(u64, 2), x.r.counts.writes);
        try t.expectEqual(limit, x.r.worker.?.limit);
    };
}

test "borrowed spill: an image larger than the whole permit streams before expiry" {
    for ([_]bool{ false, true }) |sync| for ([_]rows.Spill{ .off, .finish, .evicted }) |spill| {
        const x = try fixture.Fixture.open("borrowed-big", spill, sync, 1024);
        defer x.close();
        var image = x.r.img.*;
        image.transient = transient;
        x.r.img = &image;
        var tokens: [2200]u32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = @intCast(i + 74);
        try x.keep(&tokens, 2000);
        try t.expectEqual(@as(u64, 1024), x.r.staging_bytes);
        try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
        try t.expectEqual(@as(u64, 1), x.r.counts.writes);
        // The fallback owns a piece, never a permit-sized arena of the complete image.
        try t.expect(x.r.stage.len < x.s.entries.items[0].bytes);
        if (x.r.worker) |w| try t.expectEqual(@as(u64, 0), w.bytes());
        try restore(x, &tokens, 2000);
    };
}

test "borrowed spill: a nontransient family still defers an eviction-only save" {
    const x = try fixture.Fixture.open("borrowed-ordinary", .evicted, false, 9 << 20);
    defer x.close();
    const tokens = fixture.prompt(75);
    try x.keep(&tokens, 1000);
    try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
    try t.expectEqual(@as(usize, 0), x.f.gathered);
    try t.expect(x.r.worker == null);
    x.s.remove(0);
    try x.settle();
    try t.expectEqual(@as(u64, 1), x.r.counts.writes);
}
