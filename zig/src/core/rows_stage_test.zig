//! Staging: a block still being written is charged and gathered once, and a save that fits the permit waits for it.
const std = @import("std");
const waits = @import("rows_test_wait.zig");
const rows = @import("rows_store.zig");
const fixture = @import("rows_async_fixture.zig");
const Fixture = fixture.Fixture;
const Hold = fixture.Gate;
const t = std.testing;

/// A prompt that shares its first 513 tokens with `fixture.prompt(17)`, so their first two blocks are the same files.
fn sibling(salt: u32) [1100]u32 {
    var p = fixture.prompt(salt);
    const base = fixture.prompt(17);
    @memcpy(p[0..513], base[0..513]);
    return p;
}

fn pair() struct { first: [1100]u32, second: [1100]u32 } {
    return .{ .first = fixture.prompt(17), .second = sibling(18) };
}

/// Bounded staging streams on its caller; a save beside a held writer needs its own thread.
const Stager = struct {
    x: *Fixture,
    tokens: []const u32,
    at: u32,
    err: ?anyerror = null,
    evict: bool = false,

    fn run(s: *Stager) void {
        s.x.keep(s.tokens, s.at) catch |err| {
            s.err = err;
            return;
        };
        if (s.evict) s.x.s.remove(0);
    }
};

fn pendingAtLeast(r: *rows.Rows, n: usize) !void {
    const end = waits.deadline();
    while ((if (r.worker) |w| w.pending() else 0) < n) {
        if (rows.write.nowNs() >= end) return error.WriterTimedOut;
        rows.write.io.nap();
    }
}

test "rows stage: a block still being written is charged and gathered once" {
    const x = try Fixture.open("stage-share", .finish, false, 18 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const p = pair();
    var stage1 = Stager{ .x = x, .tokens = &p.first, .at = 900 };
    var stage2 = Stager{ .x = x, .tokens = &p.second, .at = 900 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try Hold.wait(&hold.entered);
        const thread2 = try std.Thread.spawn(.{}, Stager.run, .{&stage2});
        defer {
            hold.release.store(true, .release);
            thread2.join();
        }
        try pendingAtLeast(x.r, 2);
        try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
        const charged = x.r.worker.?.reserved();
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 2), x.r.states.items.len);
        try t.expectEqual(x.r.held(), charged);
        try t.expectEqual(x.r.blocks.count() + 2, x.f.gathered);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
    try t.expectEqual(@as(?anyerror, null), stage2.err);
    const saved = try rows.read.read(x.r, 1);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows stage: a state whose earlier job failed to write its block is not published" {
    const x = try Fixture.open("stage-lost", .finish, false, 18 << 20);
    defer x.close();
    // A two-chunk handoff backpressures a staged save while its writer is gated; stagers get threads.
    var failing: Hold = .{ .mode = .first_fails, .hold_ns = 10 * std.time.ns_per_s };
    defer failing.release.store(true, .release);
    var fast: Hold = .{ .release = .init(true) };
    x.r.file_io = failing.io();
    const p = pair();
    var stage1 = Stager{ .x = x, .tokens = &p.first, .at = 900 };
    const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
    defer thread1.join();
    try pendingAtLeast(x.r, 1);
    try Hold.wait(&failing.entered);
    x.r.worker.?.io = fast.io(); // job 1 keeps the failing gate; the next job writes through `fast`
    var stage2 = Stager{ .x = x, .tokens = &p.second, .at = 900 };
    const thread2 = try std.Thread.spawn(.{}, Stager.run, .{&stage2});
    defer thread2.join();
    try pendingAtLeast(x.r, 2); // job 2 was submitted while job 1 still promised the shared block
    failing.release.store(true, .release); // fail job 1 only after job 2 skipped that block
    try x.settle();
    try t.expectEqual(@as(usize, 0), x.r.states.items.len);
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".blk"));
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
    try t.expectEqual(@as(u64, 2), x.r.failures.count);
    try t.expectEqual(error.NoSpaceLeft, x.r.failures.records[0].?.err);
    try t.expectEqual(error.MissingBlock, x.r.failures.records[1].?.err);
    try t.expectEqual(@as(u64, 2), x.r.counts.failed);
    try t.expectEqual(@as(u64, 0), x.r.counts.writes);
    try t.expectEqual(@as(u64, 0), x.r.admission.reserved);
}

test "rows stage: a block its writer failed to write is written by the next save, though a job that skipped it is in flight" {
    const x = try Fixture.open("stage-chain", .finish, false, 18 << 20);
    defer x.close();
    var failing: Hold = .{ .mode = .first_fails, .hold_ns = 10 * std.time.ns_per_s };
    defer failing.release.store(true, .release);
    var held: Hold = .{ .hold_ns = 10 * std.time.ns_per_s };
    defer held.release.store(true, .release);
    var fast: Hold = .{ .release = .init(true) };
    x.r.file_io = failing.io();
    const first = fixture.prompt(17);
    const second = sibling(18);
    const third = sibling(19);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 900 };
    const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
    defer thread1.join();
    try pendingAtLeast(x.r, 1);
    try Hold.wait(&failing.entered);
    x.r.worker.?.io = fast.io();
    var stage2 = Stager{ .x = x, .tokens = &second, .at = 900 };
    const thread2 = try std.Thread.spawn(.{}, Stager.run, .{&stage2});
    defer thread2.join();
    try pendingAtLeast(x.r, 2);
    x.r.worker.?.io = held.io(); // the skipped job's files hold this gate in flight
    failing.release.store(true, .release);
    try Hold.wait(&held.entered);
    const end = waits.deadline();
    while (x.r.failures.count < 1 and rows.write.nowNs() < end) {
        rows.write.reap(x.r);
        rows.write.io.nap();
    }
    x.r.admission.retry_at = 0; // the back-off the first failure set has run out
    try t.expectEqual(@as(usize, 1), x.r.worker.?.pending());
    x.r.worker.?.io = fast.io(); // the next save writes while the skipped job is still in flight
    var stage3 = Stager{ .x = x, .tokens = &third, .at = 900 };
    const thread3 = try std.Thread.spawn(.{}, Stager.run, .{&stage3});
    defer thread3.join();
    try pendingAtLeast(x.r, 2);
    held.release.store(true, .release);
    try x.settle();
    try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    try t.expectEqual(error.NoSpaceLeft, x.r.failures.records[0].?.err);
    try t.expectEqual(error.MissingBlock, x.r.failures.records[1].?.err);
    try t.expectEqual(@as(u64, 2), x.r.counts.failed);
    try t.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows stage: a job a fence cancelled promises no block, so the next save writes its own" {
    const x = try Fixture.open("stage-fence", .finish, false, 18 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const p = pair();
    var stage1 = Stager{ .x = x, .tokens = &p.first, .at = 900 };
    var stage2 = Stager{ .x = x, .tokens = &p.second, .at = 900 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try Hold.wait(&hold.entered);
        rows.write.fence(x.r);
        const thread2 = try std.Thread.spawn(.{}, Stager.run, .{&stage2});
        defer {
            hold.release.store(true, .release);
            thread2.join();
        }
        try pendingAtLeast(x.r, 2);
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
        try t.expectEqual(@as(u64, 1), x.r.counts.failed);
        try t.expectEqual(error.Cancelled, x.r.failures.records[0].?.err);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
    try t.expectEqual(@as(?anyerror, null), stage2.err);
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try t.expect(x.f.restored_ok);
}

test "rows stage: a fixed handoff already holding its whole permit skips, not waits" {
    const x = try Fixture.open("stage-big", .finish, false, 9 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(22);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        x.r.worker.?.limit = x.r.worker.?.bytes();
        const second = fixture.prompt(23);
        try x.keep(&second, 1000);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
        try t.expectEqual(@as(usize, 0), x.r.queue.items.len);
        try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
        try t.expectEqual(@as(usize, 1), x.r.worker.?.pending());
        try t.expect(!rows.write.ready(x.r));
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
}

test "rows stage: a permit sized to the longest state writes a cold save the fixed permit skips" {
    for ([_]u32{ 0, 1100, 1 << 28 }) |longest| {
        const x = try Fixture.sized("stage-longest", .finish, false, 4 << 10, longest);
        defer x.close();
        const cold = try rows.write.coldCharge(x.r, longest);
        if (longest > 1100) try t.expect(cold > 1 << 30);
        try t.expectEqual(@max(4 << 10, @min(cold, x.s.budget)), x.r.staging_bytes);
        var hold: Hold = .{};
        defer hold.release.store(true, .release);
        x.r.file_io = hold.io();
        const p = fixture.prompt(29);
        if (longest == 0) {
            try x.keep(&p, 1000);
            try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
            try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
            continue;
        }
        var stage1 = Stager{ .x = x, .tokens = &p, .at = 1000 };
        {
            const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
            defer {
                hold.release.store(true, .release);
                thread1.join();
            }
            try pendingAtLeast(x.r, 1);
            try Hold.wait(&hold.entered);
            try t.expect(x.r.worker.?.bytes() <= try rows.write.coldCharge(x.r, 1000)); // the bound holds the real charge
            hold.release.store(true, .release);
            try x.settle();
            try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
            try t.expectEqual(@as(usize, 1), x.r.states.items.len);
        }
        try t.expectEqual(@as(?anyerror, null), stage1.err);
        const saved = try rows.read.read(x.r, 0);
        defer x.f.snapshots().vtable.drop(&x.f, saved);
        try x.f.snapshots().vtable.restore(&x.f, null, saved);
        try t.expect(x.f.restored_ok);
        try t.expectEqual(@import("rows_fake.zig").fresh(&p), x.f.pass(&x.s, .{ .from = 1000 }));
        try t.expectEqual(@as(u32, p.len), x.f.at);
    }
}

test "rows stage: an eviction cannot wait, so a full permit skips the copy" {
    const x = try Fixture.open("stage-evict", .evicted, false, 9 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(24);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600, .evict = true };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        x.r.worker.?.limit = x.r.worker.?.bytes();
        const second = fixture.prompt(25);
        try x.keep(&second, 600);
        x.s.remove(0);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
        try t.expectEqual(@as(usize, 0), x.r.queue.items.len);
        try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
}

test "rows stage: a newer save that fits beside the jobs in flight is staged ahead of one that waits" {
    const x = try Fixture.open("stage-ahead", .finish, false, 18 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(26);
    const third = fixture.prompt(28);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
    var stage3 = Stager{ .x = x, .tokens = &third, .at = 300 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        const used = x.r.worker.?.bytes();
        x.r.worker.?.limit = 2 * used - 1; // room beside the first job for less than another like it
        const second = fixture.prompt(27);
        try x.keep(&second, 600);
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        try t.expect(!rows.write.ready(x.r));
        const thread3 = try std.Thread.spawn(.{}, Stager.run, .{&stage3});
        defer {
            hold.release.store(true, .release);
            thread3.join();
        }
        try pendingAtLeast(x.r, 2);
        try t.expectEqual(@as(usize, 2), x.r.worker.?.pending());
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
        hold.release.store(true, .release);
        try x.idle(3);
        try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
        try t.expectEqual(@as(usize, 0), x.r.queue.items.len);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
    try t.expectEqual(@as(?anyerror, null), stage3.err);
}

test "rows stage: a close writes the saves that waited for room" {
    for ([_]rows.Spill{ .off, .finish }) |spill| {
        const x = try Fixture.open("stage-close", spill, false, 9 << 20);
        defer x.close();
        var hold: Hold = .{};
        defer hold.release.store(true, .release);
        x.r.file_io = hold.io();
        const first = fixture.prompt(29);
        var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
        {
            const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
            defer {
                hold.release.store(true, .release);
                thread1.join();
            }
            try pendingAtLeast(x.r, 1);
            try Hold.wait(&hold.entered);
            x.r.worker.?.limit = x.r.worker.?.bytes();
            const second = fixture.prompt(30);
            try x.keep(&second, 600);
            try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
            hold.release.store(true, .release);
            try t.expectEqual(@as(usize, 0), rows.write.drain(x.r, waits.deadline()));
            try t.expectEqual(@as(usize, 2), x.r.states.items.len);
            try t.expectEqual(@as(u64, 0), x.r.counts.skipped);
        }
        try t.expectEqual(@as(?anyerror, null), stage1.err);
    }
}

test "rows stage: a save that waited for room and lost its entry is counted as skipped" {
    const x = try Fixture.open("stage-lost-entry", .off, false, 9 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(31);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        x.r.worker.?.limit = x.r.worker.?.bytes();
        const second = fixture.prompt(32);
        try x.keep(&second, 600);
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        for (x.s.entries.items, 0..) |e, i| if (e == x.r.waiting.?) {
            x.s.remove(i);
            break;
        };
        try t.expect(x.r.waiting == null);
        try t.expectEqual(@as(usize, 0), x.r.queue.items.len);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
        try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
        hold.release.store(true, .release);
        try x.settle();
        try t.expectEqual(@as(usize, 1), x.r.states.items.len);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
}

test "rows stage: losing the older of two waiting saves counts it as skipped" {
    const x = try Fixture.open("stage-two-waiters", .off, false, 9 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(41);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        x.r.worker.?.limit = x.r.worker.?.bytes();
        const second = fixture.prompt(42);
        try x.keep(&second, 600);
        const older = x.r.waiting.?;
        const third = fixture.prompt(43);
        try x.keep(&third, 600);
        try t.expectEqual(@as(usize, 2), x.r.queue.items.len);
        const newer = x.r.waiting.?;
        try t.expect(newer != older);
        for (x.s.entries.items, 0..) |e, i| if (e == older) {
            x.s.remove(i);
            break;
        };
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
        try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
        try t.expect(x.r.waiting == newer);
        try t.expect(x.r.queue.items[0].entry == newer and x.r.queue.items[0].waited);
        hold.release.store(true, .release);
        try x.idle(2);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
}

test "rows stage: a save that waited and loses its entry after a reap still counts, under an id no job holds" {
    const x = try Fixture.open("stage-lost-after-reap", .off, false, 9 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(51);
    var stage1 = Stager{ .x = x, .tokens = &first, .at = 600 };
    {
        const thread1 = try std.Thread.spawn(.{}, Stager.run, .{&stage1});
        defer {
            hold.release.store(true, .release);
            thread1.join();
        }
        try pendingAtLeast(x.r, 1);
        try Hold.wait(&hold.entered);
        const limit = x.r.worker.?.limit;
        x.r.worker.?.limit = x.r.worker.?.bytes();
        const second = fixture.prompt(52);
        try x.keep(&second, 600);
        const lost = x.r.waiting.?;
        hold.release.store(true, .release);
        try x.settle();
        try t.expect(x.r.waiting == null);
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        x.r.worker.?.limit = limit;
        const third = fixture.prompt(53);
        try x.keep(&third, 600);
        const staged_as = x.r.next_job;
        try t.expectEqual(@as(usize, 1), x.r.queue.items.len);
        for (x.s.entries.items, 0..) |e, i| if (e == lost) {
            x.s.remove(i);
            break;
        };
        try t.expectEqual(@as(usize, 0), x.r.queue.items.len);
        try t.expectEqual(@as(u64, 1), x.r.counts.skipped);
        try t.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
        try t.expect(x.r.failures.records[0].?.id != staged_as);
        try x.settle();
        try t.expectEqual(@as(usize, 2), x.r.states.items.len);
    }
    try t.expectEqual(@as(?anyerror, null), stage1.err);
}
