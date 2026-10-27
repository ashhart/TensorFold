//! Spill write-behind through the real cache and writer, with a filesystem call held until the test releases it.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const Fake = @import("rows_fake.zig").Fake;
const rmTree = @import("prompt_imprint_test.zig").rmTree;

const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),

    fn write(ctx: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        g.entered.store(true, .release);
        while (!g.release.load(.acquire)) nap();
        const n = platform.c.write(fd, bytes.ptr, bytes.len);
        if (n < 0) return error.RowsWrite;
        return @intCast(n);
    }
};

fn nap() void {
    var ts: platform.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
    _ = platform.c.nanosleep(&ts, null);
}

fn wait(flag: *std.atomic.Value(bool)) bool {
    const end = rows.write.nowNs() + std.time.ns_per_s;
    while (!flag.load(.acquire) and rows.write.nowNs() < end) nap();
    return flag.load(.acquire);
}

fn keep(s: *pc.Store, f: *Fake, tokens: []const u32) !void {
    f.prompt = tokens;
    f.at = 600;
    try std.testing.expect(s.keep(tokens, 600, null, &.{}, &.{}));
}

test "spill I/O: eviction releases its RAM entry while the filesystem write is still blocked" {
    const a = std.testing.allocator;
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-spill-async-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer rmTree(root);
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_gap = 1, .min_prompt = 0 }, 1 << 30);
    const r = try rows.attach(a, &s, .{ .root = root, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1, .spill = .evicted });
    defer r.deinit();
    defer s.deinit();
    var gate: Gate = .{};
    defer gate.release.store(true, .release);
    r.file_io = .{ .ctx = &gate, .write = Gate.write };
    var tokens: [700]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    try keep(&s, &f, &tokens);
    const Evict = struct {
        store: *pc.Store,
        done: std.atomic.Value(bool) = .init(false),
        fn run(e: *@This()) void {
            e.store.remove(0);
            e.done.store(true, .release);
        }
    };
    var ev: Evict = .{ .store = &s };
    const thread = try std.Thread.spawn(.{}, Evict.run, .{&ev});
    const entered = wait(&gate.entered);
    const returned = wait(&ev.done);
    gate.release.store(true, .release);
    thread.join();
    _ = rows.write.drain(r, rows.write.nowNs() + std.time.ns_per_s);
    try std.testing.expect(entered);
    try std.testing.expect(returned);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

const fixture = @import("rows_async_fixture.zig");
const Fixture = fixture.Fixture;
const Hold = fixture.Gate;

test "spill I/O: worker files have the synchronous SHA256 and a restore equals a fresh pass" {
    try byteFiles(false);
    try byteFiles(true);
}

fn byteFiles(planned: bool) !void {
    const synchronous = try Fixture.open("sync", .evicted, true, 1 << 20);
    defer synchronous.close();
    const threaded = try Fixture.open("thread", .evicted, false, 1 << 20);
    defer threaded.close();
    const tokens = fixture.prompt(7);
    for ([_]*Fixture{ synchronous, threaded }) |x| {
        x.f.spans = &.{.{ 400, 450 }};
        x.f.planned = planned;
        x.f.starts = if (planned) &.{ 0, 256, 512, 768, 900 } else &.{};
        x.s.rules.planned = planned;
        x.r.keying.planned = planned;
        try x.keep(&tokens, 900);
        x.s.remove(0);
        try x.settle();
        try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
        const saved = try rows.read.read(x.r, 0);
        defer x.f.snapshots().vtable.drop(&x.f, saved);
        try x.f.snapshots().vtable.restore(&x.f, null, saved);
        try std.testing.expect(x.f.restored_ok);
        const kept: *@import("rows_fake.zig").Kept = @ptrCast(@alignCast(saved));
        var expected: [8]u8 = undefined;
        std.mem.writeInt(u64, &expected, @import("rows_fake.zig").fresh(tokens[0..900]), .little);
        try std.testing.expectEqualSlices(u8, &expected, kept.img[8..16]);
    }
    const d = platform.c.opendir(synchronous.r.dir) orelse return error.OpenFailed;
    defer _ = platform.c.closedir(d);
    var checked: usize = 0;
    while (platform.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (!std.mem.endsWith(u8, name, ".st") and !std.mem.endsWith(u8, name, ".blk")) continue;
        try std.testing.expectEqual(try fixture.digest(synchronous.r.dir, name), try fixture.digest(threaded.r.dir, name));
        checked += 1;
    }
    try std.testing.expectEqual(fixture.files(threaded.r.dir, ".blk") + 1, checked);
    try std.testing.expect(checked > 2);
    try std.testing.expectEqual(@as(usize, 0), fixture.files(threaded.r.dir, ".part"));
}

test "spill I/O: ENOSPC, truncated short writes and fsync failures publish nothing and retain each job failure" {
    for ([_]@TypeOf(@as(Hold, .{}).mode){ .full, .short, .sync_failure }) |mode| {
        const x = try Fixture.open("fault", .evicted, false, 1 << 20);
        defer x.close();
        var fault: Hold = .{ .mode = mode, .release = .init(true) };
        x.r.file_io = fault.io();
        const tokens = fixture.prompt(9);
        try x.keep(&tokens, 600);
        x.s.remove(0);
        try x.settle();
        try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
        try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
        try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".blk"));
        try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
        try std.testing.expectEqual(@as(u64, 1), x.r.failures.count);
        const first = x.r.failures.records[0].?;
        try std.testing.expectEqual(@as(u32, 600), first.at);
        try std.testing.expectEqual(if (mode == .full) error.NoSpaceLeft else if (mode == .short) error.RowsWrite else error.SyncFailed, first.err);
        x.r.admission.retry_at = 0;
        const other = fixture.prompt(10);
        try x.keep(&other, 600);
        x.s.remove(0);
        try x.settle();
        try std.testing.expectEqual(@as(u64, 2), x.r.failures.count);
        try std.testing.expect(first.id != x.r.failures.records[1].?.id);
        try std.testing.expectEqual(@as(u64, 0), x.r.admission.reserved);
    }
}

test "spill I/O: the byte permit skips immediately and leaves token state unchanged" {
    const x = try Fixture.open("permit", .evicted, false, 1);
    defer x.close();
    const tokens = fixture.prompt(11);
    try x.keep(&tokens, 600);
    x.s.remove(0);
    try std.testing.expectEqual(@as(usize, 0), x.f.live);
    try std.testing.expectEqual(@as(u64, 1), x.r.counts.skipped);
    try std.testing.expectEqual(@as(u64, 0), x.r.worker.?.bytes());
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
}

test "spill I/O: an entry evicted during its write has no borrowed storage and publishes once" {
    const x = try Fixture.open("in-flight", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const tokens = fixture.prompt(12);
    try x.keep(&tokens, 600);
    try Hold.wait(&hold.entered);
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    x.s.remove(0);
    try std.testing.expectEqual(@as(usize, 0), x.f.live);
    try x.keep(&tokens, 600);
    x.s.remove(0);
    try std.testing.expectEqual(@as(usize, 1), x.r.worker.?.pending());
    hold.release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(u64, 1), x.r.counts.writes);
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    x.f.prompt = &tokens;
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try std.testing.expect(x.f.restored_ok);
}

test "spill I/O: forget fences a blocked job before unlink and the old generation cannot reappear" {
    const x = try Fixture.open("forget", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const tokens = fixture.prompt(13);
    try x.keep(&tokens, 600);
    try Hold.wait(&hold.entered);
    _ = x.r.forgetAll();
    hold.release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".part"));
    try std.testing.expectEqual(error.Cancelled, x.r.failures.records[0].?.err);
}

test "spill I/O: unload releases the family while a blocked syscall owns only staged bytes" {
    const x = try Fixture.open("unload", .finish, false, 1 << 20);
    var hold: Hold = .{};
    x.r.file_io = hold.io();
    const tokens = fixture.prompt(14);
    try x.keep(&tokens, 600);
    try Hold.wait(&hold.entered);
    const began = rows.write.nowNs();
    x.s.deinit();
    try std.testing.expect(rows.write.nowNs() - began < std.time.ns_per_s);
    try std.testing.expectEqual(@as(usize, 0), x.f.live);
    x.r.deinit();
    hold.release.store(true, .release);
    try Hold.wait(&hold.left);
    rmTree(x.root);
    fixture.a.free(x.root);
    fixture.a.destroy(x);
}

test "spill I/O: shutdown returns at its deadline while a write is blocked" {
    const x = try Fixture.open("deadline", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const tokens = fixture.prompt(15);
    try x.keep(&tokens, 600);
    try Hold.wait(&hold.entered);
    const began = rows.write.nowNs();
    const left = rows.write.drain(x.r, began + 20 * std.time.ns_per_ms);
    const elapsed = rows.write.nowNs() - began;
    hold.release.store(true, .release);
    try Hold.wait(&hold.left);
    try std.testing.expectEqual(@as(usize, 1), left);
    try std.testing.expect(elapsed < std.time.ns_per_s);
    try std.testing.expect(x.r.stopped and x.r.worker == null);
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
}

test "spill I/O: a completed payload is invisible until fsync finishes, with read-back verification" {
    const x = try Fixture.open("sync-gate", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{ .release = .init(true), .sync_release = .init(false) };
    defer hold.sync_release.store(true, .release);
    x.r.file_io = hold.io();
    x.r.verify = true;
    const tokens = fixture.prompt(16);
    try x.keep(&tokens, 600);
    try Hold.wait(&hold.sync_entered);
    rows.write.reap(x.r);
    try std.testing.expectEqual(@as(usize, 0), x.r.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".st"));
    try std.testing.expectEqual(@as(usize, 0), fixture.files(x.r.dir, ".blk"));
    hold.sync_release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
    try std.testing.expectEqual(@as(u64, 1), x.r.counts.verified);
    try std.testing.expectEqual(@as(u64, 0), x.r.counts.differed);
}

test "spill I/O: two in-flight states keep shared block references across a disk eviction" {
    const x = try Fixture.open("shared", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(17);
    var second = fixture.prompt(18);
    @memcpy(second[0..513], first[0..513]);
    try x.keep(&first, 900);
    try Hold.wait(&hold.entered);
    try x.keep(&second, 900);
    try std.testing.expectEqual(@as(usize, 2), x.r.worker.?.pending());
    try std.testing.expect(x.r.worker.?.bytes() <= x.r.staging_bytes);
    hold.release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(usize, 2), x.r.states.items.len);
    var shared: usize = 0;
    var it = x.r.blocks.valueIterator();
    while (it.next()) |b| shared += @intFromBool(b.refs == 2);
    try std.testing.expect(shared > 0);
    x.r.remove(0);
    x.f.prompt = &second;
    const saved = try rows.read.read(x.r, 0);
    defer x.f.snapshots().vtable.drop(&x.f, saved);
    try x.f.snapshots().vtable.restore(&x.f, null, saved);
    try std.testing.expect(x.f.restored_ok);
}

test "spill I/O: killing a child inside a partial write leaves no published state on restart" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const x = try Fixture.open("kill", .evicted, false, 1 << 20);
    defer x.close();
    var fds: [2]c_int = undefined;
    if (platform.c.pipe(&fds) != 0) return error.PipeFailed;
    defer for (fds) |fd| {
        _ = platform.c.close(fd);
    };
    const pid = platform.c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        const Crash = struct {
            fn write(ctx: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
                const ready: *c_int = @ptrCast(@alignCast(ctx.?));
                if (platform.c.write(fd, bytes.ptr, @min(bytes.len, 7)) <= 0) return error.RowsWrite;
                _ = platform.c.write(ready.*, "R", 1);
                while (true) rows.write.io.nap();
            }
        };
        x.r.file_io = .{ .ctx = &fds[1], .write = Crash.write };
        const tokens = fixture.prompt(19);
        x.keep(&tokens, 600) catch platform.c._exit(2);
        x.s.remove(0);
        while (true) rows.write.io.nap();
    }
    var reaped = false;
    defer if (!reaped) {
        _ = platform.c.kill(pid, .KILL);
        var status: c_int = 0;
        _ = platform.c.waitpid(pid, &status, 0);
    };
    var polls = [_]std.posix.pollfd{.{ .fd = fds[0], .events = std.posix.POLL.IN, .revents = 0 }};
    try std.testing.expect(try std.posix.poll(&polls, 3000) > 0);
    var byte: [1]u8 = undefined;
    try std.testing.expectEqual(@as(isize, 1), platform.c.read(fds[0], &byte, 1));
    try std.testing.expectEqual(@as(u8, 'R'), byte[0]);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.kill(pid, .KILL));
    var status: c_int = 0;
    try std.testing.expectEqual(pid, platform.c.waitpid(pid, &status, 0));
    reaped = true;
    try std.testing.expect(fixture.files(x.r.dir, ".part") > 0);
    const reopened = try rows.Rows.open(fixture.a, x.f.snapshots(), .{ .root = x.root, .identity = 7, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1 });
    defer reopened.deinit();
    try std.testing.expectEqual(@as(usize, 0), reopened.states.items.len);
    try std.testing.expectEqual(@as(usize, 0), reopened.blocks.count());
    try std.testing.expectEqual(@as(usize, 0), fixture.files(reopened.dir, ".part"));
}

test "spill I/O: a full byte permit skips another save without waiting for the first job" {
    const x = try Fixture.open("full-permit", .finish, false, 1 << 20);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    const first = fixture.prompt(20);
    try x.keep(&first, 600);
    try Hold.wait(&hold.entered);
    const used = x.r.worker.?.bytes();
    x.r.worker.?.limit = used;
    const second = fixture.prompt(21);
    try x.keep(&second, 600);
    try std.testing.expectEqual(@as(usize, 1), x.r.worker.?.pending());
    try std.testing.expectEqual(used, x.r.worker.?.bytes());
    try std.testing.expectEqual(error.StagingFull, x.r.failures.records[0].?.err);
    hold.release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(usize, 1), x.r.states.items.len);
}

test "spill I/O: the job permit bounds metadata when many small saves queue behind a blocked write" {
    const x = try Fixture.open("job-permit", .finish, false, 1 << 30);
    defer x.close();
    var hold: Hold = .{};
    defer hold.release.store(true, .release);
    x.r.file_io = hold.io();
    var tokens: [9][1100]u32 = undefined;
    for (&tokens, 0..) |*p, i| {
        p.* = fixture.prompt(@intCast(30 + i));
        try x.keep(p, 600);
    }
    try std.testing.expectEqual(@as(usize, 8), x.r.worker.?.pending());
    try std.testing.expectEqual(@as(u64, 1), x.r.counts.skipped);
    hold.release.store(true, .release);
    try x.settle();
    try std.testing.expectEqual(@as(usize, 8), x.r.states.items.len);
}

test "spill I/O: shutdown keeps a completed job's own filesystem error" {
    const x = try Fixture.open("stop-error", .evicted, false, 1 << 20);
    defer x.close();
    var fault: Hold = .{ .mode = .full, .release = .init(true) };
    x.r.file_io = fault.io();
    const tokens = fixture.prompt(50);
    try x.keep(&tokens, 600);
    x.s.remove(0);
    const end = rows.write.nowNs() + 3 * std.time.ns_per_s;
    while (!x.r.worker.?.ready() and rows.write.nowNs() < end) rows.write.io.nap();
    try std.testing.expect(x.r.worker.?.ready());
    rows.write.stop(x.r, rows.write.nowNs());
    try std.testing.expectEqual(error.NoSpaceLeft, x.r.failures.records[0].?.err);
    try std.testing.expectEqual(@as(u64, 1), x.r.counts.failed);
    try std.testing.expectEqual(@as(u64, 0), x.r.admission.reserved);
}
