//! Disk states read ahead: restores equal to synchronous ones and fresh passes, freed once, bounded, falling back.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const prefetch = @import("rows_prefetch.zig");
const fixture = @import("rows_async_fixture.zig");
const Fake = @import("rows_fake.zig").Fake;
const fresh = @import("rows_fake.zig").fresh;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Name = rows.chain.Name;
const Allocator = std.mem.Allocator;
const gpa = std.testing.allocator;
const t = std.testing;

const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };

fn root(buf: []u8, tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-prefetch-{s}-{d}", .{ tag, platform.c.getpid() }) catch unreachable;
}

fn options(dir: []const u8) rows.Options {
    return .{ .root = dir, .identity = 7, .cap = 1 << 34, .floor = 0, .daily = 1 << 40, .lookahead = 1 };
}

fn prompt(a: Allocator, c: u32, n: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*x, i| x.* = @intCast(50_000 + c * 10_000 + i);
    return out;
}

fn request(a: Allocator, s: *pc.Store, f: *Fake, p: []const u32, history: u32) !struct { from: u32, sum: u64 } {
    f.prompt = p;
    const plan = try s.begin(a, p, history, &.{}, f.starts, null, f.spans);
    return .{ .from = plan.from, .sum = f.pass(s, plan) };
}

/// Conversation `c`'s state at 1400 tokens written under `dir` (beside what is there); its next turn's prompt.
fn written(a: Allocator, dir: []const u8, c: u32) ![]u32 {
    const turn1 = try prompt(a, c, 1500);
    defer a.free(turn1);
    var f: Fake = .{ .gpa = gpa };
    var opts = options(dir);
    opts.sync_writes = true;
    const r = try rows.Rows.open(gpa, f.snapshots(), opts);
    defer r.deinit();
    var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    _ = try request(a, &s, &f, turn1, 1400);
    const next = try std.mem.concat(a, u32, &.{ turn1[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    errdefer a.free(next);
    const left = rows.write.drain(r, rows.write.nowNs() + 60 * std.time.ns_per_s);
    if (left != 0 or r.best(next, &.{}, &.{}, 0) == null) return error.SeedNotWritten;
    return next;
}

/// A server's store over `dir`: the family, its disk states and the RAM store (in place: the store points at `f`).
const Server = struct {
    f: Fake = .{ .gpa = gpa },
    r: *rows.Rows = undefined,
    s: pc.Store = undefined,

    fn open(x: *Server, dir: []const u8) !void {
        x.r = try rows.Rows.open(gpa, x.f.snapshots(), options(dir));
        x.s = pc.Store.init(gpa, x.f.snapshots(), rules, 1 << 30);
        x.s.disk = x.r;
    }

    fn close(x: *Server) void {
        x.s.deinit();
        x.r.deinit();
    }

    /// A read ahead of the state `p`'s next request would restore.
    fn start(x: *Server, f: *prefetch.Prefetch, p: []const u32) ?Name {
        return f.start(x.r, prefetch.choose(&x.s, p, &.{}, &.{}) orelse return null);
    }
};

pub fn counts(f: *prefetch.Prefetch) prefetch.Counts {
    f.mutex.lockUncancelable(f.io);
    defer f.mutex.unlock(f.io);
    return f.counts;
}

fn bound() i128 {
    return rows.write.nowNs() + 60 * std.time.ns_per_s;
}

/// `name`'s read once it has finished or failed (null: it left the list, or the bound passed).
pub fn settled(f: *prefetch.Prefetch, name: Name) ?prefetch.Phase {
    const end = bound();
    while (rows.write.nowNs() < end) {
        f.mutex.lockUncancelable(f.io);
        const phase: ?prefetch.Phase = for (f.reads.items) |rd| {
            if (std.mem.eql(u8, &rd.name, &name)) break rd.phase;
        } else null;
        f.mutex.unlock(f.io);
        if (phase == null or phase.? == .ready or phase.? == .failed) return phase;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    return null;
}

/// First availability releases admission; terminal readiness may await backpressured chunk recycling.
pub fn available(f: *prefetch.Prefetch, name: Name) !void {
    const end = bound();
    while (f.pending(name)) {
        if (rows.write.nowNs() >= end) return error.PrefetchTimedOut;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}
/// `name`'s read phase as it stands: `take` removes the detached winner, `forget` the refused states' reads.
pub fn phased(f: *prefetch.Prefetch, name: Name) ?prefetch.Phase {
    f.mutex.lockUncancelable(f.io);
    defer f.mutex.unlock(f.io);
    for (f.reads.items) |rd| if (std.mem.eql(u8, &rd.name, &name)) return rd.phase;
    return null;
}

pub fn until(f: *prefetch.Prefetch, comptime field: []const u8, n: u64) !void {
    const end = bound();
    while (@field(counts(f), field) < n) {
        if (rows.write.nowNs() >= end) return error.PrefetchTimedOut;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
}

/// A slow reader the test opens (or fails), counting the reads that reached it.
const Gate = struct {
    var open: std.atomic.Value(bool) = .init(false);
    var entered: std.atomic.Value(u32) = .init(0);
    var fail: bool = false;

    fn reset(fails: bool) void {
        open.store(false, .release);
        entered.store(0, .release);
        fail = fails;
    }

    fn reader(f: *prefetch.Prefetch, rd: *prefetch.Read) anyerror!void {
        _ = entered.fetchAdd(1, .acq_rel);
        while (!open.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        if (fail) return error.FakeReadFailed;
        return prefetch.readFiles(f, rd);
    }

    fn reached(n: u32) void {
        while (entered.load(.acquire) < n) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
};

fn none(_: void, _: Name) bool {
    return false;
}

fn only(n: *const Name, name: Name) bool {
    return std.mem.eql(u8, n, &name);
}

test "prefetch: a seed that cannot publish reports SeedNotWritten" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "seed-failed");
    defer rmTree(dir);
    const turn1 = try prompt(a, 1, 1500);
    var f: Fake = .{ .gpa = gpa };
    var path: [1200]u8 = undefined;
    const dest = blk: {
        const r = try rows.Rows.open(gpa, f.snapshots(), options(dir));
        defer r.deinit();
        const name = rows.chain.state(r.keying, turn1[0..1401], &.{}, &.{}, 1400);
        break :blk try rows.file.path(&path, r.dir, &rows.chain.hex(name), ".st", false);
    };
    try t.expectEqual(@as(c_int, 0), platform.c.mkdir(dest, 0o700));
    defer _ = platform.c.rmdir(dest);
    try t.expectError(error.SeedNotWritten, written(a, dir, 1));
}

test "prefetch: an unfinished asynchronous seed has no state or reader after reopen" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "seed-late");
    defer rmTree(dir);
    const turn1 = try prompt(a, 1, 1500);
    const next = try std.mem.concat(a, u32, &.{ turn1[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    {
        var x: Server = .{};
        try x.open(dir);
        defer x.close();
        var gate: fixture.Gate = .{};
        defer gate.release.store(true, .release);
        x.r.file_io = gate.io();
        _ = try request(a, &x.s, &x.f, turn1, 1400);
        // `step` streams on its caller, so the stage runs beside the held writer; the drain abandons it mid-stream.
        const stager = try std.Thread.spawn(.{}, struct {
            fn run(r: *rows.Rows) void {
                _ = rows.write.step(r);
            }
        }.run, .{x.r});
        defer stager.join();
        defer gate.release.store(true, .release); // frees the worker before the join above unwinds
        try fixture.Gate.wait(&gate.entered);
        try t.expectEqual(@as(usize, 1), x.r.worker.?.pending());
        const left = rows.write.drain(x.r, rows.write.nowNs() + 20 * std.time.ns_per_ms);
        try t.expectEqual(@as(usize, 1), left);
        gate.release.store(true, .release);
        try fixture.Gate.wait(&gate.left);
    }
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{});
    try t.expect(x.start(f, next) == null);
    try t.expect(f.thread == null);
}

test "prefetch: a state read ahead restores equal to a synchronous read and to a fresh pass, nothing read at admission" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "equal");
    defer rmTree(dir);
    const again = try written(a, dir, 1);
    var sync: Server = .{};
    try sync.open(dir);
    const want = try request(a, &sync.s, &sync.f, again, 1404);
    const read_bytes = sync.r.counts.read;
    sync.close();
    try t.expectEqual(@as(u32, 1400), want.from);
    try t.expect(sync.f.restored_ok);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{});
    const name = x.start(f, again) orelse return error.NotReadAhead;
    try available(f, name);
    const got = try request(a, &x.s, &x.f, again, 1404);
    try t.expectEqual(@as(u32, 1400), got.from);
    try t.expect(x.f.restored_ok); // every byte of the restored image equals a fresh pass's
    try t.expectEqual(want.sum, got.sum);
    try t.expectEqual(fresh(again), got.sum);
    try t.expectEqual(@as(u64, 1), x.r.counts.reads);
    try t.expectEqual(read_bytes, x.r.counts.read);
    try t.expectEqual(@as(u64, 1), counts(f).used);
}

test "prefetch: cancellation before, during and after the read frees each read exactly once" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "cancel");
    defer rmTree(dir);
    const one = try written(a, dir, 1);
    const two = try written(a, dir, 2);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{});
    f.reader = Gate.reader;
    Gate.reset(false);
    defer Gate.open.store(true, .release);
    const n1 = x.start(f, one) orelse return error.NotReadAhead;
    _ = x.start(f, two) orelse return error.NotReadAhead;
    Gate.reached(1); // the first is reading, the second queued behind it
    f.sweep(&n1, only); // before: the queued one's request went
    try t.expectEqual(@as(u64, 1), counts(f).freed);
    f.sweep({}, none); // during: the reading one's request went; the I/O thread frees it when its read returns
    try t.expectEqual(@as(u64, 1), counts(f).freed);
    f.limits.reads = 1;
    try t.expect(x.start(f, one) == null); // a cancelled read keeps its permit until its blocked syscall returns
    Gate.open.store(true, .release);
    try until(f, "freed", 2);
    const n3 = x.start(f, one) orelse return error.NotReadAhead; // after: some payload is available, then its request went
    try available(f, n3);
    f.sweep({}, none);
    try until(f, "freed", 3);
    const c = counts(f);
    try t.expectEqual(@as(u64, 3), c.made);
    try t.expectEqual(@as(u64, 3), c.freed);
    try t.expectEqual(@as(u64, 3), c.cancelled);
    try t.expectEqual(@as(u64, 0), c.used);
    try t.expectEqual(@as(usize, 0), x.f.live); // no kept state was made for a read that went
    try t.expectEqual(@as(usize, 0), f.reads.items.len);
}

test "prefetch: a failed read, or a file that fails its check, falls back to today's synchronous read" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "failed");
    defer rmTree(dir);
    const again = try written(a, dir, 1);
    {
        var x: Server = .{};
        try x.open(dir);
        defer x.close();
        const f = try prefetch.make(x.r, t.io, .{});
        f.reader = Gate.reader;
        Gate.reset(true);
        defer Gate.open.store(true, .release);
        Gate.open.store(true, .release);
        const name = x.start(f, again) orelse return error.NotReadAhead;
        try t.expectEqual(@as(?prefetch.Phase, .failed), settled(f, name));
        const got = try request(a, &x.s, &x.f, again, 1404); // read from disk at admission, as before
        try t.expectEqual(@as(u32, 1400), got.from);
        try t.expect(x.f.restored_ok);
        try t.expectEqual(fresh(again), got.sum);
        try t.expectEqual(@as(u64, 0), x.r.counts.failed);
        const c = counts(f);
        try t.expectEqual(@as(u64, 1), c.failed);
        try t.expectEqual(@as(u64, 0), c.used);
        try t.expectEqual(c.made, c.freed);
    }
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    var it = x.r.blocks.keyIterator();
    var path: [1200]u8 = undefined;
    const file = try rows.file.path(&path, x.r.dir, &rows.chain.hex(it.next().?.*), ".blk", false);
    const fd = platform.c.open(file, .{ .ACCMODE = .RDWR }, @as(platform.c.mode_t, 0));
    try t.expect(fd >= 0);
    try t.expectEqual(@as(isize, 1), platform.c.pwrite(fd, "X", 1, 100));
    _ = platform.c.close(fd);
    const f = try prefetch.make(x.r, t.io, .{});
    const name = x.start(f, again) orelse return error.NotReadAhead;
    try available(f, name); // a read longer than the fixed handoff cannot finish until its request drains it
    const got = try request(a, &x.s, &x.f, again, 1404); // the drained read finds the garbled block, as today
    try t.expectEqual(@as(u32, 0), got.from);
    try t.expectEqual(fresh(again), got.sum);
    try t.expectEqual(@as(u64, 1), x.r.counts.failed);
    try t.expect(x.r.best(again, &.{}, &.{}, 0) == null);
}

test "prefetch: two requests join one producer; its detached stream belongs to the first consumer" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "twice");
    defer rmTree(dir);
    const again = try written(a, dir, 1);
    const other = try std.mem.concat(a, u32, &.{ again[0..1401], &[_]u32{ 5, 5, 5 } });
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{});
    f.reader = Gate.reader;
    Gate.reset(false);
    defer Gate.open.store(true, .release);
    const n1 = x.start(f, again) orelse return error.NotReadAhead;
    const n2 = x.start(f, other) orelse return error.NotReadAhead;
    try t.expectEqualSlices(u8, &n1, &n2);
    try t.expect(f.pending(n1));
    Gate.open.store(true, .release);
    try available(f, n1);
    try t.expectEqual(@as(u32, 1), Gate.entered.load(.acquire));
    const got1 = try request(a, &x.s, &x.f, again, 1404);
    const got2 = try request(a, &x.s, &x.f, other, 1403);
    try t.expectEqual(@as(u32, 1400), got1.from);
    try t.expectEqual(@as(u32, 1400), got2.from);
    try t.expect(x.f.restored_ok);
    try t.expectEqual(fresh(again), got1.sum);
    try t.expectEqual(fresh(other), got2.sum);
    try t.expectEqual(@as(u64, 1), x.r.counts.reads); // a detached read belongs to its first engine consumer; the second resumed from RAM
    const c = counts(f);
    try t.expectEqual(@as(u64, 1), c.made);
    try t.expectEqual(@as(u64, 1), c.used);
}

test "prefetch: admission detaches an in-progress compatible read and waits for its stream" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "detach-progress");
    defer rmTree(dir);
    const again = try written(a, dir, 1);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{ .wait_ns = 0 });
    f.reader = Gate.reader;
    Gate.reset(false);
    defer Gate.open.store(true, .release);
    const name = x.start(f, again) orelse return error.NotReadAhead;
    Gate.reached(1);
    try t.expect(!f.pending(name)); // past its bound: the request stops waiting, counted once
    try t.expect(!f.pending(name));
    try t.expectEqual(@as(u64, 1), counts(f).timeouts);
    Gate.open.store(true, .release);
    const got = try request(a, &x.s, &x.f, again, 1404);
    try t.expectEqual(@as(u32, 1400), got.from);
    try t.expect(x.f.restored_ok);
    try t.expectEqual(fresh(again), got.sum);
    try until(f, "freed", 1);
    const c = counts(f);
    try t.expectEqual(@as(u64, 1), c.used);
    try t.expectEqual(@as(u64, 1), c.made);
}

test "prefetch: finish cancels and joins a detached producer without touching its source" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "detach-cancel");
    defer rmTree(dir);
    const again = try written(a, dir, 1);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{ .reads = 1 });
    f.reader = Gate.reader;
    Gate.reset(false);
    defer Gate.open.store(true, .release);
    const i = prefetch.choose(&x.s, again, &.{}, &.{}) orelse return error.NotReadAhead;
    const name = x.start(f, again) orelse return error.NotReadAhead;
    Gate.reached(1);
    var plan = try prefetch.Plan.of(x.r, a, &x.r.states.items[i]);
    defer plan.deinit(a);
    const rd = f.take(name, &plan) orelse return error.NotReadAhead;
    Gate.open.store(true, .release);
    f.finish(rd);
    try t.expectEqual(@as(u64, 1), counts(f).freed);
    const got = try request(a, &x.s, &x.f, again, 1404);
    try t.expectEqual(@as(u32, 1400), got.from);
    try t.expect(x.f.restored_ok);
    try t.expectEqual(fresh(again), got.sum);
    try t.expectEqual(@as(u64, 1), x.r.counts.reads);
    try t.expectEqual(@as(u64, 1), counts(f).used);
}

test "prefetch: reads and bytes held are bounded, and 0 reads is off with no thread" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const dir = root(&buf, "bounds");
    defer rmTree(dir);
    const one = try written(a, dir, 1);
    const two = try written(a, dir, 2);
    var x: Server = .{};
    try x.open(dir);
    defer x.close();
    const f = try prefetch.make(x.r, t.io, .{ .reads = 1 });
    try t.expect(x.start(f, one) != null);
    try t.expect(x.start(f, one) != null); // joining the same state takes no other read
    try t.expect(x.start(f, two) == null);
    const held = f.snapshot();
    try t.expectEqual(@as(usize, 1), held.held_reads);
    try t.expectEqual(@as(u64, 2 * prefetch.chunk_bytes), held.held_bytes);
    f.stop();
    f.limits = .{ .bytes = 1 };
    try t.expect(x.start(f, one) == null);
    f.limits = .{ .reads = 0 };
    try t.expect(!f.on() and x.start(f, two) == null);
    try t.expect(f.thread == null);
    const c = counts(f);
    try t.expectEqual(@as(u64, 1), c.made);
    try t.expectEqual(@as(u64, 1), c.freed);
}

test "prefetch: TENSORFOLD_RESTORE_PREFETCH* are whole numbers; anything else keeps the default" {
    const d: prefetch.Limits = .{};
    try t.expectEqual(d, prefetch.Limits.fromText("", "", ""));
    try t.expectEqual(d, prefetch.Limits.fromText("x", "-1", "1.5"));
    const l = prefetch.Limits.fromText("0", "3", "250");
    try t.expectEqual(@as(usize, 0), l.reads);
    try t.expectEqual(@as(u64, 3 << 30), l.bytes);
    try t.expectEqual(@as(i128, 250) * std.time.ns_per_ms, l.wait_ns);
}
