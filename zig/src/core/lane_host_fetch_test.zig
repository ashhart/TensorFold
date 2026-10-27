//! A lane host reads a returning conversation's disk state ahead while another stream decodes (fake backend).
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const prefetch = @import("rows_prefetch.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const LaneHost = api.LaneHost;
const Reason = api.Reason;
const Allocator = std.mem.Allocator;
const gpa = std.testing.allocator;
const t = std.testing;

/// The fake backend's kept states (their history's tokens) as one row a token, for the rows store.
const Snaps = struct {
    fn of(ptr: *anyopaque) *lanes.fake.Fake {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return of(ptr).save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).drop(saved);
    }
    fn kept(saved: pc.Saved) *std.ArrayList(u32) {
        return @ptrCast(@alignCast(saved));
    }
    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(pc.rows.image.Part), a: Allocator) anyerror!void {
        try out.append(a, .{ .bytes = @as(u64, at) * 4, .row = 4 });
    }
    fn gather(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, out: []u8) anyerror!void {
        const src = std.mem.sliceAsBytes(kept(saved).items);
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(out[o..][0..@intCast(r.len)], src[@intCast(r.off)..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    fn blank(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
        const k = try of(ptr).gpa.create(std.ArrayList(u32));
        k.* = .empty;
        try k.appendNTimes(of(ptr).gpa, 0, at);
        return k;
    }
    fn scatter(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, in: []const u8) anyerror!void {
        const dst = std.mem.sliceAsBytes(kept(saved).items);
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(dst[@intCast(r.off)..][0..@intCast(r.len)], in[o..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    const vtable: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &.{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter } };
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |x| b.tokens.appendSlice(gpa, x) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            if (b.read().done) |r| return r;
            std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    fn read(b: *@This()) struct { n: usize, cached: ?u32, done: ?Reason } {
        b.mutex.lockUncancelable(t.io);
        defer b.mutex.unlock(t.io);
        return .{ .n = b.tokens.items.len, .cached = b.cached, .done = b.done };
    }
};

/// The other stream's stop check: a millisecond a token, so it decodes while the returning request waits, until `done`.
const Slow = struct {
    done: std.atomic.Value(bool) = .init(false),
    fn check(ctx: *anyopaque, _: []const u32) bool {
        const s: *Slow = @ptrCast(@alignCast(ctx));
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        return s.done.load(.acquire);
    }
};

/// A reader the test opens, counting the reads that reached it.
const Gate = struct {
    var open: std.atomic.Value(bool) = .init(false);
    var entered: std.atomic.Value(u32) = .init(0);

    fn reader(f: *prefetch.Prefetch, rd: *prefetch.Read) anyerror!void {
        _ = entered.fetchAdd(1, .acq_rel);
        while (!open.load(.acquire)) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        return prefetch.readFiles(f, rd);
    }
};

const Mode = enum { off, on, cancelled };

/// What a returning turn saw: its reply, where it resumed, how it ended, and the reads ahead.
const Seen = struct { tokens: [8]u32 = undefined, n: usize = 0, cached: ?u32 = null, reason: Reason = .failed, counts: prefetch.Counts = .{}, reads: u64 = 0 };

fn prompt(n: usize) [700]u32 {
    var p: [700]u32 = undefined;
    for (&p, 0..) |*x, i| x.* = @intCast((i * 7919 + n) % 997);
    return p;
}

/// One host over the fake backend with a prompt cache and a rows store under `root`; `mode` sets its reads ahead.
fn session(root: []const u8, mode: ?Mode, first: *[706]u32, seen: *Seen) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const family: pc.Snapshots = .{ .ptr = &target, .vtable = &Snaps.vtable };
    const rows = try pc.rows.Rows.open(gpa, family, .{ .root = root, .identity = 5, .cap = 1 << 30, .floor = 0, .daily = 1 << 30 });
    defer rows.deinit();
    var store = pc.Store.init(gpa, family, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.disk = rows;
    const f: ?*prefetch.Prefetch = if (mode) |m| try prefetch.make(rows, t.io, .{ .reads = @intFromBool(m != .off) }) else null;
    if (f) |x| x.reader = Gate.reader;
    Gate.open.store(mode == null or mode == .off, .release);
    Gate.entered.store(0, .release);
    var host = LaneHost.init(gpa, t.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    {
        defer host.stop();
        if (mode) |m| try returning(&host, m, first, seen) else try seed(&host, first);
    }
    seen.reads = rows.counts.reads;
    if (f) |x| seen.counts = x.counts;
}

/// Turn one of a conversation, its state at 600 written to disk while the host idles; `first` its prompt and reply.
fn seed(h: *LaneHost, first: *[706]u32) !void {
    const p = prompt(1);
    var b: Box = .{};
    defer b.tokens.deinit(gpa);
    const req: api.Request = .{ .prompt = &p, .max_tokens = 6, .history_len = 600 };
    try h.engine().submit(1, &req, .{ .ctx = &b, .event = Box.event });
    try t.expectEqual(Reason.length, b.wait());
    @memcpy(first[0..700], &p);
    @memcpy(first[700..], b.tokens.items);
    var waited: usize = 0;
    while (waited < 500) : (waited += 1) {
        var ask: api.ImprintAsk = .{ .op = .where, .prompt = first };
        try t.expect(h.engine().imprintAsk(&ask));
        try t.expect(ask.wait(t.io, 5000));
        if (ask.out.disk == 600) return;
        std.Io.sleep(t.io, .fromMilliseconds(10), .awake) catch {};
    }
    return error.NotWritten;
}

/// Another conversation decodes; the first returns, its state on disk, and waits for its read (or is cancelled).
fn returning(h: *LaneHost, mode: Mode, first: *[706]u32, seen: *Seen) !void {
    const p2 = prompt(2);
    var slow: Slow = .{};
    var other: Box = .{};
    defer other.tokens.deinit(gpa);
    const req2: api.Request = .{ .prompt = &p2, .max_tokens = 4000, .stop = .{ .ctx = &slow, .check = Slow.check } };
    try h.engine().submit(2, &req2, .{ .ctx = &other, .event = Box.event });
    defer _ = other.wait(); // a failed check still lets both requests end before their boxes go
    defer slow.done.store(true, .release);
    while (other.read().n == 0) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    var b: Box = .{};
    defer b.tokens.deinit(gpa);
    const req: api.Request = .{ .prompt = first, .max_tokens = 8, .history_len = 704 };
    try h.engine().submit(3, &req, .{ .ctx = &b, .event = Box.event });
    defer _ = b.wait();
    defer Gate.open.store(true, .release);
    if (mode != .off) {
        while (Gate.entered.load(.acquire) == 0) std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
        const n0 = other.read().n;
        std.Io.sleep(t.io, .fromMilliseconds(30), .awake) catch {};
        try t.expect(other.read().n > n0); // the other stream decodes while the read is under way
        try t.expectEqual(@as(?u32, null), b.read().cached); // and the returning request waits, not admitted
        if (mode == .cancelled) h.engine().cancel(3);
        if (mode == .cancelled) _ = b.wait();
        Gate.open.store(true, .release);
    }
    seen.reason = b.wait();
    const r = b.read();
    seen.cached = r.cached;
    seen.n = @min(r.n, seen.tokens.len);
    @memcpy(seen.tokens[0..seen.n], b.tokens.items[0..seen.n]);
    slow.done.store(true, .release);
    try t.expectEqual(Reason.stop, other.wait());
}

/// The fake's replies read the whole history: tokens a wrong restored row would change.
fn fresh(first: []const u32, out: []u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, first);
    for (out) |*x| {
        x.* = lanes.fake.next(history.items, null, history.items.len);
        try history.append(gpa, x.*);
    }
}

/// A fresh root seeded with turn one's state, then its return under `mode`; `first` is turn one's prompt and reply.
fn run(mode: Mode, first: *[706]u32) !Seen {
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-host-fetch-{s}-{d}", .{ @tagName(mode), platform.c.getpid() });
    defer rmTree(root);
    var s: Seen = .{};
    try session(root, null, first, &s);
    s = .{};
    try session(root, mode, first, &s);
    return s;
}

test "prefetch: a returning conversation's state is read ahead while another decodes; its reply equals off and fresh" {
    var first: [706]u32 = undefined;
    const off = try run(.off, &first);
    var want: [8]u32 = undefined;
    try fresh(&first, &want);
    const on = try run(.on, &first);
    for ([_]*const Seen{ &off, &on }) |s| {
        try t.expectEqual(Reason.length, s.reason);
        try t.expectEqual(@as(?u32, 600), s.cached);
        try t.expectEqualSlices(u32, &want, s.tokens[0..s.n]);
        try t.expectEqual(@as(u64, 1), s.reads);
    }
    try t.expectEqual(@as(u64, 0), off.counts.made); // TENSORFOLD_RESTORE_PREFETCH=0: read at admission, as before
    try t.expectEqual(@as(u64, 1), on.counts.made);
    try t.expectEqual(@as(u64, 1), on.counts.used);
    try t.expectEqual(@as(u64, 1), on.counts.freed);
}

test "prefetch: a returning request cancelled while its read is under way ends once; the read is freed once" {
    var first: [706]u32 = undefined;
    const s = try run(.cancelled, &first);
    try t.expectEqual(Reason.cancelled, s.reason);
    try t.expectEqual(@as(?u32, null), s.cached);
    try t.expectEqual(@as(u64, 0), s.reads);
    try t.expectEqual(@as(u64, 1), s.counts.made);
    try t.expectEqual(@as(u64, 1), s.counts.freed);
    try t.expectEqual(@as(u64, 1), s.counts.cancelled);
    try t.expectEqual(@as(u64, 0), s.counts.used);
}
