//! #1594 step 2b on the real lane host and a fake backend: a remote prefix is restored, never counted, dropped once.
const std = @import("std");
const platform = @import("sys");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const remote = @import("lane_remote.zig");
const exporting = @import("lane_export.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const t = std.testing;

/// Another machine's prefill, in the fake backend's kept-state form: the prefix's own tokens.
pub const Peer = struct {
    at: u32,
    fail: bool = false,
    fills: u32 = 0,
    drops: std.atomic.Value(u32) = .init(0),
    running: std.atomic.Value(u32) = .init(0),
    kept: std.ArrayList(u32) = .empty,

    fn wants(ptr: *anyopaque, _: u64, cached: u64) bool {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        return p.at > cached;
    }

    fn fill(ptr: *anyopaque, prompt: []const u32, _: u64) anyerror!?remote.Filled {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        p.fills += 1;
        if (p.fail) return error.LinkDown;
        p.kept.clearRetainingCapacity();
        try p.kept.appendSlice(t.allocator, prompt[0..@min(p.at, prompt.len)]);
        return .{ .saved = &p.kept, .at = p.at };
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const p: *Peer = @ptrCast(@alignCast(ptr));
        std.debug.assert(saved == @as(pc.Saved, &p.kept));
        _ = p.drops.fetchAdd(1, .acq_rel);
    }
};

/// What the backend saw at the pass's first chunk, and what the request heard.
pub const Seen = struct {
    restoring: bool = false,
    at: u32 = 0,
    remote: bool = false,
    marks: usize = 0,
    hooked: bool = false,
    prefilled: ?u32 = null,
    tokens: usize = 0,
    remote_prefix: ?u32 = null, // the finished stats' count, which the server reports as exact_mode "remote_prefix"
    pulled_prefix: ?u32 = null, // of it, pulled for this reply; the rest is the server's kept_remote_cache_tokens
    store: ?*const pc.Store = null, // the run's cache, its borrowers counted at the first chunk
    borrowed: u32 = 0, // its states still borrowed at the first chunk: a lookup let go before the pass is not
    refuse: bool = false, // the first chunk refuses a pulled copy, as qwen27_slots.opening can: the pass starts at 0
    inherit: bool = false, // the first chunk stands the stream on slot 0's remote record, as qwen27_resume.inherit does
    done: std.atomic.Value(bool) = .init(false),

    fn chunk(ctx: *anyopaque, s: *lanes.Stream, n: usize) void {
        if (n != 0) return;
        const x: *Seen = @ptrCast(@alignCast(ctx));
        const kept: []const *pc.Entry = if (x.store) |st| st.entries.items else &.{};
        var borrowed: u32 = 0;
        for (kept) |e| borrowed += e.borrowers;
        x.* = .{ .restoring = s.reuse.saved != null, .at = s.reuse.at, .remote = s.reuse.remote, .marks = s.reuse.marks.len, .hooked = s.reuse.hook != null, .store = x.store, .borrowed = borrowed, .refuse = x.refuse, .inherit = x.inherit };
        if (x.inherit) s.reuse.remote = true;
        if (!x.refuse or !s.reuse.pulled) return;
        s.reuse_failed = true; // the pass prefills from token 0 with reuse.remote still set, as on the Mac
        s.reuse.saved = null;
    }

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Seen = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .prefilled => |n| x.prefilled = n,
            .tokens => |ts| x.tokens += ts.len,
            .finished => |f| {
                x.remote_prefix = f.stats.remote_prefix;
                x.pulled_prefix = f.stats.pulled_prefix;
                x.done.store(true, .release);
            },
            else => {},
        }
    }
};

/// A producer's exporter over the fake backend's copies: what it was handed, or why not.
pub const Exp = struct {
    fake: *lanes.fake.Fake = undefined,
    fail: bool = false,
    calls: u32 = 0,
    tokens: ?usize = null,
    kept: ?usize = null,
    why: ?[]const u8 = null,

    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const e: *Exp = @ptrCast(@alignCast(ptr));
        if (e.fail) return error.SaveRefused;
        return e.fake.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 0;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const e: *Exp = @ptrCast(@alignCast(ptr));
        e.fake.drop(saved);
    }
    fn exported(ptr: *anyopaque, _: exporting.Handoff, tokens: []const u32, saved: pc.Saved) void {
        const e: *Exp = @ptrCast(@alignCast(ptr));
        e.calls += 1;
        e.tokens = tokens.len;
        e.kept = @as(*std.ArrayList(u32), @ptrCast(@alignCast(saved))).items.len;
        e.fake.drop(saved);
    }
    fn failed(ptr: *anyopaque, _: exporting.Handoff, why: []const u8) void {
        const e: *Exp = @ptrCast(@alignCast(ptr));
        e.calls += 1;
        e.why = why;
    }
    pub fn exporter(e: *Exp) exporting.Exporter {
        return .{ .ptr = e, .snapshots = .{ .ptr = e, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .exported = exported, .failed = failed };
    }
};

fn run(peer: *Peer, request: *const api.Request) !Seen {
    return runAs(peer, request, null);
}

fn runAs(peer: *Peer, request: *const api.Request, exp: ?*Exp) !Seen {
    return runWith(peer, request, exp, null);
}

pub fn runWith(peer: *Peer, request: *const api.Request, exp: ?*Exp, cache: ?*pc.Store) !Seen {
    return runKeep(peer, request, exp, cache, .{});
}

/// A run's host beyond the remote source: TF_PREFILL_KEEP, the server's warm passes, slot 0's own resume point.
pub const Host = struct { keep: bool = false, warm: bool = false, peek: ?remote.ResumePeek = null, refuse: bool = false, inherit: bool = false };

pub fn runKeep(peer: *Peer, request: *const api.Request, exp: ?*Exp, cache: ?*pc.Store, o: Host) !Seen {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var seen: Seen = .{ .store = cache, .refuse = o.refuse, .inherit = o.inherit };
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = Seen.chunk, .prefill_hook_ctx = &seen };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1, .warm_turns = o.warm });
    host.remote = .{ .ptr = peer, .wants = Peer.wants, .fill = Peer.fill, .drop = Peer.drop, .running = &peer.running };
    if (cache) |store| host.cache = store;
    host.keep_remote = o.keep;
    host.resume_peek = o.peek;
    if (exp) |e| {
        e.fake = &fake;
        host.exporter = e.exporter();
    }
    try host.start();
    defer host.stop();
    try host.engine().submit(1, request, .{ .ctx = &seen, .event = Seen.event });
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!seen.done.load(.acquire) or peer.running.load(.acquire) > 0) { // a fill's thread is done with the host
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    return seen;
}

pub const six = [_]u32{ 11, 12, 13, 14, 15, 16 };
pub const eight = six ++ [_]u32{ 17, 18 }; // a chat's next turn: six, then more

test "remote prefix: the stream restores it with nothing kept after it, counts none of it cached, and it is dropped once" {
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const seen = try run(&peer, &.{ .prompt = &six, .max_tokens = 2 });
    try t.expect(seen.restoring and seen.remote);
    try t.expectEqual(@as(u32, 4), seen.at);
    try t.expectEqual(@as(usize, 0), seen.marks);
    try t.expect(!seen.hooked);
    try t.expectEqual(@as(?u32, 0), seen.prefilled);
    try t.expectEqual(@as(u32, 1), peer.fills);
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
}

test "remote prefix: a failed fill leaves the whole prompt to this engine, and nothing is dropped" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
    var peer: Peer = .{ .at = 4, .fail = true };
    defer peer.kept.deinit(t.allocator);
    const seen = try run(&peer, &.{ .prompt = &six, .max_tokens = 2 });
    try t.expect(!seen.restoring and !seen.remote);
    try t.expectEqual(@as(u32, 1), peer.fills);
    try t.expectEqual(@as(u32, 0), peer.drops.load(.acquire));
}

test "remote prefix: one covering the whole prompt is dropped unused, and a background pass never asks" {
    var whole: Peer = .{ .at = 6 };
    defer whole.kept.deinit(t.allocator);
    const seen = try run(&whole, &.{ .prompt = &six, .max_tokens = 2 });
    try t.expect(!seen.restoring and !seen.remote);
    try t.expectEqual(@as(u32, 1), whole.drops.load(.acquire));
    var quiet: Peer = .{ .at = 4 };
    defer quiet.kept.deinit(t.allocator);
    _ = try run(&quiet, &.{ .prompt = &six, .max_tokens = 0, .background = true });
    try t.expectEqual(@as(u32, 0), quiet.fills);
}

pub const id: exporting.Handoff = .{ .id = @splat(0x2b) };

test "handoff: the producer's stream exports its state before the prompt's last token, once, and draws one token" {
    var none: Peer = .{ .at = 0 };
    var e: Exp = .{};
    const seen = try runAs(&none, &.{ .prompt = &six, .max_tokens = 8, .handoff = id }, &e);
    try t.expectEqual(@as(u32, 1), e.calls);
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expectEqual(@as(?usize, 5), e.kept);
    try t.expectEqual(@as(usize, 1), seen.tokens);
    try t.expectEqual(@as(u32, 0), none.fills);
}

test "handoff: a save that fails reaches the decoder's reason, and a server with no exporter refuses the request" {
    var none: Peer = .{ .at = 0 };
    var e: Exp = .{ .fail = true };
    _ = try runAs(&none, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &e);
    try t.expectEqual(@as(u32, 1), e.calls);
    try t.expectEqualStrings("SaveRefused", e.why.?);
    try t.expectError(error.InvalidRequest, run(&none, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }));
}

/// A family whose copies are the fake backend's kept form, made without a stream: the prompt's first `at` tokens.
pub const Kept = struct {
    pub fn bytes(_: *anyopaque, _: u32) u64 {
        return 64;
    }
    pub fn save(_: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const kept = try t.allocator.create(std.ArrayList(u32));
        kept.* = .empty;
        try kept.appendSlice(t.allocator, eight[0..at]);
        return kept;
    }
    pub fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    pub fn drop(_: *anyopaque, saved: pc.Saved) void {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        kept.deinit(t.allocator);
        t.allocator.destroy(kept);
    }
    /// The rows store's image of a copy: a word a token, written whole.
    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(pc.rows.image.Part), a: std.mem.Allocator) anyerror!void {
        try out.append(a, .{ .bytes = @as(u64, at) * 4, .row = 4 });
    }
    fn gather(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, out: []u8) anyerror!void {
        const words = std.mem.sliceAsBytes(@as(*std.ArrayList(u32), @ptrCast(@alignCast(saved))).items);
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(out[o..][0..@intCast(r.len)], words[@intCast(r.off)..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    fn blank(_: *anyopaque, _: u32) anyerror!pc.Saved {
        return error.NeverReadBack; // these tests only ever check what reaches the disk
    }
    fn scatter(_: *anyopaque, _: pc.Saved, _: []const pc.rows.image.Range, _: []const u8) anyerror!void {
        return error.NeverReadBack;
    }
};

test "remote prefix: a prefix the cache already holds starts no remote fill, and the cache's state is restored" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 2 }, null, &store);
    try t.expect(seen.restoring and !seen.remote);
    try t.expectEqual(@as(u32, 4), seen.at);
    try t.expectEqual(@as(?u32, 4), seen.prefilled);
    try t.expectEqual(@as(u32, 0), peer.fills);
}

test "remote prefix: a request a closed host refuses starts no remote fill" {
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    host.remote = .{ .ptr = &peer, .wants = Peer.wants, .fill = Peer.fill, .drop = Peer.drop, .running = &peer.running };
    try host.start();
    host.stop();
    var seen: Seen = .{};
    try t.expectError(error.Closed, host.engine().submit(1, &.{ .prompt = &six, .max_tokens = 2 }, .{ .ctx = &seen, .event = Seen.event }));
    try t.expectEqual(@as(u32, 0), peer.fills);
}

test "remote prefix: with a prompt cache on, a remote stream keeps no state of its own" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 2 }, null, &store);
    try t.expect(seen.restoring and seen.remote);
    try t.expect(!seen.hooked); // without lane_remote.wins/use the lookup would have set Job.kept here
    try t.expectEqual(@as(usize, 0), seen.marks);
    try t.expectEqual(@as(u64, 0), store.counts.kept);
}

test "handoff: a kept state exactly at the cut is not restored, so the pass still cuts there and exports" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 5, null, &.{}, &.{}));
    var peer: Peer = .{ .at = 0 };
    var e: Exp = .{};
    _ = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &e, &store);
    try t.expectEqual(@as(u32, 1), e.calls);
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expect(e.why == null);
}

test "handoff: the producer's cache keeps the exported cut, and the chat's next turn resumes there" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var peer: Peer = .{ .at = 0 };
    var first: Exp = .{};
    _ = try runWith(&peer, &.{ .prompt = six[0..5], .max_tokens = 1, .handoff = id }, &first, &store);
    try t.expectEqual(@as(?usize, 4), first.tokens);
    try t.expectEqual(@as(u64, 1), store.counts.kept);
    var next: Exp = .{};
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &next, &store);
    try t.expect(seen.restoring and !seen.remote);
    try t.expectEqual(@as(u32, 4), seen.at); // the first turn's cut, so only the new tokens are prefilled
    try t.expectEqual(@as(?usize, 5), next.tokens);
    try t.expect(next.why == null);
}

test "handoff: a gridded store keeps its grid state before the cut, not the cut, and the next turn resumes on the grid" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0, .planned = true, .grid = 3 }, 1 << 20);
    defer store.deinit();
    var peer: Peer = .{ .at = 0 };
    var first: Exp = .{};
    const one = try runWith(&peer, &.{ .prompt = six[0..5], .max_tokens = 1, .handoff = id }, &first, &store);
    try t.expectEqual(@as(?usize, 4), first.tokens); // exported at the cut
    try t.expectEqual(@as(usize, 2), one.marks); // the grid's 3, then the cut's 4
    try t.expectEqual(@as(u64, 1), store.counts.kept); // 4 is off the grid: no state the store could never resume
    var next: Exp = .{};
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &next, &store);
    try t.expect(seen.restoring);
    try t.expectEqual(@as(u32, 3), seen.at);
    try t.expectEqual(@as(?usize, 5), next.tokens);
}

/// Every lookup the passes made was reported: no kept state is still borrowed.
pub fn unborrowed(store: *const pc.Store) !void {
    for (store.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers);
}

test "handoff: the same handoff again on a gridded store resumes at the grid row before its cut and exports at the cut" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0, .planned = true, .grid = 3 }, 1 << 20);
    defer store.deinit();
    const seven = six ++ [_]u32{17}; // its cut, 6, is on the grid
    var peer: Peer = .{ .at = 0 };
    var first: Exp = .{};
    _ = try runWith(&peer, &.{ .prompt = &seven, .max_tokens = 1, .handoff = id }, &first, &store);
    try t.expectEqual(@as(u64, 2), store.counts.kept); // the cut, and the grid row before it
    var again: Exp = .{};
    const seen = try runWith(&peer, &.{ .prompt = &seven, .max_tokens = 1, .handoff = id }, &again, &store);
    try t.expect(seen.restoring);
    try t.expectEqual(@as(u32, 3), seen.at); // the kept cut restores nothing to export, and 0 would prefill it all
    try t.expectEqual(@as(?usize, 6), again.tokens);
    try t.expect(again.why == null);
    try unborrowed(&store);
}

test "handoff: the same handoff again on an unplanned store resumes at the longest state kept before its cut" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 2, null, &.{}, &.{}));
    const request: api.Request = .{ .prompt = &six, .max_tokens = 1, .history_len = 3, .handoff = id };
    var peer: Peer = .{ .at = 0 };
    var first: Exp = .{};
    _ = try runWith(&peer, &request, &first, &store);
    try t.expectEqual(@as(u64, 3), store.counts.kept); // 2, then the history's 3 and the cut, 5
    var again: Exp = .{};
    const seen = try runWith(&peer, &request, &again, &store);
    try t.expect(seen.restoring);
    try t.expectEqual(@as(u32, 3), seen.at); // not the cut, not 2 and not 0
    try t.expectEqual(@as(?usize, 5), again.tokens);
    try t.expect(again.why == null);
    try unborrowed(&store);
}

/// Kept's copies as one row a token, on the disk of a store that keeps them there (--learn-conversations).
const OnDisk = struct {
    fn bytesOf(saved: pc.Saved) []u8 {
        return std.mem.sliceAsBytes(@as(*std.ArrayList(u32), @ptrCast(@alignCast(saved))).items);
    }
    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(pc.rows.image.Part), a: std.mem.Allocator) anyerror!void {
        try out.append(a, .{ .bytes = @as(u64, at) * 4, .row = 4 });
    }
    fn gather(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, out: []u8) anyerror!void {
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(out[o..][0..@intCast(r.len)], bytesOf(saved)[@intCast(r.off)..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    fn blank(_: *anyopaque, at: u32) anyerror!pc.Saved {
        const kept = try t.allocator.create(std.ArrayList(u32));
        errdefer t.allocator.destroy(kept);
        kept.* = .empty;
        try kept.appendNTimes(t.allocator, 0, at);
        return kept;
    }
    fn scatter(_: *anyopaque, saved: pc.Saved, ranges: []const pc.rows.image.Range, in: []const u8) anyerror!void {
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(bytesOf(saved)[@intCast(r.off)..][0..@intCast(r.len)], in[o..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    const vtable: pc.Snapshots.VTable = .{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop, .rows = &.{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter } };
};

test "handoff: the same handoff on a restarted producer reads the grid row before its cut back from disk and resumes there" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    var none: u8 = 0;
    const family: pc.Snapshots = .{ .ptr = &none, .vtable = &OnDisk.vtable };
    const seven = six ++ [_]u32{17}; // its cut, 6, is on the grid
    var peer: Peer = .{ .at = 0 };
    for (0..2) |turn| { // the second server starts with nothing in RAM
        const disk = try pc.rows.Rows.open(t.allocator, family, .{ .root = root, .identity = 5, .cap = 1 << 30, .floor = 0, .daily = 1 << 30, .planned = true, .grid = 3 });
        defer disk.deinit();
        var store = pc.Store.init(t.allocator, family, .{ .min_prompt = 0, .planned = true, .grid = 3 }, 1 << 20);
        defer store.deinit();
        store.disk = disk;
        var e: Exp = .{};
        const seen = try runWith(&peer, &.{ .prompt = &seven, .max_tokens = 1, .handoff = id }, &e, &store);
        try t.expectEqual(@as(?usize, 6), e.tokens);
        try t.expect(e.why == null);
        if (turn == 0) continue; // its host's stop wrote the grid row, 3, and the cut, 6
        try t.expect(seen.restoring);
        try t.expectEqual(@as(u32, 3), seen.at); // the lookup read back the cut, which restores nothing to export
        try t.expectEqual(@as(u64, 2), disk.counts.reads);
        try unborrowed(&store);
    }
}

test "remote prefix kept (TF_PREFILL_KEEP): its pass keeps remote states; the next turn resumes one, counted cached, reported remote, and keeps remote states" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // a chat's history ends at 5: the store marks it, past the 4 pulled tokens
    const first = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true });
    try t.expect(first.restoring and first.remote and first.hooked);
    try t.expectEqual(@as(u32, 4), first.at);
    try t.expectEqual(@as(u32, 1), peer.fills);
    try t.expectEqual(@as(?u32, 0), first.prefilled); // a fresh pull is not the cache's
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire)); // its storage went once, at the first mark
    try t.expectEqual(@as(u64, 1), store.counts.kept);
    try t.expect(store.entries.items[0].remote and store.entries.items[0].at == 5); // kept past the prefix, marked remote
    // the chat's next turn: its history ends at 7, past the kept state it resumes
    const next = try runKeep(&peer, &.{ .prompt = &eight, .max_tokens = 2, .history_len = 7 }, null, &store, .{ .keep = true });
    try t.expect(next.restoring and next.remote); // a kept remote state: restored here, still remote
    try t.expectEqual(@as(u32, 5), next.at);
    try t.expectEqual(@as(u32, 1), peer.fills); // the cache's state beat a new fill
    try t.expectEqual(@as(?u32, 5), next.prefilled); // kept here, so counted as cached
    try t.expectEqual(@as(?u32, 5), next.remote_prefix); // and still reported as a remote prefix, never as exact
    try t.expectEqual(@as(u32, 1), peer.drops.load(.acquire));
    try t.expectEqual(@as(u64, 2), store.counts.kept);
    const later = store.find(&eight, &.{}, &.{}).?;
    try t.expect(later.at == 7 and later.remote); // nothing pulled this turn, yet what it keeps stands on a remote state
}

test "kept remote states: an exact keep hides the remote copy, and a remote keep rides an exact one" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    store.keep_remote = false;
    try t.expect(store.find(&six, &.{}, &.{}).?.remote);
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    try t.expect(!store.find(&six, &.{}, &.{}).?.remote); // the engine's own copy is the one a lookup finds
    try t.expectEqual(@as(u64, 2), store.counts.kept);
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{})); // the exact copy serves a remote keep: nothing new
    store.keep_remote = false;
    try t.expectEqual(@as(u64, 2), store.counts.kept);
}

test "remote prefix kept: a warm pass resuming a kept remote state keeps its end remote and off disk, and the next turn is remote" {
    var buf: [64]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-remote-warm-{d}", .{platform.c.getpid()});
    defer rmTree(root);
    var none: u8 = 0;
    const family: pc.Snapshots = .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop, .rows = &.{ .parts = Kept.parts, .gather = Kept.gather, .blank = Kept.blank, .scatter = Kept.scatter } } };
    const disk = try pc.rows.Rows.open(t.allocator, family, .{ .root = root, .identity = 2, .cap = 1 << 30, .floor = 0, .daily = 1 << 30 });
    defer disk.deinit();
    var store = pc.Store.init(t.allocator, family, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.disk = disk;
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{})); // a pulled turn's history state
    store.keep_remote = false;
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // the server's warm pass after a reply: background, no tokens, its history the whole prompt
    const warm = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 0, .background = true, .history_len = 6 }, null, &store, .{ .keep = true, .warm = true });
    try t.expect(warm.restoring and warm.remote and warm.at == 4);
    const end = store.find(&eight, &.{}, &.{}).?;
    try t.expect(end.at == 6 and end.remote); // the end it kept stands on the remote state: remote too
    try t.expectEqual(@as(usize, 0), disk.states.items.len); // nothing written, though the host's stop writes what waits
    const next = try runKeep(&peer, &.{ .prompt = &eight, .max_tokens = 2, .history_len = 7 }, null, &store, .{ .keep = true, .warm = true });
    try t.expect(next.restoring and next.remote and next.at == 6);
    try t.expectEqual(@as(?u32, 6), next.remote_prefix); // reported as a remote prefix, never as exact
    try t.expectEqual(@as(u32, 0), peer.fills);
}

test "remote prefix kept: a pull past a shorter kept state releases the lookup's borrow, and keeps only marks past the pull" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 2, null, &.{}, &.{})); // shorter than the pull: the lookup borrows it, the pull beats it
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    // a history ending where the pull does: the pass never stands there, so no mark is left to keep
    const at_pull = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 4 }, null, &store, .{ .keep = true });
    try t.expect(at_pull.restoring and at_pull.remote and at_pull.hooked and at_pull.marks == 0);
    try t.expectEqual(@as(u64, 1), store.counts.kept);
    const beyond = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true });
    try t.expect(beyond.restoring and beyond.remote and beyond.at == 4 and beyond.marks == 1);
    try t.expectEqual(@as(u32, 2), peer.fills);
    for (store.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers); // else the shorter state is never evicted
    try t.expect(store.find(&six, &.{}, &.{}).?.remote); // the pass kept 5, marked remote
}

/// Slot 0's own resume point, as a backend standing `at` tokens into every prompt reports it.
pub const Slot = struct {
    at: u32,
    pub fn peek(ctx: *anyopaque, _: []const u32, _: []const [2]u32) u32 {
        const s: *Slot = @ptrCast(@alignCast(ctx));
        return s.at;
    }
};

test "remote prefix kept: slot 0's own state as far as a kept remote state wins over it, and the pass keeps exact states" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    store.keep_remote = true;
    try t.expect(store.keep(&six, 4, null, &.{}, &.{}));
    store.keep_remote = false;
    var peer: Peer = .{ .at = 4 };
    defer peer.kept.deinit(t.allocator);
    var slot: Slot = .{ .at = 3 };
    const peek: remote.ResumePeek = .{ .ctx = &slot, .at = Slot.peek };
    const short = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 4 }, null, &store, .{ .keep = true, .peek = peek });
    try t.expect(short.restoring and short.remote and short.at == 4); // slot 0 short of it: the remote state is restored
    slot.at = 4;
    const level = try runKeep(&peer, &.{ .prompt = &six, .max_tokens = 2, .history_len = 5 }, null, &store, .{ .keep = true, .peek = peek });
    try t.expect(!level.restoring and !level.remote and level.marks == 1); // as far: slot 0's own exact state goes on
    try t.expectEqual(@as(u32, 0), peer.fills);
    for (store.entries.items) |e| try t.expectEqual(@as(u32, 0), e.borrowers); // the remote state was let go
    try t.expect(!store.find(&six, &.{}, &.{}).?.remote); // and the mark past it is kept exact
}

test "handoff: the state a repeat resumes below its cut brings its own provenance, not the cut's" {
    var none: u8 = 0;
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Kept.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    try t.expect(store.keep(&six, 3, null, &.{}, &.{})); // the engine's own, before the cut
    store.keep_remote = true;
    try t.expect(store.keep(&six, 5, null, &.{}, &.{})); // a remote copy at the cut, which the lookup finds first
    store.keep_remote = false;
    var peer: Peer = .{ .at = 0 };
    var e: Exp = .{};
    const seen = try runWith(&peer, &.{ .prompt = &six, .max_tokens = 1, .handoff = id }, &e, &store);
    try t.expect(seen.restoring and !seen.remote); // resumed at 3, the engine's own state
    try t.expectEqual(@as(u32, 3), seen.at);
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expect(!store.find(&six, &.{}, &.{}).?.remote); // the pass kept its cut as its own, hiding the remote copy
    try unborrowed(&store);
}
