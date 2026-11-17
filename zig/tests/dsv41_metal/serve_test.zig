//! The tiny model served on Metal through serve_lanes and LaneHost: replies equal the target's own serial decode.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const ff = @import("fixture_fwd.zig");

const api = host.api;
const Served = host.serve_lanes.Served;
const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = 71;

test {
    _ = @import("boot_test.zig");
    _ = @import("tree_test.zig");
    _ = @import("steps_test.zig");
}

/// Replies that have shown their first tokens so far (Box.first).
var firsts: std.atomic.Value(u64) = .init(0);

/// A reply's tokens and finish, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    reason: ?api.Reason = null,
    drafted: u64 = 0,
    cached: u32 = 0,
    /// when the reply's first tokens came, in the order of every box's (0: none yet)
    first: u64 = 0,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| {
                if (b.first == 0) b.first = firsts.fetchAdd(1, .monotonic) + 1;
                b.out.appendSlice(gpa, ts) catch {};
            },
            .finished => |fin| {
                b.done = true;
                b.reason = fin.reason;
                b.drafted = fin.stats.drafted;
            },
            .prefilled => |n| b.cached = n,
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..600_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(b: *Box) void {
        b.out.deinit(gpa);
    }
};

/// One request to send: its prompt, length, drafts and sampling.
const Ask = struct { prompt: []const u32, max: u32 = 24, drafts: bool = true, sampling: ?api.Sampling = null };

var next_id: api.Id = 1;
/// The prompt rows each reply of the last `serve` call took from kept state (its `prefilled` event), in ask order.
var cached_rows: [8]u32 = undefined;

/// The asks sent together (or each after the last finished); each reply returned gpa-owned, the drafted rows summed.
fn serve(e: api.Engine, asks: []const Ask, together: bool, out: [][]u32, drafted: *u64) !void {
    const boxes = try gpa.alloc(Box, asks.len);
    defer gpa.free(boxes);
    @memset(boxes, .{});
    defer for (boxes) |*b| b.deinit();
    const reqs = try gpa.alloc(api.Request, asks.len);
    defer gpa.free(reqs);
    for (reqs, asks) |*r, k| r.* = .{ .prompt = k.prompt, .max_tokens = k.max, .drafts = k.drafts, .sampling = k.sampling };
    for (reqs, boxes) |*r, *b| {
        next_id += 1;
        try e.submit(next_id, r, .{ .ctx = b, .event = Box.event });
        if (!together) try b.wait();
    }
    for (boxes, out, asks, 0..) |*b, *o, k, i| {
        try b.wait();
        try testing.expectEqual(api.Reason.length, b.reason.?);
        try testing.expectEqual(@as(usize, k.max), b.out.items.len);
        drafted.* += b.drafted;
        cached_rows[i] = b.cached;
        o.* = try b.out.toOwnedSlice(gpa);
    }
}

fn freeAll(xs: [][]u32) void {
    for (xs) |x| gpa.free(x);
}

/// The reference: the target's serial decode of each ask on a model of its own (no lanes, no cache, no drafts).
fn reference(asks: []const Ask, out: [][]u32) !void {
    const m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(m);
    for (asks, out) |k, *o| o.* = try m.t.serial(0, k.prompt, k.max, k.sampling);
}

fn ids(n: usize, s: u64) ![]u32 {
    return ff.ids(gpa, &fs.config(), n, s);
}

/// Ids with no user, assistant or think tag among them, so the tests place those themselves.
fn untagged(n: usize, s: u64) ![]u32 {
    const c = fs.config();
    const xs = try ids(n, s);
    for (xs) |*x| if (x.* == c.user or x.* == c.assistant or x.* == c.think_open) {
        x.* = 7;
    };
    return xs;
}

fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal serve: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

fn same(want: []const []u32, got: []const []u32) usize {
    var bad: usize = 0;
    for (want, got) |w, g| bad += @intFromBool(!std.mem.eql(u32, w, g));
    return bad;
}

test "dsv41 metal serve: (a) concurrent == solo == serial, (b) drafted == draft:false, through LaneHost" {
    const p0 = try ids(150, 1);
    defer gpa.free(p0);
    const p1 = try ids(620, 2);
    defer gpa.free(p1);
    const asks = [_]Ask{ .{ .prompt = p0 }, .{ .prompt = p1 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{});
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var solo: [2][]u32 = undefined;
    try serve(e, &asks, false, &solo, &drafted);
    defer freeAll(&solo);
    var both: [2][]u32 = undefined;
    try serve(e, &asks, true, &both, &drafted);
    defer freeAll(&both);
    const plain_asks = [_]Ask{ .{ .prompt = p0, .drafts = false }, .{ .prompt = p1, .drafts = false } };
    var plain_drafted: u64 = 0;
    var plain: [2][]u32 = undefined;
    try serve(e, &plain_asks, true, &plain, &plain_drafted);
    defer freeAll(&plain);
    const bad = [3]usize{ same(&want, &solo), same(&want, &both), same(&want, &plain) };
    say(bad[0] == 0, "(a) solo vs serial: {d} of 2 replies differ", .{bad[0]});
    say(bad[1] == 0, "(a) together vs serial: {d} of 2 replies differ ({d} windows of {d} rows in all)", .{ bad[1], m.t.stats.windows, m.t.stats.rows });
    say(bad[2] == 0 and drafted > 0 and plain_drafted == 0, "(b) draft:false vs serial: {d} differ; drafted rows {d} (drafts on), {d} (off), {d} pass launches, {d} windows cut back", .{ bad[2], drafted, plain_drafted, m.ad.passes, m.t.stats.rewinds });
    try testing.expectEqual([3]usize{ 0, 0, 0 }, bad);
    try testing.expect(drafted > 0 and m.t.stats.rewinds > 0);
    try testing.expectEqual(@as(u64, 0), plain_drafted);
}

test "dsv41 metal serve: (c) prompt-cache hits (the same prompt, a later turn) == fresh runs" {
    const p = try ids(300, 3);
    defer gpa.free(p);
    const m = try fs.open(gpa, io, seed, .{ .cache_bytes = 1 << 30 });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var first: [1][]u32 = undefined;
    try serve(e, &.{.{ .prompt = p, .max = 40 }}, false, &first, &drafted);
    defer freeAll(&first);
    const first_cached = cached_rows[0];
    // a later turn: the prompt, the reply but its last token, new ids; then the first prompt again
    const extra = try ids(50, 4);
    defer gpa.free(extra);
    const turn = try std.mem.concat(gpa, u32, &.{ p, first[0][0..39], extra });
    defer gpa.free(turn);
    const branch = try std.mem.concat(gpa, u32, &.{ p, extra });
    defer gpa.free(branch);
    const asks = [_]Ask{ .{ .prompt = turn, .max = 30 }, .{ .prompt = p, .max = 40 }, .{ .prompt = branch, .max = 30 } };
    const hits0 = m.t.stats.hits;
    var got: [3][]u32 = undefined;
    try serve(e, &asks, false, &got, &drafted);
    defer freeAll(&got);
    const hits = m.t.stats.hits - hits0;
    const reported = cached_rows[0..3].*;
    var want: [3][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const bad = same(&want, &got);
    say(bad == 0 and hits == 3, "(c) 3 cached prompts ({d} hits, {d} rows restored in all) vs fresh: {d} replies differ", .{ hits, m.t.stats.cached_rows, bad });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expectEqual(@as(u64, 3), hits);
    // each reply reports the rows it resumed (prompt_tokens_details.cached_tokens)
    say(reported[0] > p.len and reported[1] >= p.len and reported[2] >= p.len, "(c) cached rows reported {any} for prompts of {d}, {d} and {d} tokens", .{ reported, turn.len, p.len, branch.len });
    for (reported, [3]usize{ turn.len, p.len, branch.len }) |r, n| try testing.expect(r > 0 and r <= n);
    try testing.expect(reported[0] > p.len and reported[1] >= p.len and reported[2] >= p.len);
    try testing.expectEqual(@as(u32, 0), first_cached);
    // the later turn resumed past the first prompt: its reply's rows came from the cache too
    try testing.expect(m.t.stats.cached_rows >= 3 * p.len + 30);
}

test "dsv41 metal serve: (c2) a thinking turn keeps its state before <think>; the next turn resumes there == fresh" {
    for ([_]bool{ false, true }) |steps| try thinkingTurn(steps);
}

/// (c2) with whole prompts, or in steps (`steps`): the same kept state and replies.
fn thinkingTurn(steps: bool) !void {
    const c = fs.config();
    const body = try untagged(300, 7);
    defer gpa.free(body);
    const p = try std.mem.concat(gpa, u32, &.{ body, &.{ c.assistant, c.think_open } });
    defer gpa.free(p);
    const m = try fs.open(gpa, io, seed, .{ .cache_bytes = 1 << 30, .steps = steps });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var first: [1][]u32 = undefined;
    try serve(e, &.{.{ .prompt = p, .max = 30 }}, false, &first, &drafted);
    defer freeAll(&first);
    // the next turn as the template renders it: the tag closed (a stand-in id), the reply, new ids, a thinking header
    const extra = try untagged(40, 8);
    defer gpa.free(extra);
    const turn = try std.mem.concat(gpa, u32, &.{ p[0 .. p.len - 1], &.{c.think_open + 1}, first[0], extra, &.{ c.assistant, c.think_open } });
    defer gpa.free(turn);
    const hits0 = m.t.stats.hits;
    var got: [1][]u32 = undefined;
    try serve(e, &.{.{ .prompt = turn, .max = 30 }}, false, &got, &drafted);
    defer freeAll(&got);
    const resumed = cached_rows[0];
    const asks = [_]Ask{ .{ .prompt = p, .max = 30 }, .{ .prompt = turn, .max = 30 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const bad = same(&want, &.{ first[0], got[0] });
    say(bad == 0 and resumed == p.len - 1, "(c2) {s}: a thinking follow-up resumed {d} rows (its turn before <think>: {d}); split prefill and follow-up vs fresh: {d} of 2 replies differ", .{ if (steps) "in steps" else "whole prompts", resumed, p.len - 1, bad });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(m.t.stats.hits > hits0);
    try testing.expectEqual(@as(u32, @intCast(p.len - 1)), resumed);
}

test "dsv41 metal serve: (c3) a fresh conversation resumes after the last one's system part == fresh" {
    for ([_]bool{ false, true }) |steps| try systemPart(steps);
}

/// (c3) with whole prompts, or in steps (`steps`): the same kept state and replies.
fn systemPart(steps: bool) !void {
    const c = fs.config();
    const system = try untagged(300, 9);
    defer gpa.free(system);
    var msgs: [2][]u32 = undefined;
    for (&msgs, 0..) |*mm, i| mm.* = try untagged(20 + 5 * i, 10 + i);
    defer for (msgs) |mm| gpa.free(mm);
    var asks: [2]Ask = undefined;
    var prompts: [2][]u32 = undefined;
    for (&prompts, &asks, msgs) |*p, *k, mm| {
        p.* = try std.mem.concat(gpa, u32, &.{ system, &.{c.user}, mm, &.{ c.assistant, c.think_open } });
        k.* = .{ .prompt = p.*, .max = 24 };
    }
    defer for (prompts) |p| gpa.free(p);
    const m = try fs.open(gpa, io, seed, .{ .cache_bytes = 1 << 30, .steps = steps });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    var drafted: u64 = 0;
    var got: [2][]u32 = undefined;
    try serve(s.engine(), &asks, false, &got, &drafted);
    defer freeAll(&got);
    const resumed = cached_rows[0..2].*;
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const bad = same(&want, &got);
    say(bad == 0 and resumed[1] == system.len, "(c3) {s}: a new conversation resumed {d} rows (the system part: {d}); vs fresh: {d} of 2 replies differ", .{ if (steps) "in steps" else "whole prompts", resumed[1], system.len, bad });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expectEqual([2]u32{ 0, @intCast(system.len) }, resumed);
}

test "dsv41 metal serve: (d) sampled replies are their seed's: again, drafted or not, together, serial" {
    const p0 = try ids(200, 5);
    defer gpa.free(p0);
    const p1 = try ids(90, 6);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 1234, .temperature = 0.9, .top_k = 0, .top_p = 1.0 };
    const nucleus: api.Sampling = .{ .seed = 99, .temperature = 0.7, .top_k = 40, .top_p = 0.9 };
    const asks = [_]Ask{ .{ .prompt = p0, .sampling = hot }, .{ .prompt = p1, .sampling = nucleus } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{});
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var a1: [2][]u32 = undefined;
    try serve(e, &asks, true, &a1, &drafted);
    defer freeAll(&a1);
    var a2: [2][]u32 = undefined;
    try serve(e, &asks, false, &a2, &drafted);
    defer freeAll(&a2);
    const plain_asks = [_]Ask{ .{ .prompt = p0, .sampling = hot, .drafts = false }, .{ .prompt = p1, .sampling = nucleus, .drafts = false } };
    var a3: [2][]u32 = undefined;
    try serve(e, &plain_asks, true, &a3, &drafted);
    defer freeAll(&a3);
    var other_seed = hot;
    other_seed.seed = 1235;
    var a4: [1][]u32 = undefined;
    try serve(e, &.{.{ .prompt = p0, .sampling = other_seed }}, false, &a4, &drafted);
    defer freeAll(&a4);
    const bad = [3]usize{ same(&want, &a1), same(&want, &a2), same(&want, &a3) };
    const moved = !std.mem.eql(u32, a4[0], want[0]);
    say(bad[0] + bad[1] + bad[2] == 0 and moved, "(d) together / again / draft:false vs serial: {d} / {d} / {d} differ; another seed's reply {s}", .{ bad[0], bad[1], bad[2], if (moved) "differs" else "is the same" });
    try testing.expectEqual([3]usize{ 0, 0, 0 }, bad);
    try testing.expect(moved);
}

test "dsv41 metal serve: (e) a request cancelled during its prompt leaves the slot and the next request valid" {
    const long = try ids(1100, 7);
    defer gpa.free(long);
    const p0 = try ids(120, 8);
    defer gpa.free(p0);
    const p1 = try ids(140, 9);
    defer gpa.free(p1);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = p1 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .slots = 2, .cache_bytes = 1 << 30 });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    // a decoding stream, then the long prompt (cancelled as soon as it is submitted), then a new request
    var bg: Box = .{};
    defer bg.deinit();
    const r0: api.Request = .{ .prompt = p0, .max_tokens = 40 };
    try e.submit(500, &r0, .{ .ctx = &bg, .event = Box.event });
    var gone: Box = .{};
    defer gone.deinit();
    const rl: api.Request = .{ .prompt = long, .max_tokens = 64 };
    try e.submit(501, &rl, .{ .ctx = &gone, .event = Box.event });
    e.cancel(501);
    try gone.wait();
    var drafted: u64 = 0;
    var after: [1][]u32 = undefined;
    try serve(e, asks[1..], false, &after, &drafted);
    defer freeAll(&after);
    try bg.wait();
    const ok0 = std.mem.eql(u32, want[0], bg.out.items);
    const ok1 = std.mem.eql(u32, want[1], after[0]);
    say(ok0 and ok1, "(e) cancelled ({t}, {d} tokens out): the decoding reply {s}, the next request's {s} the serial one", .{ gone.reason.?, gone.out.items.len, if (ok0) "equals" else "differs from", if (ok1) "equals" else "differs from" });
    try testing.expect(gone.reason.? == .cancelled);
    try testing.expect(ok0 and ok1);
}

test "dsv41 metal serve: (f) plain rounds at four streams, drafts again at two from the target's history == serial" {
    var ps: [4][]u32 = undefined;
    for (&ps, 0..) |*p, i| p.* = try ids(90 + 40 * i, 20 + i);
    defer for (ps) |p| gpa.free(p);
    const asks = [_]Ask{ .{ .prompt = ps[0], .max = 12 }, .{ .prompt = ps[1], .max = 12 }, .{ .prompt = ps[2], .max = 60 }, .{ .prompt = ps[3], .max = 60 } };
    var want: [4][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{});
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    var drafted: u64 = 0;
    var got: [4][]u32 = undefined;
    try serve(s.engine(), &asks, true, &got, &drafted);
    defer freeAll(&got);
    const bad = same(&want, &got);
    const caught = m.t.stats.history_rows;
    say(bad == 0 and caught > 0 and drafted > 0, "(f) 4 together vs serial: {d} differ; {d} rows caught up from history, {d} drafted rows, {d} ingests", .{ bad, caught, drafted, m.ad.ingests });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(caught > 0 and drafted > 0);
}

test "dsv41 metal serve: (g) draft trees (first-position siblings) reply in full, drafted == draft:false == serial" {
    const p0 = try ids(160, 30);
    defer gpa.free(p0);
    const p1 = try ids(240, 31);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 4321, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
    const asks = [_]Ask{ .{ .prompt = p0, .max = 96 }, .{ .prompt = p1, .max = 96, .sampling = hot } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    for ([_]u32{ 1, 2 }) |sibs| {
        const m = try fs.open(gpa, io, seed, .{ .slots = 1 });
        var md = m.model();
        md.trees = .{ .siblings = sibs, .sib_rows = 4, .dup = .{ .ms = 0.0, .rows = true } };
        const s = try Served.open(gpa, io, md, ".", null);
        defer Served.close(s);
        var drafted: u64 = 0;
        var got: [2][]u32 = undefined;
        try serve(s.engine(), &asks, false, &got, &drafted);
        defer freeAll(&got);
        var plain_drafted: u64 = 0;
        var plain: [2][]u32 = undefined;
        try serve(s.engine(), &.{ .{ .prompt = p0, .max = 96, .drafts = false }, .{ .prompt = p1, .max = 96, .sampling = hot, .drafts = false } }, false, &plain, &plain_drafted);
        defer freeAll(&plain);
        const bad = [2]usize{ same(&want, &got), same(&want, &plain) };
        const pol = s.x.policy;
        say(bad[0] + bad[1] == 0 and pol.trees > 0, "(g) {d} siblings: drafted / draft:false vs serial: {d} / {d} differ; {d} tree rounds, {d} sibling wins, {d} drafted rows", .{ sibs, bad[0], bad[1], pol.trees, pol.sib_wins, drafted });
        try testing.expectEqual([2]usize{ 0, 0 }, bad);
        try testing.expect(pol.trees > 0 and drafted > 0);
    }
}

test "dsv41 metal serve: (h) two streams with draft trees on two slots, overlapping: each == draft:false == serial" {
    const p1 = try ids(240, 31);
    defer gpa.free(p1);
    const hot: api.Sampling = .{ .seed = 4321, .temperature = 0.8, .top_k = 0, .top_p = 1.0 };
    // one request on both slots: the first stops soon after the second starts, which then decodes mostly alone
    const asks = [_]Ask{ .{ .prompt = p1, .max = 72, .sampling = hot }, .{ .prompt = p1, .max = 128, .sampling = hot } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .slots = 2 });
    var md = m.model();
    md.trees = .{ .siblings = 1, .sib_rows = 4, .dup = .{ .ms = 0.0, .rows = true } };
    const s = try Served.open(gpa, io, md, ".", null);
    defer Served.close(s);
    const e = s.engine();
    // the second request arrives once the first has 64 tokens: each stream decodes alone and beside the other
    var boxes = [2]Box{ .{}, .{} };
    defer for (&boxes) |*b| b.deinit();
    var reqs: [2]api.Request = undefined;
    for (&reqs, asks) |*r, k| r.* = .{ .prompt = k.prompt, .max_tokens = k.max, .sampling = k.sampling };
    try e.submit(900, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
    for (0..600_000) |_| {
        boxes[0].mutex.lockUncancelable(io);
        const n = boxes[0].out.items.len;
        boxes[0].mutex.unlock(io);
        if (n >= 64) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try e.submit(901, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
    for (&boxes) |*b| try b.wait();
    var plain_drafted: u64 = 0;
    var plain: [2][]u32 = undefined;
    try serve(e, &.{ .{ .prompt = p1, .max = 72, .sampling = hot, .drafts = false }, .{ .prompt = p1, .max = 128, .sampling = hot, .drafts = false } }, true, &plain, &plain_drafted);
    defer freeAll(&plain);
    var bad: usize = 0;
    for (boxes, want, plain, asks) |b, w, p, k| {
        try testing.expectEqual(api.Reason.length, b.reason.?);
        try testing.expectEqual(@as(usize, k.max), b.out.items.len);
        bad += @intFromBool(!std.mem.eql(u32, w, b.out.items)) + @intFromBool(!std.mem.eql(u32, w, p));
    }
    const pol = s.x.policy;
    const st = m.t.tree.stats;
    const each = st.by_slot[0] > 0 and st.by_slot[1] > 0;
    say(bad == 0 and each, "(h) two streams, trees on: {d} of 4 replies differ from serial; {d} + {d} tree windows by slot, {d} sibling wins, {d} adopted, {d} drafted rows", .{ bad, st.by_slot[0], st.by_slot[1], pol.sib_wins, st.adopted, boxes[0].drafted + boxes[1].drafted });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(each and boxes[0].drafted > 0 and boxes[1].drafted > 0);
}

test "dsv41 metal serve: (j) with a residency owner the server gets an idle ticker that keeps the set wired; none without" {
    {
        const m = try fs.open(gpa, io, seed, .{ .resident = true });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const k = s.engine().keepaliveTarget() orelse return error.NoKeepalive;
        for (0..3) |_| k.tick(k.ctx); // the idle ticker's tick: waiting joins committed, a tiny blit through the set
        var drafted: u64 = 0;
        var got: [1][]u32 = undefined;
        try serve(s.engine(), &.{.{ .prompt = &.{ 11, 12, 13, 14 }, .max = 8 }}, false, &got, &drafted);
        defer freeAll(&got);
        k.tick(k.ctx); // and again after a request
        say(got[0].len == 8, "(j) idle ticker on a model with a residency owner: 4 ticks, a reply of {d} tokens between them", .{got[0].len});
        try testing.expectEqual(@as(usize, 8), got[0].len);
    }
    const m = try fs.open(gpa, io, seed, .{});
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    try testing.expect(s.engine().keepaliveTarget() == null);
}

/// Waits until `field` of the target's stats reaches `n` (read from LaneHost's thread's writes).
fn awaitStat(field: *const u64, n: u64) !void {
    for (0..60_000) |_| {
        if (@atomicLoad(u64, field, .monotonic) >= n) return;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return error.Timeout;
}

test "dsv41 metal serve: (g) prompts in steps: concurrent == solo == serial, drafted == draft:false, pieces between rounds" {
    const p0 = try ids(150, 1);
    defer gpa.free(p0);
    const p1 = try ids(620, 2);
    defer gpa.free(p1);
    const asks = [_]Ask{ .{ .prompt = p0 }, .{ .prompt = p1 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .steps = true });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var drafted: u64 = 0;
    var solo: [2][]u32 = undefined;
    try serve(e, &asks, false, &solo, &drafted);
    defer freeAll(&solo);
    var both: [2][]u32 = undefined;
    try serve(e, &asks, true, &both, &drafted);
    defer freeAll(&both);
    const plain_asks = [_]Ask{ .{ .prompt = p0, .drafts = false }, .{ .prompt = p1, .drafts = false } };
    var plain_drafted: u64 = 0;
    var plain: [2][]u32 = undefined;
    try serve(e, &plain_asks, true, &plain, &plain_drafted);
    defer freeAll(&plain);
    const bad = same(&want, &solo) + same(&want, &both) + same(&want, &plain);
    say(bad == 0 and m.t.stats.pieces > 0 and drafted > 0, "(g) in steps: {d} of 6 replies differ from serial; {d} pieces, {d} drafted rows", .{ bad, m.t.stats.pieces, drafted });
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expect(m.t.stats.pieces > 0 and drafted > 0 and plain_drafted == 0);
}

test "dsv41 metal serve: (h) a short request during a long prompt: first in steps, after it as whole prompts" {
    const long = try ids(1100, 30);
    defer gpa.free(long);
    const short = try ids(100, 31);
    defer gpa.free(short);
    const asks = [_]Ask{ .{ .prompt = long, .max = 4 }, .{ .prompt = short, .max = 4 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var bad: usize = 0;
    for ([_]bool{ false, true }) |steps| {
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 16, .steps = steps });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = long, .max_tokens = 4 }, .{ .prompt = short, .max_tokens = 4 } };
        try e.submit(600, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        // the short request arrives while the long prompt runs: two of its pieces in, or (whole) once it began
        if (steps) try awaitStat(&m.t.stats.pieces, 2) else try awaitStat(&m.t.stats.prompts, 1);
        try e.submit(601, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const short_first = boxes[1].first < boxes[0].first;
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const ok = equal and short_first == steps;
        say(ok, "(h) {s}: the short reply's first token came {s} the long one's ({d} pieces); replies {s} serial", .{ if (steps) "in steps" else "whole prompts", if (short_first) "before" else "after", m.t.stats.pieces, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal serve: (i) in steps, a request cancelled mid-prompt keeps nothing and the next requests == serial" {
    // no user or think tag, so no state short of the prompt is kept (Target.keepPoints)
    const long = try untagged(1100, 32);
    defer gpa.free(long);
    const p0 = try ids(120, 33);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 40 }, .{ .prompt = long } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 16, .cache_bytes = 1 << 30, .steps = true });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var bg: Box = .{};
    defer bg.deinit();
    const r0: api.Request = .{ .prompt = p0, .max_tokens = 40 };
    try e.submit(700, &r0, .{ .ctx = &bg, .event = Box.event });
    // the long prompt once the decoding reply is out, cancelled three of its pieces in
    for (0..60_000) |_| {
        bg.mutex.lockUncancelable(io);
        const out = bg.first != 0;
        bg.mutex.unlock(io);
        if (out) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    } else return error.Timeout;
    var gone: Box = .{};
    defer gone.deinit();
    const rl: api.Request = .{ .prompt = long, .max_tokens = 64 };
    const p_before = @atomicLoad(u64, &m.t.stats.pieces, .monotonic);
    try e.submit(701, &rl, .{ .ctx = &gone, .event = Box.event });
    try awaitStat(&m.t.stats.pieces, p_before + 3);
    e.cancel(701);
    try gone.wait();
    try bg.wait();
    var prefix_kept = false;
    for (m.t.cache.entries.items) |x| prefix_kept = prefix_kept or std.mem.startsWith(u32, long, x.ids);
    var drafted: u64 = 0;
    var again: [1][]u32 = undefined;
    try serve(e, asks[1..], false, &again, &drafted);
    defer freeAll(&again);
    const ok = gone.reason.? == .cancelled and m.t.stats.abandoned == 1 and std.mem.eql(u32, want[0], bg.out.items) and std.mem.eql(u32, want[1], again[0]);
    if (!ok) std.debug.print("     reason {t}, abandoned {d}, decoding reply {}, again {}\n", .{ gone.reason.?, m.t.stats.abandoned, std.mem.eql(u32, want[0], bg.out.items), std.mem.eql(u32, want[1], again[0]) });
    say(ok and !prefix_kept and cached_rows[0] == 0, "(i) cancelled after {d}+ pieces: {d} abandoned, a prefix of it kept: {}, the decoding reply and the prompt again (cached {d}) {s} serial", .{ 3, m.t.stats.abandoned, prefix_kept, cached_rows[0], if (ok) "equal" else "differ from" });
    try testing.expect(ok and !prefix_kept and cached_rows[0] == 0);
}

test "dsv41 metal serve: (j) a decode share gives a decoding reply rounds between a prompt's pieces; replies == serial" {
    const long = try ids(1100, 34);
    defer gpa.free(long);
    const p0 = try ids(120, 35);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 300, .drafts = false }, .{ .prompt = long, .max = 8 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    var bad: usize = 0;
    // a share of 0 is one round a step (no debt); 0.75 owes replies three times each step's time (and caps its rows)
    for ([_]f64{ 0, 0.75 }) |share| {
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 256, .steps = true, .decode_share = share });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = p0, .max_tokens = 300, .drafts = false }, .{ .prompt = long, .max_tokens = 8 } };
        try e.submit(800, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        try awaitStat(&m.t.stats.windows, 1);
        try e.submit(801, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const between = m.t.stats.most_between;
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const ok = equal and m.t.stats.pieces >= 2 and if (share > 0) between >= 3 else between <= 1;
        say(ok, "(j) decode share {d:.2}: at most {d} rounds between two of the prompt's {d} pieces; replies {s} serial", .{ share, between, m.t.stats.pieces, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "dsv41 metal serve: (k) fcfs steps capped at 128 and 100 rows: replies == serial, each step on the chunk staged for it" {
    const long = try ids(1100, 36);
    defer gpa.free(long);
    const p0 = try ids(120, 37);
    defer gpa.free(p0);
    const asks = [_]Ask{ .{ .prompt = p0, .max = 300, .drafts = false }, .{ .prompt = long, .max = 8 } };
    var want: [2][]u32 = undefined;
    try reference(&asks, &want);
    defer freeAll(&want);
    _ = setenv("TENSORFOLD_PREFILL_POLICY", "fcfs", 1);
    defer _ = unsetenv("TENSORFOLD_PREFILL_POLICY");
    defer _ = unsetenv("TENSORFOLD_PREFILL_TOKENS");
    var bad: usize = 0;
    // a 256-row chunk: steps of half a chunk and of rows that divide none, while a reply decodes
    for ([_][:0]const u8{ "128", "100" }) |tokens| {
        _ = setenv("TENSORFOLD_PREFILL_TOKENS", tokens, 1);
        const m = try fs.open(gpa, io, seed, .{ .slots = 2, .chunk = 256, .steps = try host.serve_lanes.stepsFromEnv() });
        const s = try Served.open(gpa, io, m.model(), ".", null);
        defer Served.close(s);
        const e = s.engine();
        var boxes: [2]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        const reqs = [2]api.Request{ .{ .prompt = p0, .max_tokens = 300, .drafts = false }, .{ .prompt = long, .max_tokens = 8 } };
        try e.submit(900, &reqs[0], .{ .ctx = &boxes[0], .event = Box.event });
        try awaitStat(&m.t.stats.windows, 1);
        try e.submit(901, &reqs[1], .{ .ctx = &boxes[1], .event = Box.event });
        for (&boxes) |*b| try b.wait();
        const equal = std.mem.eql(u32, want[0], boxes[0].out.items) and std.mem.eql(u32, want[1], boxes[1].out.items);
        const hits = m.t.pr.stage_hits;
        const ok = equal and m.t.stats.pieces >= 3 and hits + 1 >= m.t.stats.pieces;
        say(ok, "(k) fcfs, {s} rows a step: {d} pieces, {d} calls on a staged chunk; replies {s} serial", .{ tokens, m.t.stats.pieces, hits, if (equal) "equal" else "differ from" });
        bad += @intFromBool(!ok);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
