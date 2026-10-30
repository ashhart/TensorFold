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
}

/// A reply's tokens and finish, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,
    reason: ?api.Reason = null,
    drafted: u64 = 0,
    cached: u32 = 0,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
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
