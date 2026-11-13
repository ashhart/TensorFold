//! The tiny model's slots over one shared pool (TF_DSV41_POOL_TOKENS) on Metal: replies equal the unpooled serial run.
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
const new = 16;

fn ids(n: usize, s: u64) ![]u32 {
    return ff.ids(gpa, &fs.config(), n, s);
}

fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal pool: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

/// The unpooled reference: each prompt's serial greedy decode on a model of one slot.
fn reference(prompts: []const []const u32, out: [][]u32) !void {
    const m = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    defer fs.close(m);
    for (prompts, out) |p, *o| o.* = try m.t.serial(0, p, new, null);
}

/// One slot's request driven a row at a time through the target, so several interleave on purpose.
const Run = struct {
    slot: u32,
    prompt: []const u32,
    out: std.ArrayList(u32) = .empty,
    tok: u32 = 0,

    fn begin(r: *Run, t: *pkg.target.Target) !void {
        const tg = t.target();
        tg.admit(r.slot, new);
        r.tok = try tg.prefill(r.slot, r.prompt, null, r.prompt.len);
    }

    /// The next token: the pending one goes in at its position and its successor is drawn.
    fn step(r: *Run, t: *pkg.target.Target) !void {
        try r.out.append(gpa, r.tok);
        if (r.out.items.len == new) return;
        const at: u64 = r.prompt.len + r.out.items.len - 1;
        var ch: [1]u32 = undefined;
        var cs = [_][]u32{&ch};
        try t.target().window(&.{.{ .slot = r.slot, .start = at, .tokens = &.{r.tok}, .parents = null, .draws = &.{at + 1}, .sampling = null }}, &cs);
        r.tok = ch[0];
    }

    fn finish(r: *Run, t: *pkg.target.Target) !void {
        while (r.out.items.len < new) try r.step(t);
        t.target().release(r.slot);
    }
};

test "dsv41 metal pool: runs slide down under live requests, a full pool says LaneMemory, replies are bit-equal" {
    const n = [_]usize{ 300, 300, 300, 300, 600 };
    var prompts: [5][]u32 = undefined;
    for (&prompts, n, 0..) |*p, len, i| p.* = try ids(len, 40 + i);
    defer for (prompts) |p| gpa.free(p);
    var want: [5][]u32 = undefined;
    try reference(&prompts, &want);
    defer for (want) |w| gpa.free(w);
    // 2000 positions: 8 blocks; a 300-id request takes 2, the 600-id one 3
    const m = try fs.open(gpa, io, seed, .{ .slots = 4, .drafts = false, .pool = 2000 });
    defer fs.close(m);
    const t = m.t;
    var runs = [_]Run{ .{ .slot = 0, .prompt = prompts[0] }, .{ .slot = 1, .prompt = prompts[1] }, .{ .slot = 2, .prompt = prompts[2] }, .{ .slot = 3, .prompt = prompts[3] }, .{ .slot = 0, .prompt = prompts[4] } };
    defer for (&runs) |*r| r.out.deinit(gpa);
    for (runs[0..4]) |*r| try r.begin(t);
    for (0..5) |_| for (runs[0..4]) |*r| try r.step(t);
    try runs[0].finish(t);
    try runs[2].finish(t);
    // blocks [0, 2) and [4, 6) are free, no run of 3: slots 1 and 3 slide down and the new run lands on top
    try runs[4].begin(t);
    const moved = t.pool.?.runs[1].?.start == 0 and t.pool.?.runs[3].?.start == 2 and t.pool.?.runs[0].?.start == 4;
    // one block is left: a further request waits
    const extra = Run{ .slot = 2, .prompt = prompts[2] };
    var waits = false;
    t.target().admit(extra.slot, new);
    _ = t.target().prefill(extra.slot, extra.prompt, null, extra.prompt.len) catch |e| {
        waits = e == error.LaneMemory;
    };
    for ([_]usize{ 1, 3, 4 }) |i| try runs[i].finish(t);
    var bad: usize = 0;
    for (runs, want) |r, w| bad += @intFromBool(!std.mem.eql(u32, r.out.items, w));
    say(bad == 0 and moved and waits, "5 requests over 8 blocks: {d} replies differ from unpooled serial; runs slid down: {}; a full pool waits: {}", .{ bad, moved, waits });
    try testing.expect(moved and waits);
    try testing.expectEqual(@as(usize, 0), bad);
    try testing.expectEqual(@as(u32, 8), t.pool.?.free());
}

/// A reply's tokens and finish, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
            .finished => b.done = true,
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
};

test "dsv41 metal pool: through LaneHost, four requests that cannot all fit wait their turn and equal serial" {
    var prompts: [4][]u32 = undefined;
    for (&prompts, 0..) |*p, i| p.* = try ids(620, 60 + i);
    defer for (prompts) |p| gpa.free(p);
    var want: [4][]u32 = undefined;
    try reference(&prompts, &want);
    defer for (want) |w| gpa.free(w);
    // each request takes 3 blocks of the 8: two run at once, the other two wait for positions
    const m = try fs.open(gpa, io, seed, .{ .slots = 4, .pool = 2000 });
    const s = try Served.open(gpa, io, m.model(), ".", null);
    defer Served.close(s);
    const e = s.engine();
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.out.deinit(gpa);
    var reqs: [4]api.Request = undefined;
    for (&reqs, prompts, 0..) |*r, p, i| {
        r.* = .{ .prompt = p, .max_tokens = new };
        try e.submit(@intCast(900 + i), r, .{ .ctx = &boxes[i], .event = Box.event });
    }
    var bad: usize = 0;
    for (&boxes, want) |*b, w| {
        try b.wait();
        bad += @intFromBool(!std.mem.eql(u32, b.out.items, w));
    }
    say(bad == 0, "4 requests of 3 blocks over 8 through LaneHost: {d} replies differ from unpooled serial", .{bad});
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal pool: a slot past the sixteenth sizes its run by its max new tokens, not the whole window" {
    const p = try ids(300, 80);
    defer gpa.free(p);
    const m = try fs.open(gpa, io, seed, .{ .slots = 20, .drafts = false, .pool = 2000 });
    defer fs.close(m);
    const t = m.t;
    var r = Run{ .slot = 19, .prompt = p };
    defer r.out.deinit(gpa);
    try r.begin(t);
    const blocks = t.pool.?.runs[19].?.blocks;
    try r.finish(t);
    say(blocks == 2, "slot 19 of 20, a 300-id prompt with 16 new tokens: a run of {d} blocks (2 expected, 6 the whole window)", .{blocks});
    try testing.expectEqual(@as(u32, 2), blocks);
}

test "dsv41 metal pool: with the prompt cache on, moved runs and a kept state restored into a new run equal serial" {
    const n = [_]usize{ 300, 300, 300, 300, 600 };
    var prompts: [5][]u32 = undefined;
    for (&prompts, n, 0..) |*p, len, i| p.* = try ids(len, 40 + i);
    defer for (prompts) |p| gpa.free(p);
    var want: [5][]u32 = undefined;
    try reference(&prompts, &want);
    defer for (want) |w| gpa.free(w);
    const m = try fs.open(gpa, io, seed, .{ .slots = 4, .drafts = false, .pool = 2000, .cache_bytes = 1 << 30 });
    defer fs.close(m);
    const t = m.t;
    var runs = [_]Run{ .{ .slot = 0, .prompt = prompts[0] }, .{ .slot = 1, .prompt = prompts[1] }, .{ .slot = 2, .prompt = prompts[2] }, .{ .slot = 3, .prompt = prompts[3] }, .{ .slot = 0, .prompt = prompts[4] }, .{ .slot = 2, .prompt = prompts[0] } };
    defer for (&runs) |*r| r.out.deinit(gpa);
    for (runs[0..4]) |*r| try r.begin(t);
    for (0..5) |_| for (runs[0..4]) |*r| try r.step(t);
    try runs[0].finish(t);
    try runs[2].finish(t);
    const moves0 = t.pool.?.runs[1].?.start;
    try runs[4].begin(t); // seats 1 and 3 slide down under their live requests
    const moved = moves0 != t.pool.?.runs[1].?.start;
    for ([_]usize{ 1, 3 }) |i| try runs[i].finish(t);
    const hits0 = t.stats.hits;
    try runs[5].begin(t); // the first prompt again: its kept state restored into a run elsewhere
    const hit = t.stats.hits > hits0;
    try runs[5].finish(t);
    try runs[4].finish(t);
    var bad: usize = 0;
    for (runs, [_][]u32{ want[0], want[1], want[2], want[3], want[4], want[0] }) |r, w| bad += @intFromBool(!std.mem.eql(u32, r.out.items, w));
    say(bad == 0 and moved and hit, "cache on: {d} of 6 replies differ from serial; runs slid down: {}; a kept state restored: {}", .{ bad, moved, hit });
    try testing.expect(moved and hit);
    try testing.expectEqual(@as(usize, 0), bad);
}
