//! Rank 0's engine operations, each sent to rank 1 before it runs, and rank 1's loop running them in the same order.
const std = @import("std");
const eng = @import("engine.zig");
const snap = @import("snap.zig");
const Engine = eng.Engine;
const R = @import("gpu.zig").max_rows;

pub const Kind = enum(u32) { begin = 1, chunk, window, keep, draft, release, save, restore, drop };

fn send(e: *Engine, kind: Kind) !void {
    try e.pair.send(@backingInt(kind), e.words.items);
}

fn put(e: *Engine, w: []const u32) !void {
    try e.words.appendSlice(e.gpa, w);
}

pub fn begin(e: *Engine, i: u32, prompt: []const u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{i});
    try put(e, prompt);
    try send(e, .begin);
    try e.begin(i, prompt);
}

pub fn chunk(e: *Engine, i: u32, at: u32, n: u32, every: bool) !?u32 {
    e.words.clearRetainingCapacity();
    try put(e, &.{ i, at, n, @intFromBool(every) });
    try send(e, .chunk);
    return e.chunk(i, at, n, every);
}

pub fn window(e: *Engine, wins: []const eng.Win, out: []u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{@intCast(wins.len)});
    for (wins) |w| {
        try put(e, &.{ w.slot, @intCast(w.tokens.len) });
        try put(e, w.tokens);
    }
    try send(e, .window);
    try e.window(wins, out);
}

pub fn keep(e: *Engine, i: u32, kept: u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ i, kept });
    try send(e, .keep);
    try e.keep(i, kept);
}

pub fn draft(e: *Engine, reqs: []const eng.Draft) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{@intCast(reqs.len)});
    for (reqs) |q| {
        try put(e, &.{ q.slot, @intFromBool(q.prompt), q.depth, @intCast(q.follow.len) });
        try put(e, q.follow);
    }
    try send(e, .draft);
    try e.draft(reqs);
}

pub fn release(e: *Engine, i: u32) void {
    e.words.clearRetainingCapacity();
    put(e, &.{i}) catch {};
    send(e, .release) catch |err| std.log.err("glm_dsa: the peer missed a release: {s}", .{@errorName(err)});
    e.release(i);
}

pub fn save(e: *Engine, id: u32, i: u32, at: u32) !*snap.Snap {
    e.words.clearRetainingCapacity();
    try put(e, &.{ id, i, at });
    try send(e, .save);
    return snap.save(e, id, i, at);
}

pub fn restore(e: *Engine, i: u32, id: u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ i, id });
    try send(e, .restore);
    try snap.restore(e, i, id);
}

pub fn drop(e: *Engine, id: u32) void {
    e.words.clearRetainingCapacity();
    put(e, &.{id}) catch {};
    send(e, .drop) catch |err| std.log.err("glm_dsa: the peer kept a dropped state: {s}", .{@errorName(err)});
    snap.drop(e, id);
}

/// A command's words, read in order; a short command is out of step with rank 0.
const Reader = struct {
    w: []const u32,
    at: usize = 0,

    fn one(r: *Reader) !u32 {
        if (r.at >= r.w.len) return error.CommandOutOfStep;
        r.at += 1;
        return r.w[r.at - 1];
    }
    fn many(r: *Reader, n: usize) ![]const u32 {
        if (r.at + n > r.w.len) return error.CommandOutOfStep;
        r.at += n;
        return r.w[r.at - n .. r.at];
    }
};

/// Rank 1: run rank 0's commands until it closes or this Mac stops following; an error rank 0 met too is logged.
pub fn follow(e: *Engine) !void {
    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(e.gpa);
    var wins: [R]eng.Win = undefined;
    var reqs: [R]eng.Draft = undefined;
    var out: [R]u32 = undefined;
    while (try e.pair.next(e.gpa, &words)) |k| {
        var r: Reader = .{ .w = words.items };
        const kind = std.enums.fromInt(Kind, k) orelse return error.CommandOutOfStep;
        const done: anyerror!void = switch (kind) {
            .begin => e.begin(try r.one(), r.w[1..]),
            .chunk => if (e.chunk(try r.one(), try r.one(), try r.one(), try r.one() != 0)) |_| {} else |err| err,
            .window => blk: {
                const n = try r.one();
                if (n == 0 or n > R) return error.CommandOutOfStep;
                for (wins[0..n]) |*w| {
                    const slot = try r.one();
                    w.* = .{ .slot = slot, .tokens = try r.many(try r.one()) };
                }
                break :blk e.window(wins[0..n], &out);
            },
            .keep => e.keep(try r.one(), try r.one()),
            .draft => blk: {
                const n = try r.one();
                if (n == 0 or n > R) return error.CommandOutOfStep;
                for (reqs[0..n]) |*q| {
                    const slot = try r.one();
                    const prompt = try r.one() != 0;
                    const depth = try r.one();
                    q.* = .{ .slot = slot, .prompt = prompt, .depth = depth, .follow = try r.many(try r.one()) };
                }
                break :blk e.draft(reqs[0..n]);
            },
            .release => e.release(try r.one()),
            .save => if (snap.save(e, try r.one(), try r.one(), try r.one())) |_| {} else |err| err,
            .restore => snap.restore(e, try r.one(), try r.one()),
            .drop => snap.drop(e, try r.one()),
        };
        done catch |err| {
            if (err == error.PairLinkFailed or err == error.GpuFailure) return err;
            std.log.warn("glm_dsa rank 1: {s} failed as on rank 0: {s}", .{ @tagName(kind), @errorName(err) });
        };
    }
}
