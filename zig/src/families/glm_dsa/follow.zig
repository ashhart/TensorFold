//! Rank 0's engine operations, each sent to rank 1 before it runs, and rank 1's loop running them in the same order.
const std = @import("std");
const eng = @import("engine.zig");
const snap = @import("snap.zig");
const Engine = eng.Engine;
const R = @import("gpu.zig").max_rows;

pub const Kind = enum(u32) { begin = 1, chunk, window, keep, draft, release, save, restore, drop, window_draft, extend_draft, window_tree, keep_path, window_tree_draft };

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

pub fn windowDraft(e: *Engine, slot: u32, tokens: []const u32, pred_depth: u32, out: []u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ slot, pred_depth, @intCast(tokens.len) });
    try put(e, tokens);
    try send(e, .window_draft);
    try e.windowDraft(slot, tokens, pred_depth, out);
}

pub fn extendDraft(e: *Engine, slot: u32, want: u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ slot, want });
    try send(e, .extend_draft);
    try e.extendDraft(slot, want);
}

pub fn windowTree(e: *Engine, slot: u32, tokens: []const u32, parents: []const i32, out: []u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ slot, @intCast(tokens.len) });
    try put(e, tokens);
    for (parents) |p| try put(e, &.{@bitCast(p)});
    try send(e, .window_tree);
    try eng.tr.windowTree(e, slot, tokens, parents, out);
}

pub fn windowTreeDraft(e: *Engine, slot: u32, tokens: []const u32, parents: []const i32, pred_depth: u32, out: []u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ slot, pred_depth, @intCast(tokens.len) });
    try put(e, tokens);
    for (parents) |p| try put(e, &.{@bitCast(p)});
    try send(e, .window_tree_draft);
    try e.windowTreeDraft(slot, tokens, parents, pred_depth, out);
}

pub fn keepPath(e: *Engine, i: u32, path: []const u32) !void {
    e.words.clearRetainingCapacity();
    try put(e, &.{ i, @intCast(path.len) });
    try put(e, path);
    try send(e, .keep_path);
    try eng.tr.keepPath(e, i, path);
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
        const pn: u32 = if (q.path.len == q.follow.len and eng.tr.needsCompact(q.path)) @intCast(q.path.len) else 0;
        try put(e, &.{ q.slot, @intFromBool(q.prompt), q.depth, @intCast(q.follow.len), pn });
        try put(e, q.follow);
        if (pn > 0) try put(e, q.path);
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
                    const fn_len = try r.one();
                    const pn = try r.one();
                    const fol = try r.many(fn_len);
                    const path = if (pn > 0) try r.many(pn) else &.{};
                    q.* = .{ .slot = slot, .prompt = prompt, .depth = depth, .follow = fol, .path = path };
                }
                break :blk e.draft(reqs[0..n]);
            },
            .release => e.release(try r.one()),
            .save => if (snap.save(e, try r.one(), try r.one(), try r.one())) |_| {} else |err| err,
            .restore => snap.restore(e, try r.one(), try r.one()),
            .drop => snap.drop(e, try r.one()),
            .window_draft => blk: {
                const slot = try r.one();
                const pred_depth = try r.one();
                const toks = try r.many(try r.one());
                break :blk e.windowDraft(slot, toks, pred_depth, &out);
            },
            .extend_draft => e.extendDraft(try r.one(), try r.one()),
            .window_tree => blk: {
                const slot = try r.one();
                const nrows = try r.one();
                if (nrows == 0 or nrows > 8) return error.CommandOutOfStep;
                const toks = try r.many(nrows);
                const raw_p = try r.many(nrows);
                var pars: [8]i32 = undefined;
                for (raw_p, 0..) |u, j| pars[j] = @bitCast(u);
                break :blk eng.tr.windowTree(e, slot, toks, pars[0..nrows], &out);
            },
            .keep_path => blk: {
                const slot = try r.one();
                const plen = try r.one();
                break :blk eng.tr.keepPath(e, slot, try r.many(plen));
            },
            .window_tree_draft => blk: {
                const slot = try r.one();
                const pred_depth = try r.one();
                const nrows = try r.one();
                if (nrows == 0 or nrows > 8) return error.CommandOutOfStep;
                const toks = try r.many(nrows);
                const raw_p = try r.many(nrows);
                var pars: [8]i32 = undefined;
                for (raw_p, 0..) |u, j| pars[j] = @bitCast(u);
                break :blk e.windowTreeDraft(slot, toks, pars[0..nrows], pred_depth, &out);
            },
        };
        done catch |err| {
            if (err == error.PairLinkFailed or err == error.GpuFailure) return err;
            std.log.warn("glm_dsa rank {d}: {s} failed as on rank 0: {s}", .{ e.rank(), @tagName(kind), @errorName(err) });
        };
    }
}
