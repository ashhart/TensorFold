//! GLM-5.3-Flash drafted replies against plain ones on the checkpoint, greedy and sampled, keyed and argmax MTP drafts.
const std = @import("std");
const sys = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const ge = glm.engine;
const fwd = glm.forward;
const Slots = glm.slots.Slots;

const usage =
    \\tf-glm-drafts MODEL_DIR: replies drafted at depth GLM_DEPTH (3) against plain ones, greedy and sampled (seed 7,
    \\temperature 1, top_p 0.95), through generate and through the served slots (queued plain rounds; windows, keeps
    \\and drafts as the lane core runs them), sampled MTP drafts keyed and by argmax. Every drafted reply must equal its
    \\plain one; each case's drafts kept are printed. GLM_MAX (64) tokens a reply; GLM_LAYERS (read at load).
;

const sampling: tf.lanes.Sampling = .{ .seed = 7, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
const prompts = [_]Prompt{ .{ .len = 12, .period = 0 }, .{ .len = 240, .period = 20 }, .{ .len = 600, .period = 0 } };

/// Token ids: a repeated pattern of `period` (the drafts mostly kept) or a spread with none (mostly missed).
const Prompt = struct {
    len: u32,
    period: u32,

    fn ids(p: Prompt, out: []u32) []u32 {
        for (out[0..p.len], 0..) |*t, i| {
            const k: u32 = if (p.period > 0) @intCast(i % p.period) else @intCast(i);
            t.* = 1000 + (k * 7919) % 60000;
        }
        return out[0..p.len];
    }
};

/// A reply and its drafts: how many were verified and how many kept.
const Reply = struct {
    toks: []u32,
    drafted: u64 = 0,
    kept: u64 = 0,
};

const Collect = struct {
    gpa: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,

    fn prefilled(_: *anyopaque) void {}
    fn tokens(ctx: *anyopaque, t: []const u32) bool {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        c.toks.appendSlice(c.gpa, t) catch {};
        return false;
    }
    fn cancelled(_: *anyopaque) bool {
        return false;
    }
};

fn generate(gpa: std.mem.Allocator, e: *ge.Engine, ids: []const u32, max: usize, depth: usize, s: ?tf.lanes.Sampling) !Reply {
    var col: Collect = .{ .gpa = gpa };
    defer col.toks.deinit(gpa);
    e.sampling = s;
    defer e.sampling = null;
    const r = try e.generate(ids, max, &.{}, depth, .{ .ctx = &col, .prefilled = Collect.prefilled, .tokens = Collect.tokens, .cancelled = Collect.cancelled });
    return .{ .toks = try gpa.dupe(u32, col.toks.items), .drafted = r.drafted, .kept = r.accepted };
}

/// Slot 0's prompt, chunk by chunk as the server splits it, its last row drawn by `s`'s rule; its first token.
fn fill(sl: *Slots, ids: []const u32, drafts: bool, s: ?tf.lanes.Sampling) !u32 {
    try sl.begin(0, ids, drafts);
    const n: u32 = @intCast(ids.len);
    const draws: ?fwd.Draws = if (s) |p| fwd.rulesFor(p, n, sl.e.c.vocab, 1) else null; // keyed at the prompt's length
    var at: u32 = 0;
    while (at < n) {
        const rows = sl.chunkRows(at, n);
        try sl.chunk(0, at, rows, if (draws != null and at + rows == n) &draws.? else null);
        at += rows;
    }
    return ge.Engine.u32s(sl.e.sc.picks, 1)[0];
}

/// Plain rounds queued as the lane core queues them: each reads the one before's pick on the GPU, read a round later.
fn servedPlain(gpa: std.mem.Allocator, sl: *Slots, ids: []const u32, max: usize, s: ?tf.lanes.Sampling) !Reply {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    const first = try fill(sl, ids, false, s);
    defer sl.release(0);
    try out.append(gpa, first);
    var h = sl.ringPut(first);
    var prev: ?u32 = null;
    while (out.items.len + @intFromBool(prev != null) < max) {
        const next = try sl.queueRound(0, h, s);
        if (prev) |p| try out.append(gpa, try sl.ringGet(p));
        prev = next;
        h = next;
    }
    if (prev) |p| try out.append(gpa, try sl.ringGet(p));
    return .{ .toks = try out.toOwnedSlice(gpa) };
}

/// Drafted windows as the lane core runs them: verify, keep the prefix the picks agree with, draft from the kept rows.
fn servedDrafted(gpa: std.mem.Allocator, sl: *Slots, ids: []const u32, max: usize, depth: u32, s: ?tf.lanes.Sampling) !Reply {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    const first = try fill(sl, ids, true, s);
    defer sl.release(0);
    try out.append(gpa, first);
    try sl.draftAll(&.{.{ .slot = 0, .prompt = true, .follow = &.{first}, .depth = depth, .sampling = s }});
    var r: Reply = .{ .toks = &.{} };
    var pending = first;
    while (out.items.len < max) {
        const at = sl.length(0);
        const draws: ?fwd.Draws = if (s) |p| fwd.rulesFor(p, at + 1, sl.e.c.vocab, depth + 1) else null;
        try sl.window(&.{.{ .slot = 0, .pending = pending, .held = depth, .tokens = &.{} }}, if (draws) |*d| d else null);
        var picks: [16]u32 = undefined;
        @memcpy(picks[0 .. depth + 1], ge.Engine.u32s(sl.e.sc.picks, depth + 1));
        const drafts = ge.Engine.u32s(sl.slots[0].held, depth);
        var j: u32 = 0;
        while (j < depth and picks[j] == drafts[j]) j += 1;
        r.drafted += depth;
        r.kept += j;
        try out.appendSlice(gpa, picks[0 .. j + 1]);
        if (j < depth) try sl.keep(0, j + 1); // the lane core keeps only on a miss; all kept stays as the window left it
        pending = picks[j];
        try sl.draftAll(&.{.{ .slot = 0, .prompt = false, .follow = picks[0 .. j + 1], .depth = depth, .sampling = s }});
    }
    out.shrinkRetainingCapacity(max);
    r.toks = try out.toOwnedSlice(gpa);
    return r;
}

fn firstDiff(a: []const u32, b: []const u32) ?usize {
    for (0..@min(a.len, b.len)) |i| if (a[i] != b[i]) return i;
    return if (a.len == b.len) null else @min(a.len, b.len);
}

/// `got` against `want`: PASS when equal; its drafts kept.
fn check(name: []const u8, len: u32, want: Reply, got: Reply) bool {
    const d = firstDiff(want.toks, got.toks);
    std.debug.print("{s} prompt of {d}, {s}: {d} tokens, {d} of {d} drafts kept", .{ if (d == null) "PASS" else "FAIL", len, name, got.toks.len, got.kept, got.drafted });
    if (d) |at| std.debug.print(", differs from plain at {d}", .{at});
    std.debug.print("\n", .{});
    return d == null;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: {s}\n", .{usage});
        std.process.exit(2);
    }
    const max: usize = if (sys.getenv("GLM_MAX")) |v| try std.fmt.parseInt(usize, v, 10) else 64;
    const depth: u32 = if (sys.getenv("GLM_DEPTH")) |v| try std.fmt.parseInt(u32, v, 10) else 3;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const e = try ge.Engine.load(gpa, init.io, args[1], 4096);
    defer e.deinit();
    if (!e.hasMtp()) return error.NoMtpHead;
    var bad: usize = 0;
    var sums: [4][2]u64 = @splat(@splat(0)); // sampled drafts kept and verified: generate and served, argmax and keyed
    var buf: [600]u32 = undefined;
    for (prompts) |p| {
        const ids = p.ids(&buf);
        for ([2]?tf.lanes.Sampling{ null, sampling }) |s| {
            const word = if (s == null) "greedy" else "sampled";
            const plain = try generate(gpa, e, ids, max, 0, s);
            for ([2]bool{ false, true }) |keyed| {
                if (s == null and !keyed) continue;
                e.keyed_drafts = keyed;
                const r = try generate(gpa, e, ids, max, depth, s);
                var name: [64]u8 = undefined;
                bad += @intFromBool(!check(try std.fmt.bufPrint(&name, "generate {s}, {s} drafts", .{ word, if (keyed) "keyed" else "argmax" }), p.len, plain, r));
                if (s != null) sums[@intFromBool(keyed)] = .{ sums[@intFromBool(keyed)][0] + r.kept, sums[@intFromBool(keyed)][1] + r.drafted };
            }
        }
    }
    e.keyed_drafts = true;
    var sl = try Slots.init(gpa, e, 1);
    defer sl.deinit(gpa);
    for (prompts) |p| {
        const ids = p.ids(&buf);
        for ([2]?tf.lanes.Sampling{ null, sampling }) |s| {
            const word = if (s == null) "greedy" else "sampled";
            const plain = try servedPlain(gpa, &sl, ids, max, s);
            for ([2]bool{ false, true }) |keyed| {
                if (s == null and !keyed) continue;
                e.keyed_drafts = keyed;
                const r = try servedDrafted(gpa, &sl, ids, max, depth, s);
                var name: [64]u8 = undefined;
                bad += @intFromBool(!check(try std.fmt.bufPrint(&name, "served {s}, {s} drafts", .{ word, if (keyed) "keyed" else "argmax" }), p.len, plain, r));
                if (s != null) sums[2 + @as(usize, @intFromBool(keyed))] = .{ sums[2 + @as(usize, @intFromBool(keyed))][0] + r.kept, sums[2 + @as(usize, @intFromBool(keyed))][1] + r.drafted };
            }
        }
    }
    e.keyed_drafts = true;
    for ([_][]const u8{ "generate", "served" }, 0..) |where, i| std.debug.print("sampled drafts kept, {s}: argmax {d} of {d}, keyed {d} of {d}\n", .{ where, sums[2 * i][0], sums[2 * i][1], sums[2 * i + 1][0], sums[2 * i + 1][1] });
    std.debug.print("{s}: GLM-5.3-Flash drafted == plain at depth {d}, {d} cases differ\n", .{ if (bad == 0) "PASS" else "FAIL", depth, bad });
    if (bad != 0) std.process.exit(1);
}
