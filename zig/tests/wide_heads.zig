//! Head trees against the head's chain: a chain shape gives chain() bits; every lane equals the chain run on its path.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const probe = @import("wide_probe.zig");

const nm = tf.nemotron;
const fwd = nm.forward;
const shape = tf.lanes.shape;
const Probe = probe.Probe;

/// The head's tree for `sh` from scratch.x row 0 and `t`, its lanes' tokens read back (the cache keeps the root's row).
fn treeDraft(p: *Probe, sh: shape.Shape, t: u32, out: []u32) !void {
    p.ids.slice(u32, 1)[0] = t;
    const Job = struct {
        sh: shape.Shape,
        pub fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
            nm.head_tree.draft(&q.b.head.?, e, q.c, j.sh, q.b.scratch.x, 0, q.ids, 0, q.out, 4);
        }
    };
    _ = try p.run(&.{}, Job{ .sh = sh });
    @memcpy(out[0..sh.parents.len], p.out.slice(u32, 1 + sh.parents.len)[1..]);
}

/// One head step a command buffer along `path` (after the root from row 0 and `t`): the last step's best 4 tokens.
fn pathTop(p: *Probe, t: u32, path: []const u32, top: *[4]u32) !void {
    const kept = p.c.mtp_len;
    defer p.c.mtp_len = kept;
    for (0..path.len + 1) |k| {
        p.ids.slice(u32, 1)[0] = if (k == 0) t else path[k - 1];
        const Step = struct {
            k: usize,
            pub fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
                const h = &q.b.head.?;
                if (j.k == 0) h.stepAt(e, q.c, q.b.scratch.x, 0, q.ids, 0, .greedy, h.scratch.ids, 0, null) else h.stepAt(e, q.c, h.hid, 0, q.ids, 0, .greedy, h.scratch.ids, 0, null);
            }
        };
        _ = try p.run(&.{}, Step{ .k = k });
        p.c.mtp_len += 1;
    }
    var buf: [8]u32 = undefined;
    p.topDrafts(4, &buf);
    top.* = buf[0..4].*;
}

/// At sampled positions: chain shapes against chain(), and every lane of `lanes`-lane trees against its path's chain run.
pub fn check(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, name: []const u8, samples: usize, every: usize, lanes: usize) !void {
    const prompts = try probe.readPrompts(arena, path);
    const prompt = prompts.map.get(name) orelse return error.NoSuchPrompt;
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + samples * every + 128, .drafts = true, .streams = 2, .batch_rows = 32 });
    defer b.deinit();
    var p = try Probe.init(b);
    defer p.deinit();
    try p.prefill(prompt);
    var t = try p.first();
    try p.absorb(p.c.rows - 1, t);
    var chain_ok: usize = 0;
    var chain_all: usize = 0;
    var lanes_ok: usize = 0;
    var lanes_all: usize = 0;
    // odds that grow siblings at every depth, so the trees branch wide and deep
    const odds = shape.Odds.init(.{ 0.7, 0.2, 0.1, 0.05 }, 4);
    for (0..samples) |_| {
        for (0..every) |i| {
            if (i > 0) try p.absorb(0, t);
            t = try p.step(t);
        }
        const m0 = p.c.mtp_len;
        // a chain shape against the head's chain
        const depth = 8;
        var cp: [depth]i32 = undefined;
        var cr: [depth]u8 = @splat(0);
        var cd: [depth]u8 = undefined;
        for (0..depth) |i| {
            cp[i] = @as(i32, @intCast(i)) - 1;
            cd[i] = @intCast(i);
        }
        var got: [64]u32 = undefined;
        try treeDraft(&p, .{ .parents = &cp, .ranks = &cr, .depths = &cd, .expected = 0 }, t, &got);
        p.c.mtp_len = m0;
        const want = try p.drafts(0, t, depth);
        for (want, got[0..depth]) |w, g| chain_ok += @intFromBool(w == g);
        chain_all += depth;
        p.c.mtp_len = m0;
        // a branching tree: each lane's token is rank r of the chain run on its path
        const sh = try shape.best(gpa, odds, lanes, 16);
        defer sh.deinit(gpa);
        try treeDraft(&p, sh, t, &got);
        p.c.mtp_len = m0;
        for (sh.parents, sh.ranks, 0..) |_, r, i| {
            var path_tokens: [16]u32 = undefined;
            var n: usize = 0;
            var at = sh.parents[i];
            while (at >= 0) : (at = sh.parents[@intCast(at)]) n += 1;
            at = sh.parents[i];
            var k = n;
            while (at >= 0) : (at = sh.parents[@intCast(at)]) {
                k -= 1;
                path_tokens[k] = got[@intCast(at)];
            }
            var top: [4]u32 = undefined;
            try pathTop(&p, t, path_tokens[0..n], &top);
            lanes_ok += @intFromBool(top[r] == got[i]);
        }
        lanes_all += sh.parents.len;
        p.c.mtp_len = m0 + 1;
    }
    std.debug.print("{s}: chain shapes {d}/{d} drafts equal to chain(); {d}-lane trees {d}/{d} lanes equal to the chain run on their path\n", .{ name, chain_ok, chain_all, lanes, lanes_ok, lanes_all });
}
