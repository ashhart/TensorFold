//! Sibling-heavy draft trees' window cost: each node's expert picks from its own path, then the expert kernels on the tree's schedule.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const probe = @import("wide_probe.zig");

const nm = tf.nemotron;
const fwd = nm.forward;
const Probe = probe.Probe;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const K = 6; // experts a row
const layers_max = 32;
const nodes_max = 16;

/// A caterpillar tree: siblings (ranks) at each chain depth, the chain's top-1 path continuing under rank 1.
pub const Shape = struct { name: []const u8, ranks: []const usize };

pub const shapes = [_]Shape{
    .{ .name = "tree4", .ranks = &.{3} },
    .{ .name = "tree8", .ranks = &.{ 3, 2, 2 } },
    .{ .name = "tree16", .ranks = &.{ 4, 4, 4, 3 } },
};

/// Expert picks of each MoE layer for a window's rows: [layer][row][K].
const Picks = struct { rows: usize = 0, e: [layers_max][nodes_max][K]u32 = undefined };

/// Each node's picks from a window over its path (the pending token, the chain to its parent, then the node).
fn nodePicks(p: *Probe, log: mtl.Buffer, moes: usize, path: []const u32, out: *Picks) !void {
    const b = p.b;
    b.scratch.route_log = log;
    b.scratch.route_at = 0;
    _ = try p.window(path, &.{}, null);
    b.scratch.route_log = null;
    const rows = path.len;
    const v = log.slice(u32, moes * rows * K);
    const r = out.rows;
    for (0..moes) |l| @memcpy(&out.e[l][r], v[l * rows * K + (rows - 1) * K ..][0..K]);
    out.rows += 1;
}

/// route_group's tables for one layer's picks: unique experts ascending, each with its pairs in order.
fn tables(picks: *const Picks, l: usize, uids: []u32, start: []i32, count: []i32, members: []i32) usize {
    var u: usize = 0;
    var m: usize = 0;
    for (0..128) |ex| {
        var n: i32 = 0;
        const at: i32 = @intCast(m);
        for (0..picks.rows) |r| for (0..K) |k| {
            if (picks.e[l][r][k] == ex) {
                members[m] = @intCast(r * K + k);
                m += 1;
                n += 1;
            }
        };
        if (n > 0) {
            uids[u] = @intCast(ex);
            start[u] = at;
            count[u] = n;
            u += 1;
        }
    }
    return u;
}

/// GPU ms of every MoE layer's routed experts (up then down, layers in order) on `picks`' schedule.
fn expertsMs(p: *Probe, picks: *const Picks, members_pass: usize, reps: usize) !f64 {
    const b = p.b;
    const m = b.m;
    const c = m.config;
    const moes = c.count(.moe);
    const T = struct { u: mtl.Buffer, s: mtl.Buffer, n: mtl.Buffer, mem: mtl.Buffer, uc: mtl.Buffer };
    const t = T{
        .u = try m.device.buffer(layers_max * 128 * 4, opts),
        .s = try m.device.buffer(layers_max * 128 * 4, opts),
        .n = try m.device.buffer(layers_max * 128 * 4, opts),
        .mem = try m.device.buffer(layers_max * nodes_max * K * 4, opts),
        .uc = try m.device.buffer(layers_max * 16, opts),
    };
    defer inline for (.{ t.u, t.s, t.n, t.mem, t.uc }) |x| x.deinit();
    var groups: [layers_max]usize = undefined;
    for (0..moes) |l| {
        groups[l] = tables(picks, l, t.u.slice(u32, layers_max * 128)[l * 128 ..][0..128], t.s.slice(i32, layers_max * 128)[l * 128 ..][0..128], t.n.slice(i32, layers_max * 128)[l * 128 ..][0..128], t.mem.slice(i32, layers_max * nodes_max * K)[l * nodes_max * K ..][0 .. nodes_max * K]);
        t.uc.slice(i32, layers_max * 4)[l * 4] = @intCast(groups[l]);
    }
    const Job = struct {
        t: T,
        groups: []const usize,
        mb: usize,
        rows: usize,
        pub fn encode(j: @This(), q: *Probe, e: *fwd.Enc) !void {
            const f = q.b.forward();
            const s = f.s;
            var l: usize = 0;
            for (0..f.c.layers) |i| switch (f.w.layers[i]) {
                .moe => |mo| {
                    inline for (.{ "expert_up", "expert_down" }, 0..) |key, x| {
                        const fc = if (x == 0) mo.fc1 else mo.fc2;
                        e.pipe(if (j.mb == 2) f.k.get(if (x == 0) "tf_xup_rows2" else "tf_xdown_rows2") else f.k.get(key));
                        e.buf(if (x == 0) s.x else s.act, 0, 0);
                        e.buf(j.t.u, l * 128 * 4, 1);
                        e.buf(j.t.s, l * 128 * 4, 2);
                        e.buf(j.t.n, l * 128 * 4, 3);
                        e.buf(j.t.mem, l * nodes_max * K * 4, 4);
                        e.buf(j.t.uc, l * 16, 5);
                        e.buf(fc[0].buffer, fc[0].offset, 6);
                        e.buf(fc[1].buffer, fc[1].offset, 7);
                        e.buf(fc[2].buffer, fc[2].offset, 8);
                        e.buf(if (x == 0) s.act else s.ey, 0, 9);
                        const n = if (x == 0) f.c.expert_width else f.c.hidden;
                        e.run(.{ 64, n / 8, j.groups[l] }, .{ 64, 1, 1 });
                    }
                    l += 1;
                },
                else => {},
            };
        }
    };
    var ms: [32]f64 = undefined;
    const job = Job{ .t = t, .groups = groups[0..moes], .mb = members_pass, .rows = picks.rows };
    _ = try p.run(&.{}, job);
    for (0..reps) |r| ms[r] = try p.run(&.{}, job);
    return probe.median(ms[0..reps]);
}

/// Expected tokens a round (1 + each node's path probability) for the tree minus the chain of `w` rows, by the head's chances.
fn landed(probs: []const [8]f64, toks: []const [8]u32, shape: Shape, w: usize) f64 {
    _ = toks;
    var chain: f64 = 0;
    var pre: f64 = 1;
    for (0..w - 1) |j| {
        pre *= probs[j][0];
        chain += pre;
    }
    var tree: f64 = 0;
    pre = 1;
    for (shape.ranks, 0..) |ranks, d| {
        for (0..ranks) |r| tree += pre * probs[d][r];
        pre *= probs[d][0];
    }
    return tree - chain;
}

fn distinct(picks: *const Picks, moes: usize) usize {
    var total: usize = 0;
    for (0..moes) |l| {
        var seen: [128]bool = @splat(false);
        for (0..picks.rows) |r| for (picks.e[l][r]) |x| {
            total += @intFromBool(!seen[x]);
            seen[x] = true;
        };
    }
    return total;
}

/// Per sampled position: the chain window and each tree's picks, their experts timed alone, the chain window timed whole.
pub fn run(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, name: []const u8, samples: usize, every: usize, reps: usize) !void {
    const prompts = try probe.readPrompts(arena, path);
    const prompt = prompts.map.get(name) orelse return error.NoSuchPrompt;
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const c = m.config;
    const moes = c.count(.moe);
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + samples * every + 64, .drafts = true, .streams = 2, .batch_rows = 16 });
    defer b.deinit();
    const log = try m.device.buffer(moes * nodes_max * K * 4, opts);
    defer log.deinit();
    var p = try Probe.init(b);
    defer p.deinit();
    try p.prefill(prompt);
    var t = try p.first();
    try p.absorb(p.c.rows - 1, t);
    const widths = [_]usize{ 4, 8, 16 };
    var sum = std.mem.zeroes([3][6]f64); // per width: window, experts alone chain / tree (the window's kernels), distinct chain / tree, head chances
    for (0..samples) |sample| {
        // the head's cache takes every row but the last, which its chain below takes as its first step
        for (0..every) |i| {
            if (i > 0) try p.absorb(0, t);
            t = try p.step(t);
        }
        var toks: [16][8]u32 = undefined;
        var probs: [16][8]f64 = undefined;
        try p.headLevels(0, t, 15, 4, toks[0..15], probs[0..15]);
        var chain: [16]u32 = undefined;
        chain[0] = t;
        for (0..15) |j| chain[1 + j] = toks[j][0];
        for (widths, shapes, 0..) |w, shape, wi| {
            // the chain of the same rows: one window, every row's picks
            var cp = Picks{};
            b.scratch.route_log = log;
            b.scratch.route_at = 0;
            var wms: [32]f64 = undefined;
            _ = try p.window(chain[0..w], &.{}, null);
            b.scratch.route_log = null;
            const v = log.slice(u32, moes * w * K);
            for (0..moes) |l| for (0..w) |r| @memcpy(&cp.e[l][r], v[l * w * K + r * K ..][0..K]);
            cp.rows = w;
            for (0..reps) |r| wms[r] = try p.window(chain[0..w], &.{}, null);
            // the tree: the pending row, then each depth's ranks, each node from its own path
            var tp = Picks{};
            try nodePicks(&p, log, moes, chain[0..1], &tp);
            for (shape.ranks, 0..) |ranks, d| for (0..ranks) |r| {
                var pathbuf: [16]u32 = undefined;
                @memcpy(pathbuf[0 .. d + 1], chain[0 .. d + 1]);
                pathbuf[d + 1] = toks[d][r];
                try nodePicks(&p, log, moes, pathbuf[0 .. d + 2], &tp);
            };
            const mb = b.members;
            const vals = [6]f64{ probe.median(wms[0..reps]), try expertsMs(&p, &cp, mb, reps), try expertsMs(&p, &tp, mb, reps), @floatFromInt(distinct(&cp, moes)), @floatFromInt(distinct(&tp, moes)), landed(probs[0..15], toks[0..15], shape, w) };
            for (&sum[wi], vals) |*a, x| a.* += x;
        }
        std.debug.print("sample {d}/{d}\n", .{ sample + 1, samples });
    }
    const n: f64 = @floatFromInt(samples);
    std.debug.print("{s}, {d} positions: rows | chain window ms | experts alone chain / tree | distinct experts chain / tree | tree window estimate | tree - chain expected tokens a round (head chances)\n", .{ name, samples });
    for (widths, 0..) |w, i| {
        const a = sum[i];
        std.debug.print("{d:>2} | {d:.2} | {d:.2} / {d:.2} | {d:.0} / {d:.0} | {d:.2} | {d:.2}\n", .{ w, a[0] / n, a[1] / n, a[2] / n, a[3] / n, a[4] / n, (a[0] - a[1] + a[2]) / n, a[5] / n });
    }
}

/// Tree windows from the head's top tokens, each row's logits against a chain window over its own path (bit for bit).
pub fn check(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, name: []const u8, samples: usize, every: usize) !void {
    const prompts = try probe.readPrompts(arena, path);
    const prompt = prompts.map.get(name) orelse return error.NoSuchPrompt;
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const vocab = m.config.vocab;
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + samples * every + 64, .drafts = true, .streams = 2, .batch_rows = 32 });
    defer b.deinit();
    var p = try Probe.init(b);
    defer p.deinit();
    b.head.?.topk = true;
    try p.prefill(prompt);
    var t = try p.first();
    try p.absorb(p.c.rows - 1, t);
    const ranks = [_]usize{ 4, 3, 3, 2, 2, 2, 1, 1 };
    var equal: usize = 0;
    var total: usize = 0;
    const tree_logits = try arena.alloc(u16, 32 * vocab);
    for (0..samples) |sample| {
        for (0..every) |i| {
            if (i > 0) try p.absorb(0, t);
            t = try p.step(t);
        }
        var toks: [16][8]u32 = undefined;
        var probs: [16][8]f64 = undefined;
        try p.headLevels(0, t, ranks.len, 4, toks[0..ranks.len], probs[0..ranks.len]);
        // rows by depth: each depth's ranks under the previous depth's top-1 row
        var ids: [32]u32 = undefined;
        var parents: [32]i32 = undefined;
        ids[0] = t;
        parents[0] = -1;
        var n: usize = 1;
        var chain_row: i32 = 0;
        for (ranks, 0..) |rk, d| {
            const first = n;
            for (0..rk) |r| {
                ids[n] = toks[d][r];
                parents[n] = chain_row;
                n += 1;
            }
            chain_row = @intCast(first);
        }
        _ = try p.treeWindow(ids[0..n], parents[0..n]);
        @memcpy(tree_logits[0 .. n * vocab], b.scratch.logits.slice(u16, n * vocab));
        var same: usize = 0;
        for (0..n) |r| {
            var chain: [16]u32 = undefined;
            var depth: usize = 0;
            var at: i32 = @intCast(r);
            while (at >= 0) : (at = parents[@intCast(at)]) depth += 1;
            at = @intCast(r);
            var j = depth;
            while (at >= 0) : (at = parents[@intCast(at)]) {
                j -= 1;
                chain[j] = ids[@intCast(at)];
            }
            _ = try p.window(chain[0..depth], &.{}, null);
            const got = b.scratch.logits.slice(u16, depth * vocab)[(depth - 1) * vocab ..][0..vocab];
            same += @intFromBool(std.mem.eql(u16, got, tree_logits[r * vocab ..][0..vocab]));
        }
        equal += same;
        total += n;
        std.debug.print("sample {d}: tree of {d} rows, {d} rows bit-identical to a chain window over their path\n", .{ sample, n, same });
    }
    std.debug.print("{s}: {d}/{d} tree rows bit-identical; head top-4 on the GPU {d}/{d} equal to the host's ranking\n", .{ name, equal, total, p.topk_checked - p.topk_bad, p.topk_checked });
}

/// Tree windows (the head's chain with its top-4 siblings by depth) against chain windows of the same rows: GPU ms.
pub fn cost(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, model: []const u8, path: []const u8, name: []const u8, samples: usize, every: usize, reps: usize) !void {
    const prompts = try probe.readPrompts(arena, path);
    const prompt = prompts.map.get(name) orelse return error.NoSuchPrompt;
    const m = try nm.Model.load(gpa, io, model, true);
    defer m.deinit();
    const b = try nm.backend.Metal.init(gpa, m, .{ .chunk = 16, .capacity = prompt.len + samples * every + 64, .drafts = true, .streams = 2, .batch_rows = 32 });
    defer b.deinit();
    var p = try Probe.init(b);
    defer p.deinit();
    try p.prefill(prompt);
    var t = try p.first();
    try p.absorb(p.c.rows - 1, t);
    const shapes4 = [_][]const usize{ &.{3}, &.{ 3, 2, 2 }, &.{ 4, 4, 4, 3 }, &.{ 4, 4, 4, 4, 4, 4, 4, 3 } };
    var sums = std.mem.zeroes([4][2]f64);
    for (0..samples) |_| {
        for (0..every) |i| {
            if (i > 0) try p.absorb(0, t);
            t = try p.step(t);
        }
        var toks: [32][8]u32 = undefined;
        var probs: [32][8]f64 = undefined;
        try p.headLevels(0, t, 31, 4, toks[0..31], probs[0..31]);
        var chain: [32]u32 = undefined;
        chain[0] = t;
        for (0..31) |j| chain[1 + j] = toks[j][0];
        for (shapes4, 0..) |ranks, si| {
            var ids: [32]u32 = undefined;
            var parents: [32]i32 = undefined;
            ids[0] = t;
            parents[0] = -1;
            var n: usize = 1;
            var row: i32 = 0;
            for (ranks, 0..) |rk, d| {
                const at = n;
                for (0..rk) |r| {
                    ids[n] = toks[d][r];
                    parents[n] = row;
                    n += 1;
                }
                row = @intCast(at);
            }
            var ms: [2][16]f64 = undefined;
            for (0..reps) |r| {
                ms[0][r] = try p.window(chain[0..n], &.{}, null);
                ms[1][r] = try p.treeWindow(ids[0..n], parents[0..n]);
            }
            sums[si][0] += probe.median(ms[0][0..reps]);
            sums[si][1] += probe.median(ms[1][0..reps]);
        }
    }
    const k: f64 = @floatFromInt(samples);
    std.debug.print("{s}, {d} positions, GPU ms medians of {d}: rows | chain window | tree window (head chain + top-4 siblings)\n", .{ name, samples, reps });
    for (shapes4, 0..) |ranks, si| {
        var n: usize = 1;
        for (ranks) |r| n += r;
        std.debug.print("{d:>2} | {d:.2} | {d:.2}\n", .{ n, sums[si][0] / k, sums[si][1] / k });
    }
}
