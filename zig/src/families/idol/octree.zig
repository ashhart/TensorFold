//! The octree decoder's host part: the level modulation, systematic resampling, compaction and the leaves' jitter.
const std = @import("std");
const dc = @import("decoder_config.zig");
const project = @import("host.zig").project;

const H = dc.hidden;
const P = dc.children;

fn half(x: f32) f32 {
    return @as(f16, @floatCast(x));
}

fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// torch.linspace(start, end, n) in fp32 as its CPU kernel computes it: fma from the nearer end.
pub fn linspace(out: []f32, start: f32, end: f32) void {
    const n = out.len;
    const step = (end - start) / @as(f32, @floatFromInt(n - 1));
    for (out, 0..) |*o, i| {
        o.* = if (i < n / 2) @mulAdd(f32, step, @floatFromInt(i), start) else @mulAdd(f32, -step, @floatFromInt(n - 1 - i), end);
    }
}

/// Where torch's CPU pow (built at init, not saved: the model's constants) rounds 2^linspace the other way, in ulps.
const torch_pow = [_]struct { usize, i32 }{ .{ 43, -1 }, .{ 72, -1 }, .{ 95, -1 }, .{ 113, 1 }, .{ 138, -1 }, .{ 159, 1 } };

/// PcdAbsolutePositionEmbedderv2's frequencies torch.pow(2.0, linspace(0, 10, 170)): correctly rounded but six.
pub fn pointFreqs(out: *[dc.pe_freqs]f32) void {
    linspace(out, 0, 10);
    for (out) |*f| f.* = @floatCast(@exp2(@as(f64, f.*)));
    for (torch_pow) |e| out[e[0]] = @bitCast(@as(i32, @bitCast(out[e[0]])) + e[1]);
}

/// The level modulation's parts in block order: shift, 1 + scale, gate for attention, then for the MLP; fp16.
pub const Mods = [6][H]f16;

/// The fp32 tensors the level embedding reads: l_embedder's MLP, then adaLN_modulation's linear.
pub const LevelWeights = struct { l1: []const f32, l1_b: []const f32, l2: []const f32, l2_b: []const f32, ada: []const f32, ada_b: []const f32 };

/// adaLN_modulation(l_embedder(2^level)) as fp16, the two scales as fp16(1 + scale), as the blocks read them.
pub fn levelMods(w: LevelWeights, level: u5, out: *Mods) void {
    const hf = dc.level_freqs / 2;
    const t: f32 = @floatFromInt(@as(u32, 1) << level);
    const lg: f32 = @floatCast(-@log(@as(f64, dc.level_period)));
    var freq: [dc.level_freqs]f32 = undefined;
    for (0..hf) |i| {
        const e: f32 = lg * @as(f32, @floatFromInt(i)) / @as(f32, hf);
        const a = ((t * @as(f32, @floatCast(@exp(@as(f64, e))))) * 2) * std.math.pi;
        freq[i] = @cos(a);
        freq[hf + i] = @sin(a);
    }
    var mid: [H]f32 = undefined;
    project(w.l1, w.l1_b, &freq, &mid);
    for (&mid) |*v| v.* = silu(v.*);
    var emb: [H]f32 = undefined;
    project(w.l2, w.l2_b, &mid, &emb);
    for (&emb) |*v| v.* = silu(v.*);
    var mod: [6 * H]f32 = undefined;
    project(w.ada, w.ada_b, &emb, &mod);
    for (mod, 0..) |v, i| {
        const part = i / H;
        out[part][i % H] = @floatCast(if (part == 1 or part == 4) half(1.0 + half(v)) else v);
    }
}

/// The parents' cell centres in [0, 1]: (coords + 0.5) / 2^(level - 1), fp32 [rows, 3].
pub fn centres(coords: []const [3]u32, level: u5, out: []f32) void {
    const res: f32 = @floatFromInt(@as(u32, 1) << (level - 1));
    for (coords, 0..) |c, r| for (0..3) |a| {
        out[r * 3 + a] = (@as(f32, @floatFromInt(c[a])) + 0.5) / res;
    };
}

/// softmax over each row of 8 logits, in place.
pub fn softmax(rows: []f32) void {
    var r: usize = 0;
    while (r < rows.len) : (r += P) {
        const x = rows[r..][0..P];
        var top = x[0];
        for (x[1..]) |v| top = @max(top, v);
        var sum: f32 = 0;
        for (x) |*v| {
            v.* = @exp(v.* - top);
            sum += v.*;
        }
        for (x) |*v| v.* /= sum;
    }
}

/// Where systematic_sample's uniform draws come from: replayed from a reference run, or a generator.
pub const Draws = union(enum) {
    native: std.Random.DefaultPrng,
    replay: *const Replay,

    /// The draw for the row of parent `coord` at `level`, `row` in the level's order.
    pub fn draw(d: *Draws, level: u5, coord: [3]u32, row: usize) f32 {
        return switch (d.*) {
            .native => |*g| g.random().float(f32),
            .replay => |r| r.draw(level, coord, row),
        };
    }

    /// The leaves' jitter value `i` (point i / 3, axis i % 3).
    pub fn jitter(d: *Draws, i: usize) f32 {
        return switch (d.*) {
            .native => |*g| g.random().float(f32),
            .replay => |r| if (i < r.jitter.len) r.jitter[i] else std.math.nan(f32),
        };
    }
};

/// A reference run's draws: each level's parents and their draws (NaN where torch drew none), and the jitter.
pub const Replay = struct {
    parents: [dc.max_level][]const [3]u32,
    draws: [dc.max_level][]const f32,
    jitter: []const f32,

    /// The reference's draw for the parent at `coord`: by row when the rows agree, else by searching its parents.
    pub fn draw(r: *const Replay, level: u5, coord: [3]u32, row: usize) f32 {
        const ps = r.parents[level - 1];
        if (row < ps.len and std.mem.eql(u32, &ps[row], &coord)) return r.draws[level - 1][row];
        for (ps, 0..) |p, i| if (std.mem.eql(u32, &p, &coord)) return r.draws[level - 1][i];
        return std.math.nan(f32);
    }
};

/// systematic_sample for one parent: `n` points among 8 children by probabilities `p` and offset draw `u`.
pub fn split(p: *const [P]f32, n: u32, u: f32, out: *[P]u32) void {
    out.* = @splat(0);
    if (n == 0) return;
    var q: [P]f32 = undefined;
    var sum: f32 = 0;
    for (&q, p) |*o, v| {
        o.* = @max(v, 0);
        sum += o.*;
    }
    const by = @max(sum, 1);
    for (&q) |*v| v.* = if (sum == 0) 1.0 / @as(f32, P) else v.* / by;
    // cdf clamped to fp32's 1 - 1e-12, which is 1
    var cdf: [P]f32 = undefined;
    var acc: f32 = 0;
    for (&cdf, q) |*c, v| {
        acc += v;
        c.* = @min(acc, 1.0);
    }
    const nf: f32 = @floatFromInt(n);
    const first = u / nf;
    for (0..n) |j| {
        const us = @min(first + @as(f32, @floatFromInt(j)) / nf, 1.0);
        var idx: usize = 0;
        while (idx < P and cdf[idx] < us) idx += 1;
        out[@min(idx, P - 1)] += 1;
    }
}

/// One octree level: the parents' counts split by `probs` [rows, 8]; the children that got points, in order.
pub const Level = struct {
    coords: []const [3]u32,
    counts: []const u32,
};

/// The 8 children of every parent, split by probs and draws, kept where they got points (parent-major, i + 2j + 4k).
pub fn descend(gpa: std.mem.Allocator, parents: Level, probs: []const f32, level: u5, draws: *Draws, splits: ?[]u32) !Level {
    std.debug.assert(probs.len == parents.coords.len * P);
    var coords: std.ArrayList([3]u32) = .empty;
    errdefer coords.deinit(gpa);
    var counts: std.ArrayList(u32) = .empty;
    errdefer counts.deinit(gpa);
    for (parents.coords, parents.counts, 0..) |c, n, r| {
        var got: [P]u32 = undefined;
        const u = if (n == 0) 0 else draws.draw(level, c, r);
        split(probs[r * P ..][0..P], n, u, &got);
        if (splits) |s| @memcpy(s[r * P ..][0..P], &got);
        for (got, 0..) |m, i| {
            if (m == 0) continue;
            const o = [3]u32{ @intCast(i & 1), @intCast((i >> 1) & 1), @intCast(i >> 2) };
            try coords.append(gpa, .{ c[0] * 2 + o[0], c[1] * 2 + o[1], c[2] * 2 + o[2] });
            try counts.append(gpa, m);
        }
    }
    return .{ .coords = try coords.toOwnedSlice(gpa), .counts = try counts.toOwnedSlice(gpa) };
}

/// The anchors: each leaf repeated by its count, (coords + jitter) / 2^level, fp32 [points, 3].
pub fn anchors(leaves: Level, level: u5, draws: *Draws, out: []f32) !void {
    var at: usize = 0;
    const res: f32 = @floatFromInt(@as(u32, 1) << level);
    for (leaves.coords, leaves.counts) |c, n| for (0..n) |_| {
        if (at + 3 > out.len) return error.AnchorCount;
        for (0..3) |a| {
            const j = draws.jitter(at + a);
            if (std.math.isNan(j)) return error.MissingDraw;
            out[at + a] = (@as(f32, @floatFromInt(c[a])) + j) / res;
        }
        at += 3;
    };
    if (at != out.len) return error.AnchorCount;
}

test "linspace takes each value from its nearer end with an fma, as torch's CPU kernel" {
    var v: [7]f32 = undefined;
    linspace(&v, 0, 1);
    try std.testing.expectEqual(@as(f32, 0), v[0]);
    try std.testing.expectEqual(@as(f32, 1), v[6]);
    var f: [dc.pe_freqs]f32 = undefined;
    pointFreqs(&f);
    try std.testing.expectEqual(@as(f32, 1), f[0]);
    try std.testing.expectEqual(@as(f32, 1024), f[dc.pe_freqs - 1]);
    try std.testing.expectEqual(@as(f32, 0.05917159840464592), blk: {
        var l: [dc.pe_freqs]f32 = undefined;
        linspace(&l, 0, 10);
        break :blk l[1];
    });
}

test "systematic resampling keeps the count and follows the cdf" {
    var out: [P]u32 = undefined;
    const p = [P]f32{ 0.5, 0, 0.25, 0, 0, 0, 0, 0.25 };
    split(&p, 8, 0.5, &out);
    try std.testing.expectEqualSlices(u32, &.{ 4, 0, 2, 0, 0, 0, 0, 2 }, &out);
    // one point: the draw alone picks the child
    split(&p, 1, 0.6, &out);
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 1, 0, 0, 0, 0, 0 }, &out);
    // a zero row falls back to uniform, and sums under 1 are not rescaled (torch divides by max(sum, 1))
    split(&@as([P]f32, @splat(0)), 8, 0.5, &out);
    try std.testing.expectEqualSlices(u32, &@as([P]u32, @splat(1)), &out);
    // searchsorted left: a draw landing on a cdf step goes to the lower child, so u = 0 doubles the first
    split(&@as([P]f32, @splat(0)), 8, 0.0, &out);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1, 1, 1, 1, 1, 1, 0 }, &out);
    const low = [P]f32{ 0.25, 0.25, 0, 0, 0, 0, 0, 0 };
    split(&low, 4, 0.5, &out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 0, 0, 0, 0, 0, 2 }, &out);
}

test "descend packs children parent-major in i + 2j + 4k order; anchors land inside their leaves" {
    const a = std.testing.allocator;
    const parents: Level = .{ .coords = &.{ .{ 0, 0, 0 }, .{ 1, 0, 1 } }, .counts = &.{ 2, 1 } };
    var probs: [2 * P]f32 = @splat(0);
    probs[3] = 1; // child (1, 1, 0) of the first parent
    probs[P + 4] = 1; // child (0, 0, 1) of the second
    var d: Draws = .{ .native = std.Random.DefaultPrng.init(1) };
    const next = try descend(a, parents, &probs, 2, &d, null);
    defer a.free(next.coords);
    defer a.free(next.counts);
    try std.testing.expectEqualSlices(u32, &.{ 2, 1 }, next.counts);
    try std.testing.expectEqual([3]u32{ 1, 1, 0 }, next.coords[0]);
    try std.testing.expectEqual([3]u32{ 2, 0, 3 }, next.coords[1]);
    var pts: [9]f32 = undefined;
    try anchors(next, 2, &d, &pts);
    for (0..3) |p| for (0..3) |ax| {
        const c: f32 = @floatFromInt(next.coords[if (p < 2) 0 else 1][ax]);
        try std.testing.expect(pts[p * 3 + ax] >= c / 4 and pts[p * 3 + ax] < (c + 1) / 4);
    };
}
