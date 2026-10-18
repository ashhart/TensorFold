//! First-position siblings for DSpark rounds, each its own Markov chain beside the main one; cost model picks the tree.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const dspark = @import("dspark.zig");

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

pub const max_siblings = 3;
const max_nodes = 64;

/// What a sibling costs: `ms` a duplicated pending row; `rows`: duplicates take window rows (shadow slots).
pub const Dup = struct { ms: f64 = 0.0, rows: bool = false };

pub const Plan = struct { k: usize, lens: [max_siblings]usize, ms: f64, expected: f64 };

const Node = struct { p: f64, kind: u2, j: u8 }; // kind 0 main, 1 sibling root, 2 sibling row

fn likelier(_: void, a: Node, b: Node) bool {
    return a.p > b.p;
}

/// Tree maximizing E - rate (verify[R-1] + dup.ms x siblings) over likeliest nodes; `qm` main, `sibs`, `cont` chances.
pub fn plan(qm: []const f64, sibs: []const f64, cont: []const f64, verify: []const f64, rate: f64, least: usize, dup: Dup, sib_rows: usize, max_rows: usize) Plan {
    const top = @min(verify.len, max_rows);
    var nodes: [max_nodes]Node = undefined;
    var n: usize = 0;
    var p: f64 = 1.0;
    for (qm) |q| {
        p *= q;
        nodes[n] = .{ .p = p, .kind = 0, .j = 0 };
        n += 1;
    }
    for (sibs[0..@min(sibs.len, max_siblings)], 1..) |sp, j| {
        p = sp;
        for (0..sib_rows) |i| {
            if (i > 0) p *= if (i - 1 < cont.len) cont[i - 1] else 0.0;
            if (p <= 0.0 or n == max_nodes) break;
            nodes[n] = .{ .p = p, .kind = if (i == 0) 1 else 2, .j = @intCast(j) };
            n += 1;
        }
    }
    const head: usize = @intFromBool(qm.len > 0); // the main chain's first draft before any sibling
    std.sort.insertion(Node, nodes[head..n], {}, likelier); // stable: parents stay before children
    var out: Plan = .{ .k = 0, .lens = @splat(0), .ms = 0.0, .expected = 0.0 };
    var best_s = -std.math.inf(f64);
    var e: f64 = 1.0;
    var dups: usize = 0;
    var best: usize = 0;
    for (0..n + 1) |m| {
        if (m > 0) {
            e += nodes[m - 1].p;
            dups += @intFromBool(nodes[m - 1].kind == 1);
        }
        if (m + 1 + (if (dup.rows) dups else 0) > top) break;
        if (m < least) continue;
        const ms = verify[m] + dup.ms * @as(f64, @floatFromInt(dups));
        const s = e - rate * ms;
        if (s > best_s + 1e-12) {
            best = m;
            best_s = s;
            out.expected = e;
            out.ms = ms;
        }
    }
    for (nodes[0..best]) |x| {
        if (x.kind == 0) out.k += 1 else out.lens[x.j - 1] += 1;
    }
    return out;
}

/// TF_DSV41_TREE (0 off, 2-4 branches), _TREE_DUP_MS (0.4), _TREE_ROWS (4 a branch); _TREE_PC is refused.
pub const Settings = struct {
    siblings: u32 = 0,
    sib_rows: u32 = 4,
    dup: Dup = .{ .ms = 0.4, .rows = true },

    pub fn parse(tree: ?[]const u8, dup_ms: ?[]const u8, rows: ?[]const u8, pc: ?[]const u8) !Settings {
        const b = try std.fmt.parseInt(u32, trimmed(tree) orelse "0", 10);
        if (b == 1 or b > max_siblings + 1) return error.TreeBranches;
        const ms = try std.fmt.parseFloat(f64, trimmed(dup_ms) orelse "0.4");
        const r = try std.fmt.parseInt(u32, trimmed(rows) orelse "4", 10);
        if (ms < 0.0 or r < 1 or r > 15) return error.TreeKnobs;
        if (trimmed(pc)) |v| if (!std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "off") and !std.mem.eql(u8, v, "false")) return error.TreePcNotPorted;
        return .{ .siblings = if (b == 0) 0 else b - 1, .sib_rows = r, .dup = .{ .ms = ms, .rows = true } };
    }

    pub fn fromEnv() !Settings {
        return parse(env("TF_DSV41_TREE"), env("TF_DSV41_TREE_DUP_MS"), env("TF_DSV41_TREE_ROWS"), env("TF_DSV41_TREE_PC"));
    }

    fn env(name: [*:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
    }

    fn trimmed(v: ?[]const u8) ?[]const u8 {
        const t = std.mem.trim(u8, v orelse return null, " \t");
        return if (t.len == 0) null else t;
    }
};

pub const Sibling = struct { token: u32, p: f64 };

const Cand = struct { z: f64, id: u32, score: f64 = 0.0, p: f64 = 0.0, at: usize = 0 };

fn byZ(_: void, a: Cand, b: Cand) bool {
    return a.z > b.z or (a.z == b.z and a.id < b.id);
}

fn byScore(_: void, a: Cand, b: Cand) bool {
    return a.score > b.score or (a.score == b.score and a.at < b.at);
}

/// The first `out.len` tokens of `z` over `ids` not in `exclude`, in keyed order, with their draft probabilities.
pub fn ranked(scratch: Allocator, ids: []const i32, z: []const f32, position: u64, sampling: ?lanes.Sampling, exclude: []const u32, out: []Sibling) !usize {
    var c: std.ArrayList(Cand) = .empty;
    defer c.deinit(scratch);
    for (ids, z) |id, v| if (id >= 0) try c.append(scratch, .{ .z = v, .id = @intCast(id) });
    if (c.items.len == 0 or out.len == 0) return 0;
    std.sort.pdq(Cand, c.items, {}, byZ);
    var keep = c.items;
    const probs = try scratch.alloc(f64, keep.len);
    defer scratch.free(probs);
    if (sampling) |s| {
        keep = keep[0..@max(1, @min(if (s.top_k != 0) @as(usize, s.top_k) else keep.len, keep.len))];
        const t = @max(s.temperature, 1e-6);
        for (keep) |*x| x.z /= t; // z is now the scaled value
        var top = keep[0].z;
        for (keep) |x| top = @max(top, x.z);
        for (keep, probs[0..keep.len]) |x, *q| q.* = exp(x.z - top);
        normalize(probs[0..keep.len]);
        if (s.top_p > 0.0 and s.top_p < 1.0) {
            var cum: f64 = 0.0;
            var below: usize = 0;
            for (probs[0..keep.len]) |q| {
                cum += q;
                below += @intFromBool(cum < s.top_p);
            }
            keep = keep[0..@min(keep.len, below + 1)];
        }
        if (s.min_p > 0.0) {
            const floor = keep[0].z + s.minLog();
            var n: usize = 0;
            for (keep) |x| n += @intFromBool(x.z >= floor);
            keep = keep[0..n];
        }
        normalize(probs[0..keep.len]);
        for (keep, probs[0..keep.len], 0..) |*x, q, i| {
            x.p = q;
            x.at = i;
            x.score = x.z - log(-log(lanes.sampling.uniform(s.seed, position, x.id)));
        }
        std.sort.pdq(Cand, keep, {}, byScore);
    } else {
        for (keep, probs) |x, *q| q.* = exp(x.z - keep[0].z);
        normalize(probs);
        for (keep, probs) |*x, q| x.p = q;
    }
    var n: usize = 0;
    for (keep) |x| {
        if (std.mem.indexOfScalar(u32, exclude, x.id) != null) continue;
        out[n] = .{ .token = x.id, .p = x.p };
        n += 1;
        if (n == out.len) break;
    }
    return n;
}

fn normalize(p: []f64) void {
    const total = lanes.sampling.pairwiseSum(p);
    for (p) |*x| x.* /= total;
}

/// The Markov chain behind `first`: the drafts of positions 1 .. rows - 1 (the chain rule keyed at pos0 + i).
pub fn continuation(scratch: Allocator, m: dspark.Markov, ids: []const i32, base: []const f32, c: usize, first: u32, rows: usize, params: dspark.Params, out: []u32) !usize {
    const z = try scratch.alloc(f32, c);
    defer scratch.free(z);
    const n_rows = ids.len / c;
    var prev = first;
    var n: usize = 0;
    var i: usize = 1;
    while (i < @min(rows, n_rows)) : (i += 1) {
        const row = ids[i * c ..][0..c];
        dspark.markovZ(m, row, base[i * c ..][0..c], prev, z);
        const tok = (try dspark.pick(scratch, row, z, params.pos0 + i, params.sampling())) orelse break;
        out[n] = tok;
        n += 1;
        prev = tok;
    }
    return n;
}

test "tree settings read as Python's tree.settings" {
    const off = try Settings.parse(null, null, null, null);
    try std.testing.expectEqual(@as(u32, 0), off.siblings);
    const on = try Settings.parse("3", " 0.5 ", "6", "0");
    try std.testing.expectEqual(@as(u32, 2), on.siblings);
    try std.testing.expectEqual(@as(u32, 6), on.sib_rows);
    try std.testing.expect(on.dup.rows and on.dup.ms == 0.5);
    try std.testing.expectError(error.TreeBranches, Settings.parse("1", null, null, null));
    try std.testing.expectError(error.TreeKnobs, Settings.parse("2", null, "16", null));
    try std.testing.expectError(error.TreePcNotPorted, Settings.parse("2", null, null, "1"));
}

test "a confident main chain takes no sibling; an unsure first draft buys one" {
    var verify: [16]f64 = undefined;
    for (&verify, 0..) |*v, r| v.* = 21.4 + 4.0 * @as(f64, @floatFromInt(r));
    const sure = plan(&.{ 0.95, 0.95, 0.95 }, &.{0.03}, &.{ 0.95, 0.95 }, &verify, 0.05, 1, .{}, 4, 16);
    try std.testing.expectEqual(@as(usize, 0), sure.lens[0]);
    const unsure = plan(&.{ 0.4, 0.9 }, &.{0.45}, &.{0.9}, &verify, 0.03, 1, .{}, 4, 16);
    try std.testing.expect(unsure.lens[0] >= 1 and unsure.k >= 1);
}
