//! sparse_index's properties on random keys, each with a planted fault its check must catch.
const std = @import("std");
const si = @import("sparse_index.zig");
const core_nvfp4 = @import("nvfp4_store.zig");
const testing = std.testing;
const gpa = testing.allocator;

const dim = 8;
const heads = 3;

fn randoms(seed: u64, n: usize) ![]f64 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try gpa.alloc(f64, n);
    for (out) |*v| v.* = prng.random().floatNorm(f64);
    return out;
}

const Rig = struct {
    keys: []f64,
    q: []f64,
    fn init(seed: u64, n: usize) !Rig {
        return .{ .keys = try randoms(seed, n * dim), .q = try randoms(seed + 1, heads * dim) };
    }
    fn deinit(r: Rig) void {
        gpa.free(r.keys);
        gpa.free(r.q);
    }
    fn sel(r: Rig, p: si.Policy, t: usize) !si.Selection {
        return si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, null);
    }
};

const flat: si.Policy = .{ .topk = 16, .sink = 2, .local = 4 };
const two: si.Policy = .{ .topk = 16, .block = 4, .pool_blocks = 6, .sink = 2, .local = 4 };

test "sparse index: every selection is causal, ascending, without repeats, keeps its mandatory keys and its budget" {
    const r = try Rig.init(1, 200);
    defer r.deinit();
    for ([_]si.Policy{ flat, two, .{ .topk = 16, .block = 4, .pool_blocks = 6, .sink = 2, .local = 4, .mandatory_in_topk = true } }) |p| {
        for ([_]usize{ 0, 1, 5, 17, 63, 64, 65, 199 }) |t| {
            const s = try r.sel(p, t);
            defer s.deinit(gpa);
            try si.check(p, t, s.keys);
        }
    }
}

test "sparse index: the check catches a future key, a repeat, a missing sink key and one key over budget (planted)" {
    const r = try Rig.init(2, 100);
    defer r.deinit();
    const s = try r.sel(flat, 60);
    defer s.deinit(gpa);
    const bad = try gpa.alloc(u32, s.keys.len + 1);
    defer gpa.free(bad);
    @memcpy(bad[0..s.keys.len], s.keys);
    bad[s.keys.len] = 61; // the next position: the future
    try testing.expectError(error.Future, si.check(flat, 60, bad));
    bad[s.keys.len] = s.keys[s.keys.len - 1]; // the last one again
    try testing.expectError(error.Unsorted, si.check(flat, 60, bad));
    try testing.expectError(error.MissingMandatory, si.check(flat, 60, s.keys[1..])); // key 0, a sink key
    const over = try gpa.alloc(u32, 61);
    defer gpa.free(over);
    for (over, 0..) |*o, i| o.* = @intCast(i);
    try testing.expectError(error.OverBudget, si.check(flat, 60, over));
}

test "sparse index: at most topk + mandatory keys in context: every key is attended, whatever the scores or stages" {
    const r = try Rig.init(3, 40);
    defer r.deinit();
    for ([_]si.Policy{ flat, two, .{ .topk = 22, .block = 4, .pool_blocks = 6 } }) |p| {
        const most = if (p.mandatory_in_topk) p.topk else p.topk + p.sink + p.local;
        const t = @min(most, 22) - 1; // 22 keys: within every policy's reach (two-stage: 6 blocks x 4 >= 22)
        const s = try r.sel(p, t);
        defer s.deinit(gpa);
        try testing.expectEqual(t + 1, s.keys.len);
    }
    // planted: one key past the reach of a flat policy without mandatory keys leaves exactly one out
    const p: si.Policy = .{ .topk = 16 };
    const s = try r.sel(p, 16);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(usize, 16), s.keys.len);
}

test "sparse index: blocks with every block kept, or one key a block, are the flat search" {
    const r = try Rig.init(4, 150);
    defer r.deinit();
    for ([_]usize{ 3, 40, 149 }) |t| {
        const a = try r.sel(flat, t);
        defer a.deinit(gpa);
        var p = flat;
        p.block = 4;
        p.pool_blocks = 0;
        const b = try r.sel(p, t);
        defer b.deinit(gpa);
        try testing.expectEqualSlices(u32, a.keys, b.keys);
        // the flat search rescoring a pool of every key agrees too
        const c = try si.select(gpa, flat, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, .{ .sel = &b, .owner = 0 });
        defer c.deinit(gpa);
        try testing.expectEqualSlices(u32, a.keys, c.keys);
    }
}

test "sparse index: two stages: tokens come only from the kept blocks or the mandatory ranges" {
    const r = try Rig.init(5, 300);
    defer r.deinit();
    const t = 250;
    const s = try r.sel(two, t);
    defer s.deinit(gpa);
    try testing.expect(s.pool.len <= two.pool_blocks * two.block);
    for (s.keys) |k| {
        const in_pool = std.mem.indexOfScalar(u32, s.pool, k) != null;
        try testing.expect(in_pool or k < two.sink or k + two.local > t);
    }
}

test "sparse index: the selection for position t ignores keys after t (chunked prompts, rolled-back drafts)" {
    const r = try Rig.init(6, 260);
    defer r.deinit();
    for ([_]si.Policy{ flat, two }) |p| for ([_]usize{ 10, 99, 200 }) |t| {
        const whole = try r.sel(p, t);
        defer whole.deinit(gpa);
        const cut = try si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = r.keys[0 .. (t + 1) * dim], .dim = dim }, t, null);
        defer cut.deinit(gpa);
        try testing.expectEqualSlices(u32, whole.keys, cut.keys);
        // drafts appended, then rejected: new keys past t change nothing for t
        const more = try gpa.dupe(f64, r.keys);
        defer gpa.free(more);
        for (more[(t + 1) * dim ..]) |*v| v.* *= -3;
        const drafted = try si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = more, .dim = dim }, t, null);
        defer drafted.deinit(gpa);
        try testing.expectEqualSlices(u32, whole.keys, drafted.keys);
    };
}

test "sparse index: reuse: final selection is the owner's positions; a pool rescored with another query can differ" {
    const r = try Rig.init(7, 400);
    defer r.deinit();
    const t = 380;
    const owner = try r.sel(two, t);
    defer owner.deinit(gpa);
    // final reuse takes the owner's keys as they are; pool reuse rescores the owner's candidates with its own query
    const other = try randoms(99, heads * dim);
    defer gpa.free(other);
    const follower = try si.select(gpa, two, .{ .q = other, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, .{ .sel = &owner, .owner = 0 });
    defer follower.deinit(gpa);
    try si.check(two, t, follower.keys);
    for (follower.keys) |k| try testing.expect(std.mem.indexOfScalar(u32, owner.pool, k) != null or k < two.sink or k + two.local > t);
    try testing.expect(!std.mem.eql(u32, follower.keys, owner.keys)); // a different query picks other tokens of the pool
    // the owner's own query over its own pool gives back its final selection
    const same = try si.select(gpa, two, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, .{ .sel = &owner, .owner = 0 });
    defer same.deinit(gpa);
    try testing.expectEqualSlices(u32, owner.keys, same.keys);
}

test "sparse index: a follower refuses a pool from another position or another owner (planted: both were accepted)" {
    const r = try Rig.init(8, 200);
    defer r.deinit();
    const at19 = try si.select(gpa, two, .{ .q = r.q, .heads = heads, .layer = 19 }, .{ .k = r.keys, .dim = dim }, 150, null);
    defer at19.deinit(gpa);
    try testing.expectEqual(@as(u32, 19), at19.owner);
    try testing.expectEqual(@as(usize, 150), at19.t);
    const q: si.Query = .{ .q = r.q, .heads = heads, .layer = 24 };
    const k: si.Keys = .{ .k = r.keys, .dim = dim };
    try testing.expectError(error.ForeignPool, si.select(gpa, two, q, k, 151, .{ .sel = &at19, .owner = 19 })); // another t
    try testing.expectError(error.ForeignPool, si.select(gpa, two, q, k, 150, .{ .sel = &at19, .owner = 29 })); // another owner
    const ok = try si.select(gpa, two, q, k, 150, .{ .sel = &at19, .owner = 19 });
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(u32, 24), ok.owner);
    // another sequence's pool at the same layer and position (planted: it was accepted)
    var other = q;
    other.seq = 2;
    try testing.expectError(error.ForeignPool, si.select(gpa, two, other, k, 150, .{ .sel = &at19, .owner = 19 }));
}

test "sparse index: summary precision: NVFP4 keys then the mean differs from the mean then NVFP4; each is selectable" {
    // one block of two keys a group of 16 wide: 1.0 and 1.5 store exactly; their mean 1.25 does not (E2M1 step 0.5)
    var keys: [2 * 16]f64 = undefined;
    for (0..16) |i| {
        keys[i] = if (i == 0) 6 else 1; // a group maximum of 6: scale 1, steps of 0.5 around 1
        keys[16 + i] = if (i == 0) 6 else 1.5;
    }
    const f: core_nvfp4.Format = .{};
    var out: [16]f64 = undefined;
    const base: si.Policy = .{ .topk = 1, .block = 2, .key_heads = 1 };
    try si.blockSummary(gpa, base, .{ .k = &keys, .dim = 16 }, 0, 1, &out);
    try testing.expectEqual(@as(f64, 1.25), out[1]); // as given
    var p = base;
    p.summary_nvfp4 = f; // mean, then stored: 1.25 rounds to 1.0 or 1.5
    try si.blockSummary(gpa, p, .{ .k = &keys, .dim = 16 }, 0, 1, &out);
    try testing.expect(out[1] != 1.25);
    p = base;
    p.keys_nvfp4 = f; // stored keys (exact here), then the mean: 1.25 stays
    try si.blockSummary(gpa, p, .{ .k = &keys, .dim = 16 }, 0, 1, &out);
    try testing.expectEqual(@as(f64, 1.25), out[1]);
    // the summary's input: original keys vs stored keys, with keys that do not store exactly (1.2 -> 1.0)
    for (16..32) |i| keys[i] = if (i == 16) 6 else 1.2;
    p.summary_input = .stored;
    try si.blockSummary(gpa, p, .{ .k = &keys, .dim = 16 }, 0, 1, &out);
    const stored = out[1];
    p.summary_input = .original;
    try si.blockSummary(gpa, p, .{ .k = &keys, .dim = 16 }, 0, 1, &out);
    try testing.expectApproxEqAbs(@as(f64, 1.1), out[1], 1e-12);
    try testing.expect(stored != out[1]);
}

test "sparse index: scores: ReLU drops negative heads, max takes the strongest, ties go to the lower position" {
    const q = [_]f64{ 1, 0, -1, 0 }; // two heads of dim 2
    const key = [_]f64{ 2, 0 };
    try testing.expectEqual(@as(f64, 2), si.score(.{ .topk = 1 }, .{ .q = &q, .heads = 2 }, &key));
    try testing.expectEqual(@as(f64, 0), si.score(.{ .topk = 1, .phi = .identity }, .{ .q = &q, .heads = 2 }, &key));
    try testing.expectEqual(@as(f64, 2), si.score(.{ .topk = 1, .reduce = .max, .phi = .identity }, .{ .q = &q, .heads = 2 }, &key));
    const w = [_]f64{ 0.5, 3 };
    try testing.expectEqual(@as(f64, 1), si.score(.{ .topk = 1 }, .{ .q = &q, .heads = 2, .weights = &w }, &key));
    // equal keys everywhere: the lowest positions win
    const keys: [20]f64 = @splat(1);
    const s = try si.select(gpa, .{ .topk = 3 }, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 9, null);
    defer s.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, s.keys);
}

test "sparse index: summaries: mean of unit keys and mean then compare differ on a block with one loud key" {
    // block 0: one key of norm 10 and three of norm 0.1 along another axis; block 1: middling keys along that axis
    const keys = [_]f64{ 10, 0, 0, 0.1, 0, 0.1, 0, 0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 };
    const q = [_]f64{ 0, 1 };
    const base: si.Policy = .{ .topk = 4, .block = 4, .pool_blocks = 1 };
    var unit = base;
    unit.norm_first = true;
    const a = try si.select(gpa, base, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 7, null);
    defer a.deinit(gpa);
    const b = try si.select(gpa, unit, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 7, null);
    defer b.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7 }, a.pool); // plain mean: block 1 (0.075 against 0.7)
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, b.pool); // unit keys: block 0 (1.06 against 1.0)
}

test "sparse index: NaN propagates through relu, a max over heads and blocks, and unit summaries (as the kernels)" {
    const nan = std.math.nan(f64);
    const q = [_]f64{ 1, -1 };
    const key = [_]f64{nan};
    try testing.expect(std.math.isNan(si.score(.{ .topk = 1 }, .{ .q = &q, .heads = 2 }, &key)));
    try testing.expect(std.math.isNan(si.score(.{ .topk = 1, .reduce = .max }, .{ .q = &q, .heads = 2 }, &key)));
    const keys = [_]f64{ 1, 2, nan, 3 };
    var out: [2]f64 = undefined;
    for ([_]si.Summary{ .max, .mean }) |s| for ([_]bool{ false, true }) |unit| {
        si.summarise(.{ .topk = 1, .block = 2, .summary = s, .norm_first = unit }, .{ .k = &keys, .dim = 2 }, 0, 1, &out);
        try testing.expect(std.math.isNan(out[0]));
        try testing.expect(std.math.isNan(out[1]) == unit); // a plain summary keeps the other dim's 2 and 3
    };
}

test "sparse index: dense is a rule: ten keys under a budget of 16 all attended; without it the open block drops" {
    // the review's counter-case: block 4 over 10 keys keeps blocks 0-1 only, so keys 8 and 9 would drop out
    const keys = try randoms(41, 10 * dim);
    defer gpa.free(keys);
    const q = try randoms(42, heads * dim);
    defer gpa.free(q);
    const p: si.Policy = .{ .topk = 16, .block = 4, .open_block = false, .dense = true };
    const s = try si.select(gpa, p, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, 9, null);
    defer s.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }, s.keys);
    // not a shortcut here: the same policy without the rule keeps blocks 0-1 only (keys 8 and 9 out)
    var off = p;
    off.dense = false;
    const n = try si.select(gpa, off, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, 9, null);
    defer n.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, n.keys);
    // one key past the budget: the stages decide again, and the open block's keys are gone
    const big = try randoms(43, 20 * dim);
    defer gpa.free(big);
    const t = try si.select(gpa, p, .{ .q = q, .heads = heads }, .{ .k = big, .dim = dim }, 16, null);
    defer t.deinit(gpa);
    try testing.expect(std.mem.indexOfScalar(u32, t.keys, 16) == null);
}

test "sparse index: blocks at their boundaries: the open block counts only when the policy keeps it" {
    const open: si.Policy = .{ .topk = 1, .block = 4 };
    const closed: si.Policy = .{ .topk = 1, .block = 4, .open_block = false };
    // keys 0..t: 7 keys (one open block of 3), 8 (two full), 9 (two full, one open of 1)
    try testing.expectEqual(@as(usize, 2), si.blocks(open, 6));
    try testing.expectEqual(@as(usize, 1), si.blocks(closed, 6));
    try testing.expectEqual(@as(usize, 2), si.blocks(open, 7));
    try testing.expectEqual(@as(usize, 2), si.blocks(closed, 7));
    try testing.expectEqual(@as(usize, 3), si.blocks(open, 8));
    try testing.expectEqual(@as(usize, 2), si.blocks(closed, 8));
    try testing.expectEqual(@as(usize, 0), si.blocks(closed, 2));
}

test "sparse index: one order for scores: NaN above +inf, -0 equal to +0 (ties to the lower position)" {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    // one head, one dim, identity: each key's score is its value
    const vals = [_]f64{ 1, -0.0, inf, nan, 0, -nan, 2, -inf };
    const one = [_]f64{1};
    const p: si.Policy = .{ .topk = 4, .phi = .identity };
    const s = try si.select(gpa, p, .{ .q = &one, .heads = 1 }, .{ .k = &vals, .dim = 1 }, vals.len - 1, null);
    defer s.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 2, 3, 5, 6 }, s.keys); // both NaNs, +inf, then 2
    const z = [_]f64{ -0.0, 0, -1 };
    const t = try si.select(gpa, .{ .topk = 1, .phi = .identity }, .{ .q = &one, .heads = 1 }, .{ .k = &z, .dim = 1 }, 2, null);
    defer t.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{0}, t.keys); // -0 ties +0: the lower position
    try testing.expect(si.above(nan, inf) and !si.above(inf, nan) and !si.above(0.0, -0.0) and !si.above(-0.0, 0.0));
}

test "sparse index: key heads and scale: query head h reads key head h / group, every score times scale" {
    // two key heads of one dim; four query heads, two a key head
    const key = [_]f64{ 2, -3 };
    const q = [_]f64{ 1, 1, 1, 1 };
    const p: si.Policy = .{ .topk = 1, .phi = .identity, .key_heads = 2, .scale = 0.5 };
    try testing.expectEqual(@as(f64, (2 + 2 - 3 - 3) * 0.5), si.score(p, .{ .q = &q, .heads = 4 }, &key));
    var relu = p;
    relu.phi = .relu;
    try testing.expectEqual(@as(f64, 2), si.score(relu, .{ .q = &q, .heads = 4 }, &key));
}

test "sparse index: the pool shortcut equals every stage; the dense rule equals them only with open blocks scored" {
    const n = 70;
    const keys = try randoms(51, n * dim);
    defer gpa.free(keys);
    const q = try randoms(52, heads * dim);
    defer gpa.free(q);
    // block 8: up to 9 blocks; pool 9 keeps them all (bypass), topk 12 (dense up to 12 keys, with 2 sinks 14)
    for ([_]si.Policy{ .{ .topk = 12, .block = 8, .pool_blocks = 9 }, .{ .topk = 12, .block = 8, .pool_blocks = 9, .sink = 2 }, .{ .topk = 12, .block = 8 } }) |p| {
        var d = p;
        d.dense = true; // these configs score the open block: the rule changes nothing, and the claim stops there
        for (0..n) |t| {
            const a = try si.select(gpa, d, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, t, null);
            defer a.deinit(gpa);
            const b = try si.selectStaged(gpa, p, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, t, null);
            defer b.deinit(gpa);
            try testing.expectEqualSlices(u32, b.keys, a.keys);
            if (t + 1 > si.budget(p, t)) try testing.expectEqualSlices(u32, b.pool, a.pool);
        }
    }
    // planted: a pool of 3 blocks keeps fewer than all, so the shortcut must not apply there (staged differs from flat)
    const small: si.Policy = .{ .topk = 12, .block = 8, .pool_blocks = 3 };
    var differ = false;
    for (24..n) |t| {
        const a = try si.select(gpa, small, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, t, null);
        defer a.deinit(gpa);
        const f = try si.select(gpa, .{ .topk = 12 }, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = dim }, t, null);
        defer f.deinit(gpa);
        if (!std.mem.eql(u32, a.keys, f.keys)) differ = true;
    }
    try testing.expect(differ);
}

test "sparse index: NVFP4-stored keys move picks across the top-k boundary; flips counts them" {
    const n = 512;
    const w = 16; // NVFP4 groups of 16
    const keys = try randoms(61, n * w);
    defer gpa.free(keys);
    var flipped: usize = 0;
    var picks: usize = 0;
    for (0..8) |i| {
        const q = try randoms(70 + i, heads * w);
        defer gpa.free(q);
        const p: si.Policy = .{ .topk = 64 };
        var fp = p;
        fp.keys_nvfp4 = .{};
        const a = try si.select(gpa, p, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = w }, n - 1, null);
        defer a.deinit(gpa);
        const b = try si.select(gpa, fp, .{ .q = q, .heads = heads }, .{ .k = keys, .dim = w }, n - 1, null);
        defer b.deinit(gpa);
        try testing.expectEqual(a.keys.len, b.keys.len);
        flipped += si.flips(a.keys, b.keys);
        picks += a.keys.len;
    }
    // close scores near the boundary trade places, far ones never: some flips, most picks kept
    try testing.expect(flipped > 0 and flipped * 4 < picks);
    try testing.expectEqual(@as(usize, 0), si.flips(&.{ 1, 4, 9 }, &.{ 1, 4, 9 }));
    try testing.expectEqual(@as(usize, 2), si.flips(&.{ 1, 4, 9 }, &.{ 1, 5, 8 }));
}
