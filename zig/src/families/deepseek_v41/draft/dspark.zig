//! DSpark host glue: block ids/positions, context rows, keyed params, the chain rule; drafts only ever propose.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const Sampling = lanes.Sampling;

extern "c" fn exp(x: f64) f64;

/// The drafter's dimensions from the checkpoint's config (dspark_* and sliding_window).
pub const Shape = struct {
    block: u32 = 5, // dspark_block_size: drafts a pass
    noise: u32 = 128799, // dspark_noise_token_id: the block's rows after the anchor
    window: u32 = 128, // sliding_window: context rows a draft row sees
    rank: u32 = 256, // dspark_markov_rank
    hidden: u32 = 4096,
    candidates: u32 = 64, // a position's candidates, gathered over the ranks (world x min(64, 128 / world))

    /// The context ring's rows: a power of two holding the window and the block.
    pub fn ring(s: Shape) u32 {
        return std.math.ceilPowerOfTwo(u32, s.window + s.block) catch unreachable;
    }
};

/// Block ids [anchor, noise x (n - 1)] for one slot.
pub fn blockIds(s: Shape, anchor: u32, out: []u32) void {
    @memset(out, s.noise);
    out[0] = anchor;
}

/// A slot's draft rows sit at P .. P + n - 1 (P: the pending token's position).
pub fn positions(p: u64, out: []u64) void {
    for (out, 0..) |*o, i| o.* = p + i;
}

/// The context a draft row sees: positions first .. P - 1 (at most `window`, not before `valid`), ring slots in order.
pub const Context = struct {
    first: u64,
    count: u32,

    pub fn of(s: Shape, p: u64, valid: u64) Context {
        const a = @max(p -| s.window, valid);
        return .{ .first = a, .count = @intCast(if (p > a) p - a else 0) };
    }

    /// The ring slots of the context rows (the attention's "compressed" list), -1 past them; `out.len` = window.
    pub fn slots(c: Context, s: Shape, out: []i32) void {
        const ring = s.ring();
        @memset(out, -1);
        for (out[0..c.count], 0..) |*o, i| o.* = @intCast((c.first + i) % ring);
    }
};

/// The chain kernel's per-slot parameters: (seed, POS0, top_k, sampled) and (T, top_p, min_p).
pub const Params = struct {
    seed: u64 = 0,
    pos0: u64,
    top_k: u32 = 0,
    sampled: bool = false,
    temperature: f64 = 1.0,
    top_p: f64 = 1.0,
    min_p: f64 = 0.0,

    /// For a slot whose pending token sits at `p`: the first draft is the token at p + 1, keyed there.
    pub fn of(smp: ?Sampling, p: u64) Params {
        const s = smp orelse return .{ .pos0 = p + 1 };
        if (s.temperature <= 0.0) return .{ .pos0 = p + 1 };
        return .{ .seed = s.seed & ((@as(u64, 1) << 63) - 1), .pos0 = p + 1, .top_k = s.top_k, .sampled = true, .temperature = s.temperature, .top_p = s.top_p, .min_p = s.min_p };
    }

    pub fn sampling(p: Params) ?Sampling {
        if (!p.sampled) return null;
        return .{ .seed = p.seed, .temperature = p.temperature, .top_k = p.top_k, .top_p = p.top_p, .min_p = p.min_p };
    }
};

pub fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7FFF + ((u >> 16) & 1)) >> 16); // round to nearest even
}

/// The Markov head's tables (bf16 [V, rank]) and the confidence head's projection (fp32 [hidden + rank]).
pub const Markov = struct {
    w1: []const u16, // markov_head.embed
    w2: []const u16, // markov_head.head
    rank: u32,
    conf: ?[]const f32 = null,

    /// markov_w2[id] . markov_w1[prev] (bf16 products, fp32 sum).
    pub fn bias(m: Markov, prev: u32, id: u32) f32 {
        const a = m.w1[@as(usize, prev) * m.rank ..][0..m.rank];
        const b = m.w2[@as(usize, id) * m.rank ..][0..m.rank];
        var acc: f32 = 0.0;
        for (a, b) |x, y| acc += bf16(x) * bf16(y);
        return acc;
    }

    /// sigmoid(w_conf . [h ; markov_w1[prev]]).
    pub fn confidence(m: Markov, hid: []const f32, prev: u32) f32 {
        if (m.conf == null) return 0.5;
        var acc: [1]f32 = undefined;
        m.hidDots(hid, 1, &acc);
        return m.confidenceAfter(acc[0], hid.len, prev);
    }

    /// Each row's hidden confidence part w_conf[0..d] . h, rows in vector lanes, each in index order: same bits.
    pub fn hidDots(m: Markov, hid: []const f32, rows: usize, out: []f32) void {
        const cw = m.conf.?;
        const d = hid.len / rows;
        var r0: usize = 0;
        while (r0 < rows) : (r0 += vlanes) {
            const nr = @min(vlanes, rows - r0);
            var acc: V = @splat(0.0);
            var col: [vlanes][*]const f32 = undefined;
            for (0..vlanes) |j| col[j] = hid[(r0 + @min(j, nr - 1)) * d ..].ptr;
            for (0..d) |t| {
                var h: V = undefined;
                inline for (0..vlanes) |j| h[j] = col[j][t];
                acc += h * @as(V, @splat(cw[t]));
            }
            const sums: [vlanes]f32 = acc;
            for (0..nr) |j| out[r0 + j] = sums[j];
        }
    }

    /// A confidence from its hidden part (`hidDots`): the Markov part after `prev` added in order, then the sigmoid.
    pub fn confidenceAfter(m: Markov, hid_part: f32, d: usize, prev: u32) f32 {
        const cw = m.conf orelse return 0.5;
        var acc = hid_part;
        const me = m.w1[@as(usize, prev) * m.rank ..][0..m.rank];
        for (me, cw[d..][0..m.rank]) |x, w| acc += bf16(x) * w;
        return @floatCast(1.0 / (1.0 + exp(-@as(f64, acc))));
    }
};

/// Vector lanes of the chain's sums: each lane one candidate's (or row's) sum in its own sequential order.
const vlanes = 8;
const V = @Vector(vlanes, f32);

/// z = base + Markov bias after `prev` (-inf where id < 0); candidates in vector lanes, each in rank order: same bits.
pub fn markovZ(m: Markov, ids: []const i32, base: []const f32, prev: u32, z: []f32) void {
    const a = m.w1[@as(usize, prev) * m.rank ..][0..m.rank];
    var c0: usize = 0;
    while (c0 < ids.len) : (c0 += vlanes) {
        const nc = @min(vlanes, ids.len - c0);
        var row: [vlanes][*]const u16 = undefined;
        for (0..vlanes) |j| {
            const id = if (j < nc) ids[c0 + j] else -1;
            row[j] = if (id < 0) a.ptr else m.w2[@as(usize, @intCast(id)) * m.rank ..].ptr; // dead lanes: any row
        }
        var acc: V = @splat(0.0);
        for (a, 0..) |x, r| {
            var b: V = undefined;
            inline for (0..vlanes) |j| b[j] = bf16(row[j][r]);
            acc += @as(V, @splat(bf16(x))) * b;
        }
        const sums: [vlanes]f32 = acc;
        for (0..nc) |j| {
            const id = ids[c0 + j];
            z[c0 + j] = if (id < 0) -std.math.inf(f32) else base[c0 + j] + sums[j];
        }
    }
}

/// The first of (value desc, id asc) among live candidates.
pub fn greedy(ids: []const i32, z: []const f32) ?u32 {
    var best: ?usize = null;
    for (ids, z, 0..) |id, v, i| {
        if (id < 0) continue;
        if (best) |b| {
            if (v > z[b] or (v == z[b] and id < ids[b])) best = i;
        } else best = i;
    }
    return if (best) |b| @intCast(ids[b]) else null;
}

/// The keyed choice over a row's live candidates (greedy when `s` is null).
pub fn pick(scratch: Allocator, ids: []const i32, z: []const f32, position: u64, s: ?Sampling) !?u32 {
    const smp = s orelse return greedy(ids, z);
    var vals: std.ArrayList(f64) = .empty;
    defer vals.deinit(scratch);
    var live: std.ArrayList(u64) = .empty;
    defer live.deinit(scratch);
    for (ids, z) |id, v| if (id >= 0) {
        try vals.append(scratch, v);
        try live.append(scratch, @intCast(id));
    };
    if (live.items.len == 0) return null;
    return @intCast(try lanes.sampling.choose(scratch, vals.items, live.items, position, smp));
}

/// One slot's chain: candidates `ids` / base logits `base` [n][c] -> drafts and confidences [n].
pub fn chain(scratch: Allocator, m: Markov, ids: []const i32, base: []const f32, c: usize, anchor: u32, params: Params, hid: ?[]const f32, drafts: []u32, conf: []f32) !void {
    const z = try scratch.alloc(f32, c);
    defer scratch.free(z);
    const d = if (hid) |h| h.len / drafts.len else 0;
    // every row's hidden part of its confidence at once (it does not depend on the chain)
    const hp = try scratch.alloc(f32, drafts.len);
    defer scratch.free(hp);
    if (hid != null and m.conf != null) m.hidDots(hid.?[0 .. d * drafts.len], drafts.len, hp);
    var prev = anchor;
    for (drafts, conf, 0..) |*t, *q, i| {
        const row_ids = ids[i * c ..][0..c];
        markovZ(m, row_ids, base[i * c ..][0..c], prev, z);
        const tok = (try pick(scratch, row_ids, z, params.pos0 + i, params.sampling())) orelse prev;
        q.* = if (hid != null) m.confidenceAfter(hp[i], d, prev) else 0.5;
        t.* = tok;
        prev = tok;
    }
}

const Ranked = struct { value: f32, id: u32 };

fn rankedBefore(_: void, a: Ranked, b: Ranked) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// A row's best `ids.len` by (value desc, id asc) as token ids (`id0`: first column), bounded selection, no full sort.
pub fn candidates(scratch: Allocator, logits: []const f32, id0: u32, ids: []i32, vals: []f32) !void {
    const k = @min(ids.len, logits.len);
    const best = try scratch.alloc(Ranked, k);
    defer scratch.free(best);
    var n: usize = 0;
    for (logits, 0..) |v, i| {
        const r: Ranked = .{ .value = v, .id = @intCast(i) };
        if (n == k) {
            if (k == 0 or !rankedBefore({}, r, best[k - 1])) continue;
            n -= 1;
        }
        var j = n;
        while (j > 0 and rankedBefore({}, r, best[j - 1])) : (j -= 1) best[j] = best[j - 1];
        best[j] = r;
        n += 1;
    }
    for (ids[0..k], vals[0..k], best[0..k]) |*i, *v, r| {
        i.* = @intCast(r.id + id0);
        v.* = r.value;
    }
    @memset(ids[k..], -1);
    @memset(vals[k..], -std.math.inf(f32));
}

test "candidates: the bounded selection == the first k of a full sort (ties by id)" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(4101);
    const rnd = prng.random();
    const logits = try a.alloc(f32, 5000);
    defer a.free(logits);
    for (logits) |*v| v.* = @floatFromInt(rnd.intRangeAtMost(i32, -40, 40)); // many ties
    const all = try a.alloc(Ranked, logits.len);
    defer a.free(all);
    for (all, logits, 0..) |*r, v, i| r.* = .{ .value = v, .id = @intCast(i) };
    std.sort.pdq(Ranked, all, {}, rankedBefore);
    for ([_]usize{ 1, 3, 64, 200 }) |k| {
        var ids: [256]i32 = undefined;
        var vals: [256]f32 = undefined;
        try candidates(a, logits, 7, ids[0 .. k + 2], vals[0 .. k + 2]);
        for (all[0..k], ids[0..k], vals[0..k]) |r, i, v| {
            try std.testing.expectEqual(@as(i32, @intCast(r.id + 7)), i);
            try std.testing.expectEqual(r.value, v);
        }
    }
    // a row shorter than the list: its ids, then -1
    var ids: [4]i32 = undefined;
    var vals: [4]f32 = undefined;
    try candidates(a, &.{ 1.0, 3.0, 3.0 }, 10, &ids, &vals);
    try std.testing.expectEqualSlices(i32, &.{ 11, 12, 10, -1 }, &ids);
}

/// window = [pending, d_1 .. d_k], chosen = the target's choice after each row -> drafts kept.
pub fn accept(window: []const u32, chosen: []const u32) usize {
    var a: usize = 0;
    for (window[1..], chosen[0 .. window.len - 1]) |d, c| {
        if (d != c) break;
        a += 1;
    }
    return a;
}

test "the context ring and the block's ids" {
    const s: Shape = .{};
    try std.testing.expectEqual(@as(u32, 256), s.ring());
    const c = Context.of(s, 300, 0);
    try std.testing.expectEqual(@as(u64, 172), c.first);
    try std.testing.expectEqual(@as(u32, 128), c.count);
    var sl: [128]i32 = undefined;
    c.slots(s, &sl);
    try std.testing.expectEqual(@as(i32, 172), sl[0]);
    try std.testing.expectEqual(@as(i32, 299 % 256), sl[127]);
    try std.testing.expectEqual(@as(u32, 0), Context.of(s, 10, 10).count);
    var ids: [5]u32 = undefined;
    blockIds(s, 42, &ids);
    try std.testing.expectEqualSlices(u32, &.{ 42, 128799, 128799, 128799, 128799 }, &ids);
    try std.testing.expectEqual(@as(usize, 2), accept(&.{ 1, 5, 6, 9 }, &.{ 5, 6, 7, 8 }));
}
