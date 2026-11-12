//! The DFlash2 drafter's kernels: copies byte for byte, the conv by its fp32 steps, the attention against float64.
const std = @import("std");
const rig = @import("rig.zig");
const draft = @import("../ops/draft.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;
const Tensor = @import("../ops/types.zig").Tensor;
const Kind = @import("../ops/types.zig").Kind;
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const untouched: u16 = 0xABAB;

fn kind(t: *const Rig) Kind {
    return if (t.bf16) .bf16 else .f16;
}

/// `v` rounded to the activation (to nearest even), as the kernels' `rounded` does.
fn round(t: *const Rig, v: f32) f32 {
    if (!t.bf16) return @as(f16, @floatCast(v));
    const u: u32 = @bitCast(v);
    const r = (u +% (0x7fff + ((u >> 16) & 1))) & 0xffff0000;
    return @bitCast(r);
}

fn bitsOf(t: *const Rig, v: f32) u16 {
    if (!t.bf16) return @bitCast(@as(f16, @floatCast(v)));
    return @intCast(@as(u32, @bitCast(round(t, v))) >> 16);
}

/// `n` random 16-bit activations and their values.
fn randomAct(t: *const Rig, rng: *rig.Rng, n: usize, scale: f32) !struct { []u16, []f64 } {
    const b = try gpa.alloc(u16, n);
    errdefer gpa.free(b);
    const v = try gpa.alloc(f64, n);
    for (b, v) |*x, *y| {
        x.* = t.bits(rng.unit() * scale);
        y.* = t.value(x.*);
    }
    return .{ b, v };
}

test "a gather copies tap ring rows from a start, wrapping the ring, and a kept path's rows by value" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xD1A };
    const ring_rows = 10;
    const width = 3 * 64; // three taps of hidden 64
    const ring = try gpa.alloc(u16, ring_rows * width);
    defer gpa.free(ring);
    for (ring) |*x| x.* = @truncate(rng.next());
    var dev_ring = try t.upload(ring);
    defer dev_ring.free();
    var out = try t.alloc(8 * width * 2);
    defer out.free();
    const o = t.ops(&t.on);
    const start = 2 * ring_rows + 7; // rows 7, 8, 9, 0, 1, 2
    try draft.gatherTaps(o, at(dev_ring), ring_rows, start, 6, width, at(out));
    try t.stream.synchronize();
    var got = try rig.download(u16, out);
    defer gpa.free(got);
    for (0..6) |i| {
        const row = (start + i) % ring_rows;
        try std.testing.expectEqualSlices(u16, ring[row * width ..][0..width], got[i * width ..][0..width]);
    }
    const path = [_]u32{ 0, 2, 5, 3 };
    try draft.gatherRows(o, at(dev_ring), ring_rows, start, &path, width, at(out));
    try t.stream.synchronize();
    gpa.free(got);
    got = try rig.download(u16, out);
    for (path, 0..) |r, i| {
        const row = (start + r) % ring_rows;
        try std.testing.expectEqualSlices(u16, ring[row * width ..][0..width], got[i * width ..][0..width]);
    }
    var many: [draft.most_rows + 1]u32 = @splat(0);
    try std.testing.expectError(error.BadShape, draft.gatherRows(o, at(dev_ring), ring_rows, 0, &many, width, at(out)));
    try std.testing.expectError(error.BadShape, draft.gatherTaps(o, at(dev_ring), ring_rows, 0, 1, width + 1, at(out)));
}

test "a ring put stores fp32 keys rounded to the activation and values as they are at each row's slot" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xD1B };
    const ring_rows = 6;
    const width = 2 * 128;
    const rows = 4;
    const pos0 = 3 * ring_rows + 4; // slots 4, 5, 0, 1
    const keys = try gpa.alloc(f32, rows * width);
    defer gpa.free(keys);
    for (keys) |*k| k.* = rng.unit() * 3 + 0x1p-12;
    const vals, const vv = try randomAct(t, &rng, rows * width, 2);
    defer gpa.free(vals);
    defer gpa.free(vv);
    const fill = try gpa.alloc(u16, ring_rows * width);
    defer gpa.free(fill);
    @memset(fill, untouched);
    var rk = try t.upload(fill);
    defer rk.free();
    var rv = try t.upload(fill);
    defer rv.free();
    var dk = try t.upload(keys);
    defer dk.free();
    var dv = try t.upload(vals);
    defer dv.free();
    const o = t.ops(&t.on);
    try draft.ringPut(o, at(dk), .{ .ptr = at(dv), .kind = kind(t) }, at(rk), at(rv), rows, width, pos0, ring_rows);
    try t.stream.synchronize();
    const got_k = try rig.download(u16, rk);
    defer gpa.free(got_k);
    const got_v = try rig.download(u16, rv);
    defer gpa.free(got_v);
    const want_k = try gpa.dupe(u16, fill);
    defer gpa.free(want_k);
    const want_v = try gpa.dupe(u16, fill);
    defer gpa.free(want_v);
    for (0..rows) |i| {
        const slot = (pos0 + i) % ring_rows;
        for (0..width) |c| want_k[slot * width + c] = bitsOf(t, keys[i * width + c]);
        @memcpy(want_v[slot * width ..][0..width], vals[i * width ..][0..width]);
    }
    try std.testing.expectEqualSlices(u16, want_k, got_k);
    try std.testing.expectEqualSlices(u16, want_v, got_v);
    const wide: Tensor = .{ .ptr = at(dv), .kind = .f32 };
    try std.testing.expectError(error.BadShape, draft.ringPut(o, at(dk), wide, at(rk), at(rv), rows, width, 0, ring_rows));
}

test "the dynamic conv matches its fp32 steps rounded a step, and float64, within each stream's block" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xD1C };
    const d = 512;
    const group = 16;
    const groups = d / group;
    const block = 8;
    const rows = 2 * block; // two streams' blocks: a stream's first row sees no lag
    const x, const xv = try randomAct(t, &rng, rows * d, 2);
    defer gpa.free(x);
    defer gpa.free(xv);
    const dyn, const dv = try randomAct(t, &rng, rows * 4 * groups, 1);
    defer gpa.free(dyn);
    defer gpa.free(dv);
    const res, const rv = try randomAct(t, &rng, rows * d, 4);
    defer gpa.free(res);
    defer gpa.free(rv);
    const base = try gpa.alloc(f32, 4 * d);
    defer gpa.free(base);
    for (base) |*b| b.* = rng.unit() * 0.5;
    var bx = try t.upload(x);
    defer bx.free();
    var bd = try t.upload(dyn);
    defer bd.free();
    var br = try t.upload(res);
    defer br.free();
    var bb = try t.upload(base);
    defer bb.free();
    var out = try t.alloc(rows * d * 2);
    defer out.free();
    const o = t.ops(&t.on);
    const k = kind(t);
    for ([_]usize{ 0, 1 }) |branch| for ([_]bool{ false, true }) |with_res| {
        const r: ?Tensor = if (with_res) .{ .ptr = at(br), .kind = k } else null;
        try draft.dynConv(o, .{ .ptr = at(bx), .kind = k }, .{ .ptr = at(bd), .kind = k }, at(bb), r, .{ .ptr = at(out), .kind = k }, rows, d, group, branch, block);
        try t.stream.synchronize();
        const got = try rig.download(u16, out);
        defer gpa.free(got);
        for (0..rows) |row| for (0..d) |c| {
            var total: f32 = 0;
            var exact: f64 = 0;
            var mag: f64 = 0;
            for (0..2) |l| {
                if (row % block < l) continue;
                const xi = (row - l) * d + c;
                const xf: f32 = @floatCast(xv[xi]);
                const bf = base[(branch * 2 + l) * d + c];
                const df: f32 = @floatCast(dv[row * 4 * groups + (branch * 2 + l) * groups + c / group]);
                total = round(t, total + round(t, bf * xf));
                total = round(t, total + round(t, df * xf));
                exact += (@as(f64, bf) + df) * xv[xi];
                mag += (@abs(@as(f64, bf)) + @abs(df)) * @abs(xv[xi]);
            }
            if (with_res) {
                total = round(t, total + @as(f32, @floatCast(rv[row * d + c])));
                exact += rv[row * d + c];
                mag += @abs(rv[row * d + c]);
            }
            const g = got[row * d + c];
            try rig.expect(g == bitsOf(t, total), "dyn conv branch {d} res {} row {d} col {d}: 0x{x:0>4}, want 0x{x:0>4}", .{ branch, with_res, row, c, g, bitsOf(t, total) });
            // five roundings to the activation, each within half its ulp of what it rounds
            const ulp: f64 = if (t.bf16) 0x1p-8 else 0x1p-11;
            const err = @abs(t.value(g) - exact);
            try rig.expect(err <= 5 * ulp * (mag + @abs(exact)), "dyn conv row {d} col {d}: {d} vs float64 {d}", .{ row, c, t.value(g), exact });
        };
    };
    const wide: Tensor = .{ .ptr = at(bx), .kind = .f32 };
    try std.testing.expectError(error.BadShape, draft.dynConv(o, wide, wide, at(bb), null, wide, rows, d, group, 0, block));
}

test "mask rows drops each stream's first block row and packs the rest in stream order" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xD1D };
    const d = 256;
    const block = 8;
    const streams = 3;
    const x = try gpa.alloc(u16, streams * block * d);
    defer gpa.free(x);
    for (x) |*v| v.* = @truncate(rng.next());
    var bx = try t.upload(x);
    defer bx.free();
    var out = try t.alloc(streams * (block - 1) * d * 2);
    defer out.free();
    const k = kind(t);
    try draft.maskRows(t.ops(&t.on), .{ .ptr = at(bx), .kind = k }, .{ .ptr = at(out), .kind = k }, streams, d, block);
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    for (0..streams) |s| for (1..block) |r| {
        const from = (s * block + r) * d;
        const to = (s * (block - 1) + r - 1) * d;
        try std.testing.expectEqualSlices(u16, x[from..][0..d], got[to..][0..d]);
    };
}

const heads = 32;
const kv_heads = 8;
const hd = draft.head;
const kv_width = kv_heads * hd;

/// A frame of streams over one shared geometry: rings of `ring_rows` rows of `layers` layers, a block each.
const Frame = struct {
    ring_rows: usize,
    window: usize,
    layer: usize,
    layers: usize,
    segs: []const struct { begin: usize, position: usize },
};

fn attentionCase(t: *Rig, f: Frame, block: usize, masked: bool) !void {
    var rng: rig.Rng = .{ .state = 0xD1E + block };
    const streams = f.segs.len;
    const rows = streams * block;
    const ring_elems = f.layers * 2 * f.ring_rows * kv_width;
    var rings: [4]DeviceBuffer = undefined;
    var ring_vals: [4][]f64 = undefined;
    var ring_bits: [4][]u16 = undefined;
    for (0..streams) |s| {
        ring_bits[s], ring_vals[s] = try randomAct(t, &rng, ring_elems, 1);
        rings[s] = try t.upload(ring_bits[s]);
    }
    defer for (0..streams) |s| {
        rings[s].free();
        gpa.free(ring_bits[s]);
        gpa.free(ring_vals[s]);
    };
    const q = try gpa.alloc(f32, rows * heads * hd);
    defer gpa.free(q);
    for (q) |*v| v.* = @floatCast(t.value(t.bits(rng.unit() * 2)));
    const kb = try gpa.alloc(f32, rows * kv_width);
    defer gpa.free(kb);
    for (kb) |*v| v.* = @floatCast(t.value(t.bits(rng.unit() * 2)));
    const vb, const vv = try randomAct(t, &rng, rows * kv_width, 1);
    defer gpa.free(vb);
    defer gpa.free(vv);
    const mask = try gpa.alloc(u32, rows);
    defer gpa.free(mask);
    // a chain's causal block: row m sees block rows 0..m; unmasked, every block row
    for (mask, 0..) |*w, m| w.* = if (masked) (@as(u32, 2) << @intCast(m % block)) - 1 else 0;
    var segs: [4]draft.Segment = undefined;
    for (f.segs, 0..) |sg, s| segs[s] = .{ .ring = at(rings[s]), .begin = @intCast(sg.begin), .position = @intCast(sg.position) };
    var dq = try t.upload(q);
    defer dq.free();
    var dkb = try t.upload(kb);
    defer dkb.free();
    var dvb = try t.upload(vb);
    defer dvb.free();
    var dm = try t.upload(mask);
    defer dm.free();
    var ds = try t.upload(segs[0..streams]);
    defer ds.free();
    var parts = try t.alloc(draft.Partials.bytes(rows, heads));
    defer parts.free();
    var out = try t.alloc(rows * heads * hd * 2);
    defer out.free();
    const k = kind(t);
    const shape: draft.Attention = .{ .rows = rows, .block_rows = block, .heads = heads, .kv_heads = kv_heads, .ring_rows = f.ring_rows, .window = f.window, .layer = f.layer };
    const p = draft.Partials.at(at(parts), rows, heads);
    try draft.attention(t.ops(&t.on), at(dq), at(dkb), .{ .ptr = at(dvb), .kind = k }, if (masked) at(dm) else null, at(ds), shape, p, .{ .ptr = at(out), .kind = k });
    try t.stream.synchronize();
    const got = try rig.download(u16, out);
    defer gpa.free(got);
    const scores = try gpa.alloc(f64, f.window + block);
    defer gpa.free(scores);
    const ulp: f64 = if (t.bf16) 0x1p-8 else 0x1p-11;
    for (0..rows) |m| for (0..heads) |h| {
        const s = m / block;
        const local = m % block;
        const sg = f.segs[s];
        const kvh = h / (heads / kv_heads);
        const first = @min(sg.position, @max(sg.begin, (sg.position + local + 1) -| f.window));
        const ring_keys = sg.position - first;
        const kr = f.layer * 2 * f.ring_rows * kv_width;
        const vr = kr + f.ring_rows * kv_width;
        const qv = q[(m * heads + h) * hd ..][0..hd];
        var top = -std.math.inf(f64);
        for (0..ring_keys + block) |i| {
            scores[i] = -std.math.inf(f64);
            if (i >= ring_keys and masked and (mask[m] >> @intCast(i - ring_keys)) & 1 == 0) continue;
            var dot: f64 = 0;
            for (0..hd) |j| {
                const key = if (i < ring_keys) ring_vals[s][kr + ((first + i) % f.ring_rows) * kv_width + kvh * hd + j] else kb[((s * block + i - ring_keys) * kv_heads + kvh) * hd + j];
                dot += qv[j] * key;
            }
            scores[i] = dot / @sqrt(@as(f64, hd));
            top = @max(top, scores[i]);
        }
        var sum: f64 = 0;
        for (scores[0 .. ring_keys + block]) |*x| {
            x.* = @exp(x.* - top);
            sum += x.*;
        }
        for (0..hd) |j| {
            var want: f64 = 0;
            for (0..ring_keys + block) |i| {
                const v = if (i < ring_keys) ring_vals[s][vr + ((first + i) % f.ring_rows) * kv_width + kvh * hd + j] else vv[(s * block + i - ring_keys) * kv_width + kvh * hd + j];
                want += scores[i] / sum * v;
            }
            const g = t.value(got[(m * heads + h) * hd + j]);
            // the output's rounding, and fp32 scores, exp and sums over at most 2304 keys of values within 1
            const allow = ulp * @abs(want) + 0x1p-14;
            try rig.expect(@abs(g - want) <= allow, "draft attention block {d} masked {} row {d} head {d} dim {d}: {d}, float64 {d}", .{ block, masked, m, h, j, g, want });
        }
    };
}

test "the block attention sees each stream's ring window and its block rows, in float64's bound" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    // stream 0: a context longer than the window, wrapping its ring; 1: shorter, from its begin; 2: no ring keys
    const f: Frame = .{
        .ring_rows = 316,
        .window = 300,
        .layer = 1,
        .layers = 2,
        .segs = &.{ .{ .begin = 0, .position = 1000 }, .{ .begin = 640, .position = 700 }, .{ .begin = 50, .position = 50 } },
    };
    try attentionCase(t, f, 8, false);
    try attentionCase(t, f, 8, true);
    // the stock window: keys past one 256-key chunk on every row, up to the ninth partial
    const stock: Frame = .{ .ring_rows = 2048 + 16, .window = 2048, .layer = 0, .layers = 1, .segs = &.{.{ .begin = 0, .position = 5000 }} };
    try attentionCase(t, stock, 8, false);
}

test "the block attention refuses a window its partials cannot hold and a mask past 32 block rows" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    const vb: Tensor = .{ .ptr = 0, .kind = kind(t) };
    const p = draft.Partials.at(0, 8, heads);
    var at_: draft.Attention = .{ .rows = 8, .block_rows = 8, .heads = heads, .kv_heads = kv_heads, .ring_rows = 4096, .window = 2304, .layer = 0 };
    try std.testing.expectError(error.BadShape, draft.attention(o, 0, 0, vb, null, 0, at_, p, vb));
    at_ = .{ .rows = 64, .block_rows = 64, .heads = heads, .kv_heads = kv_heads, .ring_rows = 2064, .window = 2048, .layer = 0 };
    try std.testing.expectError(error.BadShape, draft.attention(o, 0, 0, vb, 1, 0, at_, p, vb));
    const wide: Tensor = .{ .ptr = 0, .kind = .f32 };
    try std.testing.expectError(error.BadShape, draft.attention(o, 0, 0, wide, null, 0, at_, p, vb));
}
