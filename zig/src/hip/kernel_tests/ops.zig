//! Elementwise, rotation, gate, draw and routing kernels against host references, exact or within one rounding.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ops = @import("../ops/ops.zig");
const Tensor = ops.Tensor;
const Kind = ops.Kind;

fn act(t: *const Rig) Kind {
    return if (t.bf16) .bf16 else .f16;
}

/// The activation's round-to-nearest-even of `v`, as torch rounds.
fn round(t: *const Rig, v: f64) u16 {
    const f: f32 = @floatCast(v);
    if (!t.bf16) return rig.f16Bits(f);
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn tensor(b: anytype, kind: Kind) Tensor {
    return .{ .ptr = at(b), .kind = kind };
}

/// Within `ulps` units in the last place of the activation type, relative to the value's own size.
fn close(t: *const Rig, got: u16, want: f64, ulps: f64) bool {
    return closeBy(t, got, want, ulps, 0);
}

/// `close` with `slack` more: what a rotation may move when powf and the host's pow differ by an ulp.
fn closeBy(t: *const Rig, got: u16, want: f64, ulps: f64, slack: f64) bool {
    const step: f64 = if (t.bf16) 1.0 / 128.0 else 1.0 / 1024.0;
    return @abs(t.value(got) - want) <= ulps * step * @max(@abs(want), 1e-3) + slack;
}

test "casts, sums, column copies and cache writes give torch's bits" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xCA57 };
    const n = 4099;
    const hf = try gpa.alloc(f32, n);
    defer gpa.free(hf);
    for (hf) |*x| x.* = @floatCast(rng.normal() * 3.7);
    var f = try t.upload(hf);
    defer f.free();
    var a = try t.alloc(n * 2);
    defer a.free();
    try o.cast(tensor(f, .f32), tensor(a, act(t)), n);
    var back = try t.alloc(n * 4);
    defer back.free();
    try o.cast(tensor(a, act(t)), tensor(back, .f32), n);
    const hy = try gpa.alloc(u16, n);
    defer gpa.free(hy);
    for (hy) |*x| x.* = t.bits(rng.unit());
    var y = try t.upload(hy);
    defer y.free();
    var sum = try t.alloc(n * 2);
    defer sum.free();
    try o.add(tensor(a, act(t)), tensor(y, act(t)), tensor(sum, act(t)), n);
    try t.stream.synchronize();
    const ga = try rig.download(u16, a);
    defer gpa.free(ga);
    const gb = try rig.download(f32, back);
    defer gpa.free(gb);
    const gs = try rig.download(u16, sum);
    defer gpa.free(gs);
    for (hf, ga, gb, hy, gs) |x, rounded, widened, other, s| {
        try std.testing.expectEqual(round(t, x), rounded);
        try std.testing.expectEqual(@as(f32, @floatCast(t.value(rounded))), widened);
        try std.testing.expectEqual(round(t, @as(f32, @floatCast(t.value(rounded))) + @as(f32, @floatCast(t.value(other)))), s);
    }

    // columns [5, 105) of 7 rows of 300, then rows into a 3-head cache at slot 11, from the host and the device
    var cols = try t.alloc(7 * 100 * 2);
    defer cols.free();
    try o.copyCols(tensor(y, act(t)), 300, 5, at(cols), 7, 100);
    const kv_heads = 3;
    const d = 20;
    const len = 4;
    const total = 32;
    var caches: [2]@import("../memory.zig").DeviceBuffer = .{ try t.alloc(kv_heads * total * d * 2), try t.alloc(kv_heads * total * d * 2) };
    defer for (&caches) |*c| c.free();
    for (&caches) |*c| try c.fill8(0);
    try o.kvWrite(tensor(y, act(t)), at(caches[0]), len, kv_heads, d, total, 11);
    const slot = [_]i32{11};
    var dev_slot = try t.upload(&slot);
    defer dev_slot.free();
    try o.kvWriteAt(tensor(y, act(t)), at(caches[1]), len, kv_heads, d, total, at(dev_slot));
    try t.stream.synchronize();
    const gc = try rig.download(u16, cols);
    defer gpa.free(gc);
    for (0..7) |r| try std.testing.expectEqualSlices(u16, hy[r * 300 + 5 ..][0..100], gc[r * 100 ..][0..100]);
    for (caches) |c| {
        const gk = try rig.download(u16, c);
        defer gpa.free(gk);
        for (0..kv_heads) |h| for (0..total) |p| for (0..d) |j| {
            const want: u16 = if (p >= 11 and p < 11 + len) hy[((p - 11) * kv_heads + h) * d + j] else 0;
            try std.testing.expectEqual(want, gk[(h * total + p) * d + j]);
        };
    }
}

test "argmax and top-k follow torch's order: the larger value, then the lower index" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xA4C };
    const rows = 5;
    const n = 3001;
    const hl = try gpa.alloc(u16, rows * n);
    defer gpa.free(hl);
    for (hl) |*x| x.* = t.bits(rng.unit() * 8.0);
    // ties: row 1's top value twice, row 3's at both ends
    hl[1 * n + 7] = t.bits(9.0);
    hl[1 * n + 2000] = t.bits(9.0);
    hl[3 * n + 0] = t.bits(9.5);
    hl[3 * n + n - 1] = t.bits(9.5);
    var logits = try t.upload(hl);
    defer logits.free();
    var arg = try t.alloc(rows * 4);
    defer arg.free();
    try o.argmaxRows(tensor(logits, act(t)), rows, n, at(arg));
    const ks = [_]i32{ 1, 8, 0, 40, 3 };
    const stride = 40;
    var dev_ks = try t.upload(&ks);
    defer dev_ks.free();
    var ids = try t.alloc(rows * stride * 4);
    defer ids.free();
    var vals = try t.alloc(rows * stride * 2);
    defer vals.free();
    try ids.fill8(0xff);
    try o.topkRows(tensor(logits, act(t)), rows, n, at(dev_ks), stride, at(ids), at(vals));
    try t.stream.synchronize();
    const ga = try rig.download(i32, arg);
    defer gpa.free(ga);
    const gi = try rig.download(i32, ids);
    defer gpa.free(gi);
    const gv = try rig.download(u16, vals);
    defer gpa.free(gv);
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (0..rows) |r| {
        const row = hl[r * n ..][0..n];
        for (order, 0..) |*x, i| x.* = i;
        const Ctx = struct {
            t: *const Rig,
            row: []const u16,
            fn before(c: @This(), a: usize, b: usize) bool {
                const va = c.t.value(c.row[a]);
                const vb = c.t.value(c.row[b]);
                return va > vb or (va == vb and a < b);
            }
        };
        std.mem.sort(usize, order, Ctx{ .t = t, .row = row }, Ctx.before);
        try std.testing.expectEqual(@as(i32, @intCast(order[0])), ga[r]);
        // the k chosen by (value desc, id asc), each with its value; the kernel writes them in its own order
        const k: usize = @intCast(ks[r]);
        const got = gi[r * stride ..][0..k];
        for (got, gv[r * stride ..][0..k]) |id, v| try std.testing.expectEqual(row[@intCast(id)], v);
        const chosen = try gpa.alloc(i32, k);
        defer gpa.free(chosen);
        for (chosen, order[0..k]) |*c, i| c.* = @intCast(i);
        const sorted = try gpa.dupe(i32, got);
        defer gpa.free(sorted);
        std.mem.sort(i32, sorted, {}, std.sort.asc(i32));
        std.mem.sort(i32, chosen, {}, std.sort.asc(i32));
        try std.testing.expectEqualSlices(i32, chosen, sorted);
    }
    try std.testing.expectEqual(@as(i32, 7), ga[1]);
    try std.testing.expectEqual(@as(i32, 0), ga[3]);
    // k = 0 leaves the row untouched
    try std.testing.expectEqual(@as(i32, -1), gi[2 * stride]);
}

test "the MoE route sorts pairs by expert, stably, into items of at most a tile" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x2017E };
    for ([_][3]usize{ .{ 257, 512, 16 }, .{ 128, 37, 4 }, .{ 64, 1000, 64 } }) |shape| {
        const experts = shape[0];
        const pairs = shape[1];
        const tile = shape[2];
        const picks = try gpa.alloc(i32, pairs);
        defer gpa.free(picks);
        // a skewed pick so some experts fill several tiles and others none
        for (picks) |*p| p.* = @intCast(@min(experts - 1, (rng.next() >> 40) % 16 * (rng.next() >> 50) % experts));
        const counts = try gpa.alloc(usize, experts);
        defer gpa.free(counts);
        @memset(counts, 0);
        for (picks) |p| counts[@intCast(p)] += 1;
        var capacity: usize = 0;
        for (counts) |c| capacity += (c + tile - 1) / tile;
        capacity += 3;
        var dev_picks = try t.upload(picks);
        defer dev_picks.free();
        var members = try t.alloc(pairs * 4);
        defer members.free();
        var items = try t.alloc(capacity * 12);
        defer items.free();
        try o.moeRoute(at(dev_picks), pairs, experts, tile, at(members), at(items), capacity);
        try t.stream.synchronize();
        const gm = try rig.download(i32, members);
        defer gpa.free(gm);
        const gi = try rig.download(i32, items);
        defer gpa.free(gi);
        var place: usize = 0;
        var slot: usize = 0;
        for (0..experts) |e| {
            const start = place;
            for (picks, 0..) |p, i| if (p == e) {
                try std.testing.expectEqual(@as(i32, @intCast(i)), gm[place]);
                place += 1;
            };
            var left = counts[e];
            var first = start;
            while (left > 0) : (slot += 1) {
                const take = @min(left, tile);
                try std.testing.expectEqualSlices(i32, &.{ @intCast(e), @intCast(first), @intCast(take) }, gi[slot * 3 ..][0..3]);
                first += take;
                left -= take;
            }
        }
        // the slots past the plan are empty items
        for (slot..capacity) |s| try std.testing.expectEqual(@as(i32, 0), gi[s * 3 + 2]);
    }
}

fn silu(x: f64) f64 {
    return x / (1 + @exp(-x));
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

test "silu products, attention gates and the MoE activation sit within one rounding of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x51E };
    const n = 2048;
    const hg = try gpa.alloc(u16, n);
    defer gpa.free(hg);
    const hu = try gpa.alloc(u16, n);
    defer gpa.free(hu);
    for (hg, hu) |*g, *u| {
        g.* = t.bits(rng.unit() * 6);
        u.* = t.bits(rng.unit() * 2);
    }
    var g = try t.upload(hg);
    defer g.free();
    var u = try t.upload(hu);
    defer u.free();
    var out = try t.alloc(n * 2);
    defer out.free();
    try o.siluMul(tensor(g, act(t)), tensor(u, act(t)), tensor(out, act(t)), n);
    // the gate: out[r, h * d + j] = att * sigmoid(qg's second half), att as (heads, len, d)
    const len = 4;
    const heads = 8;
    const d = 64;
    const hatt = try gpa.alloc(f32, len * heads * d);
    defer gpa.free(hatt);
    for (hatt) |*x| x.* = rng.unit();
    const hqg = try gpa.alloc(u16, len * heads * 2 * d);
    defer gpa.free(hqg);
    for (hqg) |*x| x.* = t.bits(rng.unit() * 4);
    var att = try t.upload(hatt);
    defer att.free();
    var qg = try t.upload(hqg);
    defer qg.free();
    var gated = try t.alloc(len * heads * d * 2);
    defer gated.free();
    try o.attnGate(at(att), tensor(qg, act(t)), tensor(gated, act(t)), len, heads, d, false);
    // the MoE activation: silu(min(g, limit)) * clamp(u, -limit, limit), both halves fp32
    const pairs = 6;
    const width = 96;
    const hboth = try gpa.alloc(f32, pairs * 2 * width);
    defer gpa.free(hboth);
    for (hboth) |*x| x.* = rng.unit() * 10;
    var both = try t.upload(hboth);
    defer both.free();
    var moe = try t.alloc(pairs * width * 2);
    defer moe.free();
    const limit: f32 = 7;
    try o.moeAct(at(both), tensor(moe, act(t)), pairs, width, limit);
    try t.stream.synchronize();
    const go = try rig.download(u16, out);
    defer gpa.free(go);
    for (hg, hu, go) |gv, uv, got| {
        const s = t.value(round(t, silu(t.value(gv))));
        try std.testing.expect(close(t, got, s * t.value(uv), 1));
    }
    const gg = try rig.download(u16, gated);
    defer gpa.free(gg);
    for (0..len) |r| for (0..heads) |h| for (0..d) |j| {
        const want = @as(f64, hatt[(h * len + r) * d + j]) * sigmoid(t.value(hqg[(r * heads + h) * 2 * d + d + j]));
        try std.testing.expect(close(t, gg[(r * heads + h) * d + j], want, 1));
    };
    const gm = try rig.download(u16, moe);
    defer gpa.free(gm);
    for (0..pairs) |p| for (0..width) |c| {
        const gv = @min(@as(f64, hboth[p * 2 * width + c]), limit);
        const uv = std.math.clamp(@as(f64, hboth[p * 2 * width + width + c]), -limit, limit);
        try std.testing.expect(close(t, gm[p * width + c], silu(gv) * uv, 1));
    };
}

/// How far a rotated value may move when powf is one ulp from the host's pow: the angle moves by pos * 2^-23.
fn angleSlack(pos: usize, x1: f64, x2: f64) f64 {
    return @as(f64, @floatFromInt(pos)) * 0x1p-23 * (@abs(x1) + @abs(x2));
}

/// torch's rotation of one pair at position `pos`, in f32 as the kernel and torch compute the angle.
fn rotation(i: usize, half: usize, pos: usize, theta: f32) [2]f64 {
    const freq: f32 = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
    const ang: f32 = @as(f32, @floatFromInt(pos)) * freq;
    return .{ @cos(@as(f64, ang)), @sin(@as(f64, ang)) };
}

test "RoPE for a prompt, a decode row and the fused q / k norm rotate within one rounding of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0x20BE };
    const theta: f32 = 10_000_000;
    const len = 5;
    const heads = 4;
    const d = 256;
    const rotary = 64;
    const half = rotary / 2;
    const pos0 = 1000;
    const hx = try gpa.alloc(u16, len * heads * d);
    defer gpa.free(hx);
    for (hx) |*x| x.* = t.bits(rng.unit() * 2);
    var x = try t.upload(hx);
    defer x.free();
    // the prompt's rotation, written (heads, len, d)
    var out = try t.alloc(len * heads * d * 2);
    defer out.free();
    try o.ropePrefill(tensor(x, act(t)), tensor(out, act(t)), len * d, d, len, heads, d, rotary, pos0, theta);
    // a decode row: fp32 in and out, its position from the device
    const hf = try gpa.alloc(f32, 3 * d);
    defer gpa.free(hf);
    for (hf) |*v| v.* = rng.unit();
    var f = try t.upload(hf);
    defer f.free();
    var fy = try t.alloc(3 * d * 4);
    defer fy.free();
    const positions = [_]i32{ 7, 70000, 123 };
    var dev_pos = try t.upload(&positions);
    defer dev_pos.free();
    try o.ropeDecode(at(f), at(fy), 3, d, rotary, 0, theta, at(dev_pos), 1);
    // the fused path: rms norm with a weight, then the rotation at each row's device position, into a cache
    const hw = try gpa.alloc(f32, d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = 1 + rng.unit() / 4;
    var w = try t.upload(hw);
    defer w.free();
    const total = 80000;
    const qk_rows = 3;
    var cache = try t.alloc(heads * total * d * 2);
    defer cache.free();
    try o.qkRope(tensor(x, act(t)), heads * d, d, at(w), 1e-6, qk_rows, heads, d, rotary, theta, at(dev_pos), null, at(cache), total);
    try t.stream.synchronize();
    const go = try rig.download(u16, out);
    defer gpa.free(go);
    for (0..len) |r| for (0..heads) |h| for (0..d) |j| {
        const base = (r * heads + h) * d;
        var want: f64 = t.value(hx[base + j]);
        var slack: f64 = 0;
        if (j < rotary) {
            const i = if (j < half) j else j - half;
            const cs = rotation(i, half, pos0 + r, theta);
            const x1 = t.value(hx[base + i]);
            const x2 = t.value(hx[base + i + half]);
            want = if (j < half) x1 * cs[0] - x2 * cs[1] else x1 * cs[1] + x2 * cs[0];
            slack = angleSlack(pos0 + r, x1, x2);
        }
        try std.testing.expect(closeBy(t, go[(h * len + r) * d + j], want, 2, slack));
    };
    const gf = try rig.download(f32, fy);
    defer gpa.free(gf);
    for (0..3) |r| for (0..d) |j| {
        const row = hf[r * d ..][0..d];
        var want: f64 = row[j];
        if (j < rotary) {
            const i = if (j < half) j else j - half;
            const cs = rotation(i, half, @intCast(positions[r]), theta);
            want = if (j < half) row[i] * cs[0] - row[i + half] * cs[1] else row[i] * cs[1] + row[i + half] * cs[0];
        }
        const i = if (j < half) j else if (j < rotary) j - half else j;
        const slack = if (j < rotary) angleSlack(@intCast(positions[r]), row[i], row[@min(i + half, d - 1)]) else 0;
        try std.testing.expect(@abs(gf[r * d + j] - want) < 1e-6 + slack);
    };
    const gc = try gpa.alloc(u16, d);
    defer gpa.free(gc);
    for (0..qk_rows) |r| for (0..heads) |h| {
        const p: usize = @intCast(positions[r]);
        try cache.download(((h * total + p) * d) * 2, std.mem.sliceAsBytes(gc));
        const src = hx[(r * heads + h) * d ..][0..d];
        var ss: f64 = 0;
        for (src) |v| ss += t.value(v) * t.value(v);
        const inv = 1 / @sqrt(ss / d + 1e-6);
        for (0..d) |j| {
            const normed = struct {
                fn of(tt: *const Rig, s: []const u16, weight: []const f32, k: usize, scale: f64) f64 {
                    return tt.value(round(tt, tt.value(s[k]) * scale * weight[k]));
                }
            }.of;
            var want = normed(t, src, hw, j, inv);
            var slack: f64 = 0;
            if (j < rotary) {
                const i = if (j < half) j else j - half;
                const cs = rotation(i, half, p, theta);
                const x1 = normed(t, src, hw, i, inv);
                const x2 = normed(t, src, hw, i + half, inv);
                want = if (j < half) x1 * cs[0] - x2 * cs[1] else x1 * cs[1] + x2 * cs[0];
                slack = angleSlack(p, x1, x2);
            }
            try std.testing.expect(closeBy(t, gc[j], want, 3, slack));
        }
    };
}

test "the linear attention's prompt conv and DeltaNet gates sit within 1e-5 of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const o = t.ops(&t.on);
    var rng: rig.Rng = .{ .state = 0xC0417 };
    const len = 37;
    const channels = 300;
    const kernel = 4;
    const kept = kernel - 1;
    const hx = try gpa.alloc(u16, len * channels);
    defer gpa.free(hx);
    for (hx) |*x| x.* = t.bits(rng.unit() * 2);
    const hw = try gpa.alloc(f32, channels * kernel);
    defer gpa.free(hw);
    for (hw) |*x| x.* = rng.unit() / 2;
    const hs = try gpa.alloc(f32, kept * channels);
    defer gpa.free(hs);
    for (hs) |*x| x.* = rng.unit();
    var x = try t.upload(hx);
    defer x.free();
    var w = try t.upload(hw);
    defer w.free();
    var state = try t.upload(hs);
    defer state.free();
    var out = try t.alloc(len * channels * 4);
    defer out.free();
    var next = try t.alloc(kept * channels * 4);
    defer next.free();
    try o.convPrefill(tensor(x, act(t)), at(w), at(state), at(out), at(next), len, channels, kernel);
    const count = 9;
    const heads = 32;
    const ha = try gpa.alloc(u16, count * heads);
    defer gpa.free(ha);
    const hb = try gpa.alloc(u16, count * heads);
    defer gpa.free(hb);
    for (ha, hb) |*a, *b| {
        a.* = t.bits(rng.unit() * 6);
        b.* = t.bits(rng.unit() * 4);
    }
    const hlog = try gpa.alloc(f32, 2 * heads);
    defer gpa.free(hlog);
    for (hlog) |*v| v.* = rng.unit();
    var a = try t.upload(ha);
    defer a.free();
    var b = try t.upload(hb);
    defer b.free();
    var logs = try t.upload(hlog);
    defer logs.free();
    var gate = try t.alloc(count * heads * 4);
    defer gate.free();
    var beta = try t.alloc(count * heads * 4);
    defer beta.free();
    try o.gdnGatePrefill(tensor(a, act(t)), tensor(b, act(t)), at(logs), at(logs) + heads * 4, at(gate), at(beta), count, heads);
    try t.stream.synchronize();
    const go = try rig.download(f32, out);
    defer gpa.free(go);
    const gn = try rig.download(f32, next);
    defer gpa.free(gn);
    for (0..channels) |c| {
        const window = struct {
            fn at_(tt: *const Rig, xs: []const u16, ss: []const f32, ch: usize, n_ch: usize, k: usize, kp: usize) f64 {
                return if (k < kp) ss[k * n_ch + ch] else tt.value(xs[(k - kp) * n_ch + ch]);
            }
        }.at_;
        for (0..len) |r| {
            var acc: f64 = 0;
            for (0..kernel) |tap| acc += window(t, hx, hs, c, channels, r + tap, kept) * hw[c * kernel + tap];
            try std.testing.expect(@abs(go[r * channels + c] - silu(acc)) < 1e-5);
        }
        for (0..kept) |k| try std.testing.expectEqual(@as(f32, @floatCast(window(t, hx, hs, c, channels, len + k, kept))), gn[k * channels + c]);
    }
    const gg = try rig.download(f32, gate);
    defer gpa.free(gg);
    const gb = try rig.download(f32, beta);
    defer gpa.free(gb);
    for (0..count * heads) |i| {
        const h = i % heads;
        const xv = t.value(ha[i]) + hlog[heads + h];
        const sp = if (xv > 20) xv else @log(1 + @exp(xv));
        const want_gate = @exp(-@exp(@as(f64, hlog[h])) * sp);
        try std.testing.expect(@abs(gg[i] - want_gate) <= 1e-5 * @max(want_gate, 1e-3));
        try std.testing.expect(@abs(gb[i] - sigmoid(t.value(hb[i]))) < 1e-6);
    }
}
