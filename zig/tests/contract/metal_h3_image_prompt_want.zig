//! The prompt cases' float64 outputs and ranges, gate 4's simulated sums, and attention's planted inputs.
const std = @import("std");
const contract = @import("contract");
const ref = @import("metal_h3_image_prompt_ref.zig");
const case = @import("metal_h3_image_prompt_case.zig");

const c = contract.spec;
const ar = contract.attention_range;
const threads = contract.reference_threads;
const Case = case.Case;
const Range = ref.Range;
const val = ref.val;

/// hp_attention's score scale as the launcher passes it: fp32 1 / sqrt(128).
pub const scale: f32 = 1.0 / @sqrt(128.0);

fn sized(t: *Case, n: usize) !void {
    t.lo = try t.alloc(f64, n);
    t.hi = try t.alloc(f64, n);
    t.allow = try t.alloc(f64, n);
}

/// Output i: the float64 value at the kernel's bf16 points and its range (a non-finite value is checked by class).
fn put(t: *Case, w: []f64, mag: []f64, i: usize, point: f64, r: Range) void {
    const finite = std.math.isFinite(point);
    w[i] = point;
    mag[i] = if (finite) @abs(point) else 0;
    t.lo[i] = r.lo;
    t.hi[i] = r.hi;
    t.allow.?[i] = if (finite) @max(@abs(point - r.lo), @abs(r.hi - point)) else 0;
}

fn squares(x: []const u16, s: ref.Sum, terms: []f64) f64 {
    for (terms, x) |*o, h| o.* = val(h) * val(h);
    return s.of(terms);
}

/// Where a head's values sit: the q heads then the key heads, in their buffers and in the outputs.
const HeadAt = struct { buf: usize, at: usize, out: usize };

fn headAt(t: *Case, r: usize, h: usize) HeadAt {
    const H = t.shape.n;
    const kv = case.kvHeads(t.form, H);
    if (t.form.fused) {
        const at = r * 3 * H * 128 + (h % H) * 384 + (if (h < H) @as(usize, 0) else 128);
        return .{ .buf = 0, .at = at, .out = at };
    }
    if (h < H) return .{ .buf = 0, .at = (r * H + h) * 128, .out = (r * H + h) * 128 };
    const at = (r * kv + h - H) * 128;
    return .{ .buf = 1, .at = at, .out = t.shape.rows * H * 128 + at };
}

pub fn reference(t: *Case, w: []f64, mag: []f64) !void {
    const s = t.shape;
    const f = t.form;
    t.allow = null;
    t.lo = &.{};
    t.hi = &.{};
    switch (f.kind) {
        .gemm => {
            const g: ref.Gemm = .{ .x = t.in[0], .w = t.in[1], .b = if (f.bias) t.in[2] else null, .M = s.rows, .N = s.n, .K = s.k };
            try g.run(t.a, null, w, mag);
        },
        .norm => {
            try sized(t, w.len);
            const W = s.k;
            const terms = try t.alloc(f64, W);
            for (0..s.rows) |r| {
                const x = t.in[0][r * W ..][0..W];
                const inv = ref.inverse(squares(x, .{}, terms), @floatFromInt(W), f.eps, t.spec.depth(s), t.stress);
                for (x, t.in[1], 0..) |xh, wh, i| put(t, w, mag, r * W + i, ref.normed(val(xh), inv.r, val(wh), f.mode), ref.normedRange(val(xh), inv.range, val(wh), f.mode));
            }
        },
        .heads => try heads(t, null, w, mag),
        .attention => try attention(t, w, mag),
        .glu => {
            try sized(t, w.len);
            const W = s.n;
            const gr = if (f.fused) 2 * W else W;
            const vs = if (f.fused) t.in[0] else t.in[1];
            const vo: usize = if (f.fused) W else 0;
            for (0..s.rows) |r| for (0..W) |i| {
                const gv = val(t.in[0][r * gr + i]);
                const vv = val(vs[r * gr + vo + i]);
                put(t, w, mag, r * W + i, ref.glu(gv, vv, f.mode), ref.gluRange(gv, vv, f.mode));
            };
        },
        .add => for (w, mag, t.in[0], t.in[1]) |*o, *m, x, y| {
            o.* = val(x) + val(y);
            m.* = @abs(val(x)) + @abs(val(y));
        },
        .scatter => {
            for (w, mag, t.in[0]) |*o, *m, x| {
                o.* = val(x);
                m.* = @abs(val(x));
            }
            const W = s.n;
            for (t.idx, 0..) |row, i| for (0..W) |e| {
                const at = @as(usize, row) * W + e;
                const y = val(t.in[1][i * W + e]);
                w[at] += y;
                mag[at] += @abs(y);
            };
        },
    }
}

/// hp_heads in float64 (`sim`: gate 4's sums, points alone): each head normed, then turned when the form ropes.
fn heads(t: *Case, sim: ?ref.Sum, w: []f64, mag: []f64) !void {
    const s = t.shape;
    const f = t.form;
    const H = s.n;
    const kv = case.kvHeads(f, H);
    if (sim == null) try sized(t, w.len);
    var terms: [128]f64 = undefined;
    var point: [128]f64 = undefined;
    var range: [128]Range = undefined;
    for (0..s.rows) |r| {
        for (0..H + kv) |h| {
            const p = headAt(t, r, h);
            const x = t.in[p.buf][p.at..][0..128];
            const wt = t.in[if (h < H) 2 else 3];
            const inv = ref.inverse(squares(x, sim orelse .{}, &terms), 128, f.eps, t.spec.depth(s), t.stress);
            for (0..128) |e| {
                point[e] = ref.normed(val(x[e]), inv.r, val(wt[e]), f.mode);
                if (sim == null) range[e] = ref.normedRange(val(x[e]), inv.range, val(wt[e]), f.mode);
            }
            for (0..128) |e| {
                var o = point[e];
                var or_ = range[e];
                if (!f.fused) {
                    // rotate-half: channel e pairs with e + 64 negated below the half, e - 64 above
                    const q = if (e < 64) e + 64 else e - 64;
                    const sign: f64 = if (e < 64) -1 else 1;
                    const at = r * 64 + e % 64;
                    o = ref.turned(point[e], sign * point[q], t.tab[0][at], t.tab[1][at]);
                    if (sim == null) or_ = ref.turnedRange(range[e], if (e < 64) range[q].neg() else range[q], t.tab[0][at], t.tab[1][at]);
                }
                if (sim == null) put(t, w, mag, p.out + e, o, or_) else w[p.out + e] = o;
            }
        }
        if (!f.fused) continue;
        // the refiner's v heads ride in the same rows, untouched
        for (0..H) |h| for (0..128) |e| {
            const at = r * 3 * H * 128 + h * 384 + 256 + e;
            const v = val(t.in[0][at]);
            if (sim == null) put(t, w, mag, at, v, Range.of(v)) else w[at] = v;
        };
    }
}

/// Query `h`'s row `r` and key head `kv`'s rows, in the case's buffers.
fn qAt(t: *Case, r: usize, h: usize) usize {
    const H = t.shape.rows;
    return if (t.form.fused) r * 3 * H * 128 + h * 384 else (r * H + h) * 128;
}

fn kvAt(t: *Case, j: usize, kvh: usize, v: bool) struct { buf: usize, at: usize } {
    const H = t.shape.rows;
    if (t.form.fused) return .{ .buf = 0, .at = j * 3 * H * 128 + kvh * 384 + (if (v) @as(usize, 256) else 128) };
    return .{ .buf = if (v) 2 else 1, .at = (j * case.kvHeads(t.form, H) + kvh) * 128 };
}

const AttnJob = struct {
    t: *Case,
    from: usize,
    to: usize,
    w: []f64,
    mag: []f64,
    scratch: []f64,

    pub fn run(j: *AttnJob) void {
        const t = j.t;
        const H = t.shape.rows;
        const T = t.shape.k;
        const group = H / case.kvHeads(t.form, H);
        const kf = j.scratch[0 .. T * 128];
        const vf = j.scratch[T * 128 .. 2 * T * 128];
        const sc = j.scratch[2 * T * 128 ..][0..T];
        const wt = j.scratch[2 * T * 128 + T ..][0..T];
        const tau = j.scratch[2 * T * 128 + 2 * T ..][0..T];
        var q: [128]f64 = undefined;
        var point: [128]f64 = undefined;
        var half: [128]f64 = undefined;
        var have: usize = std.math.maxInt(usize);
        for (j.from..j.to) |h| {
            if (h / group != have) {
                have = h / group;
                for (0..T) |key| for ([_]bool{ false, true }, [_][]f64{ kf, vf }) |isv, dst| {
                    const p = kvAt(t, key, have, isv);
                    for (dst[key * 128 ..][0..128], t.in[p.buf][p.at..][0..128]) |*o, x| o.* = val(x);
                };
            }
            for (0..T) |r| {
                for (&q, t.in[0][qAt(t, r, h)..][0..128]) |*o, x| o.* = val(x);
                const end = if (t.form.causal) r + 1 else T;
                const f = ref.form(end, ref.rises(&q, kf, end, scale, sc));
                ar.head(&q, kf, vf, end, 128, scale, f, wt, tau, &point, &half);
                for (0..128) |e| {
                    const i = (r * H + h) * 128 + e;
                    j.w[i] = point[e];
                    j.mag[i] = if (std.math.isFinite(point[e])) @abs(point[e]) else 0;
                    t.allow.?[i] = ar.allowBf16(point[e], half[e], 0, 0);
                    const ends = ar.intervalBf16(point[e], half[e], 0, 0);
                    t.lo[i] = ends[0];
                    t.hi[i] = ends[1];
                }
            }
        }
    }
};

/// hp_attention within attention_range's float64 range, heads over 8 threads; gate 4's equal scores exact.
fn attention(t: *Case, w: []f64, mag: []f64) !void {
    const H = t.shape.rows;
    const T = t.shape.k;
    try sized(t, w.len);
    if (t.stress) {
        t.lo = &.{};
        t.hi = &.{};
        for (0..T) |r| {
            const end = if (t.form.causal) r + 1 else T;
            var sum: f64 = 0;
            for (0..end) |j| sum += val(t.in[kvAt(t, j, 0, true).buf][kvAt(t, j, 0, true).at]);
            const o = sum / @as(f64, @floatFromInt(end));
            for (w[r * H * 128 ..][0 .. H * 128], mag[r * H * 128 ..][0 .. H * 128], t.allow.?[r * H * 128 ..][0 .. H * 128]) |*x, *m, *l| {
                x.* = o;
                m.* = @abs(o);
                l.* = c.boundExact(t.spec, o);
            }
        }
        return;
    }
    const n = threads.count(threads.maximum, H);
    const scratch = try t.alloc(f64, n * (2 * T * 128 + 3 * T));
    var jobs: [threads.maximum]AttnJob = undefined;
    for (jobs[0..n], 0..) |*j, i| j.* = .{ .t = t, .from = H * i / n, .to = H * (i + 1) / n, .w = w, .mag = mag, .scratch = scratch[i * (2 * T * 128 + 3 * T) ..][0 .. 2 * T * 128 + 3 * T] };
    threads.run(AttnJob, jobs[0..n], .normal);
}

pub fn simulate(t: *Case, s: ref.Sum, out: []f64) !void {
    const sh = t.shape;
    const f = t.form;
    switch (f.kind) {
        .gemm => {
            const g: ref.Gemm = .{ .x = t.in[0], .w = t.in[1], .b = if (f.bias) t.in[2] else null, .M = sh.rows, .N = sh.n, .K = sh.k };
            try g.run(t.a, s, out, out);
        },
        .norm => {
            const W = sh.k;
            const terms = try t.alloc(f64, W);
            for (0..sh.rows) |r| {
                const x = t.in[0][r * W ..][0..W];
                const r_ = ref.inverse(squares(x, s, terms), @floatFromInt(W), f.eps, 0, true).r;
                for (x, t.in[1], 0..) |xh, wh, i| out[r * W + i] = ref.normed(val(xh), r_, val(wh), f.mode);
            }
        },
        .heads => try heads(t, s, out, out),
        .attention => {
            // equal scores: each row is its keys' values summed, over its keys' count summed the same way
            if (!t.stress) return error.AttentionSumsNeedEqualScores;
            const H = sh.rows;
            const T = sh.k;
            const terms = try t.alloc(f64, 2 * T);
            for (terms[0..T], 0..) |*v, j| v.* = val(t.in[kvAt(t, j, 0, true).buf][kvAt(t, j, 0, true).at]);
            @memset(terms[T..], 1);
            for (0..T) |r| {
                const end = if (f.causal) r + 1 else T;
                @memset(out[r * H * 128 ..][0 .. H * 128], s.of(terms[0..end]) / s.of(terms[T..][0..end]));
            }
        },
        .glu, .add, .scatter => return error.NoSums,
    }
}

fn setKv(t: *Case, j: usize, kvh: usize, isv: bool, ch: usize, v: u16) void {
    const p = kvAt(t, j, kvh, isv);
    t.in[p.buf][p.at + ch] = v;
}

/// Gate 4: q = k = 0, weights 1; values 6,144-8,128 by 64, then negated: prefix sums near 2^22 cancel to 4.
pub fn stressAttention(t: *Case, T: usize, kv: usize, rng: std.Random) !void {
    for (t.in[0..3]) |x| @memset(x, 0);
    // each 32nd key's chain passes fp16's range; a few keep 127 so 1 - 127 leaves the 4 ones bf16 holds
    const half = T / 2;
    const vals = try t.alloc(f32, T);
    for (vals, 0..) |*v, j| {
        const nudge = j >= half and j - half < 8 and (j - half) % 2 == 0;
        v.* = if (j < half)
            (if (j < 8 and j % 2 == 0) 127 else @floatFromInt(64 * (96 + rng.uintLessThan(u32, 32))))
        else if (j < 2 * half) @as(f32, if (nudge) 1 else 0) - vals[j - half] else 0;
    }
    for (0..T) |j| for (0..kv) |kvh| {
        const p = kvAt(t, j, kvh, true);
        @memset(t.in[p.buf][p.at..][0..128], c.toBf16(vals[j]));
    };
}

/// Gate 1's plants in q, k and v, an infinity only where its rows end on whole blocks; `last_key`: the finding's.
pub fn plantAttention(t: *Case, T: usize, H: usize, kv: usize) void {
    if (t.form.last_key) {
        // causal: row 1 ends on key 1, two keys; full: every row ends on key T - 1
        setKv(t, if (t.form.causal) 1 else T - 1, 0, true, 2, case.inf16);
        return;
    }
    if (T < 64) return;
    t.in[0][qAt(t, 5, 3 % H) + 3] = case.nan16;
    setKv(t, 60, kv - 1, false, 7, case.nan16);
    setKv(t, 50, (kv + 1) % kv, false, 1, case.inf16);
    setKv(t, 30, 0, true, 5, case.nan16);
    setKv(t, 31, 1 % kv, true, 6, case.inf16);
    setKv(t, 35, 1 % kv, true, 6, case.ninf16);
    setKv(t, 39, 2 % kv, true, 8, case.ninf16);
}
