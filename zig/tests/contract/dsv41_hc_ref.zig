//! Float64 model of DeepSeek V4.1's hyper-connection boundary for the contract: the mixes, the collapse, the expand.

//! model.py hc_mixes 951-957, hc_pre 959-963, hc_post 964-969, kernel.py hc_split_sinkhorn 407-462.

//! The mix reads the stream x [rows][dim] and hc_fn [24][dim] fp32; the collapse, the streams and the pre mix.

//! The expand reads the branch, the residual, post and comb.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const rows = contract.rows;
const Accum = c.Accum;

pub const mult = 4;
pub const mix_hc = 24; // (2 + hc_mult) * hc_mult: pre 0-3, post 4-7, comb 8-23 (kernel.py:430-435)
pub const tg = 256;

pub const Kind = enum { mix, pre, post };

/// Outputs a call of `kind` makes; the mix writes three buffers, pre [rows][4], post [rows][4], comb [rows][16].
pub fn count(kind: Kind, rows_n: usize, dim: usize) usize {
    return switch (kind) {
        .mix => rows_n * (2 * mult + 16),
        .pre => rows_n * dim,
        .post => rows_n * mult * dim,
    };
}

pub fn bf(h: u16) f64 {
    return @as(f64, @as(f32, @bitCast(@as(u32, @as(u32, h) << 16))));
}

pub fn toBf(v: f32) u16 {
    return c.toBf16(v);
}

/// A sum: float64 when `acc` is none, else folded to `acc` in `order` (gate 4's what-if).
fn add(acc: Accum, order: rows.Order, t: []const f64) f64 {
    return switch (acc) {
        .none => blk: {
            var s: f64 = 0;
            for (t) |v| s += v;
            break :blk s;
        },
        else => rows.sumIn(acc, order, t),
    };
}

/// One row's raw comb through Sinkhorn in kernel.py:436-461's order: row softmax, a column pass, iters-1 (row, column).

/// Every inner sum of the Sinkhorn runs in index order, with the +eps inside the divisor.
fn sinkhorn(z: []const f64, iters: u32, eps: f64, out: []f64) void {
    for (0..mult) |j| {
        var s: f64 = 0;
        const mx = @max(@max(z[4 * j], z[4 * j + 1]), @max(z[4 * j + 2], z[4 * j + 3]));
        for (0..mult) |k| {
            out[4 * j + k] = @exp(z[4 * j + k] - mx);
            s += out[4 * j + k];
        }
        for (0..mult) |k| out[4 * j + k] = out[4 * j + k] / s + eps;
    }
    var col: [mult]f64 = undefined;
    for (0..mult) |k| {
        col[k] = ((out[k] + out[4 + k]) + out[8 + k]) + out[12 + k];
        for (0..mult) |j| out[4 * j + k] = out[4 * j + k] / (col[k] + eps);
    }
    for (1..iters) |_| {
        for (0..mult) |j| {
            const s = ((out[4 * j] + out[4 * j + 1]) + out[4 * j + 2]) + out[4 * j + 3];
            for (0..mult) |k| out[4 * j + k] = out[4 * j + k] / (s + eps);
        }
        for (0..mult) |k| {
            var s: f64 = 0;
            for (0..mult) |j| s += out[4 * j + k];
            for (0..mult) |j| out[4 * j + k] = out[4 * j + k] / (s + eps);
        }
    }
}

/// One family's host inputs at one shape; `compute` fills want/mag in the order the rig gathers its outputs.
pub const In = struct {
    a: std.mem.Allocator,
    kind: Kind,
    dim: usize,
    rows_n: usize,
    iters: u32 = 20,
    eps: f32 = 1e-20,
    hc_eps: f32 = 1e-6,
    alpha: f32 = 2.0,
    x: []u16, // mix and post: the stream or branch [rows][dim]; pre: the streams [rows][4][dim]
    fnw: []f32, // hc_fn [24][dim] fp32, as the release stores it (the mix only)
    scale: []f32, // [3] (the mix only)
    base: []f32, // [24] (the mix only)
    mixf: []f32, // the pre mix [rows][4] the collapse consumes
    postf: []f32, // the post weights [rows][4] (the expand only)
    comb: []f32, // the doubly stochastic comb [rows][4][4] (the expand only)
    res: []u16, // the expand's residual [rows][4][dim]

    pub fn init(a: std.mem.Allocator, kind: Kind, dim: usize, rows_n: usize) !In {
        const stream: usize = if (kind == .pre) mult else 1;
        return .{
            .a = a,
            .kind = kind,
            .dim = dim,
            .rows_n = rows_n,
            .x = try a.alloc(u16, rows_n * stream * dim),
            .fnw = try a.alloc(f32, if (kind == .mix) mix_hc * dim else 1),
            .scale = try a.alloc(f32, if (kind == .mix) 3 else 1),
            .base = try a.alloc(f32, if (kind == .mix) mix_hc else 1),
            .mixf = try a.alloc(f32, rows_n * (if (kind == .mix) @as(usize, 1) else mult)),
            .postf = try a.alloc(f32, if (kind == .post) rows_n * mult else 1),
            .comb = try a.alloc(f32, if (kind == .post) rows_n * 16 else 1),
            .res = try a.alloc(u16, if (kind == .post) rows_n * mult * dim else 1),
        };
    }

    pub fn deinit(i: *In) void {
        i.a.free(i.x);
        i.a.free(i.fnw);
        i.a.free(i.scale);
        i.a.free(i.base);
        i.a.free(i.mixf);
        i.a.free(i.postf);
        i.a.free(i.comb);
        i.a.free(i.res);
    }

    /// Ordinary values; gate 1's plants; or gate 4's cancelling integers, exact in fp32 and lost in fp16 and bf16.
    pub fn fill(i: *In, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (mode == .stress) return i.stress();
        for (i.fnw) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.02;
        for (i.scale) |*v| v.* = 0.5 + rng.float(f32);
        for (i.base) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.5;
        for (i.mixf) |*v| v.* = 0.25 + rng.float(f32);
        for (i.postf) |*v| v.* = 0.25 + rng.float(f32);
        for (i.comb) |*v| v.* = 0.25 + rng.float(f32);
        for (i.x) |*v| v.* = toBf((rng.float(f32) * 2 - 1) * 8);
        for (i.res) |*v| v.* = toBf((rng.float(f32) * 2 - 1) * 8);
        if (mode == .nonfinite) i.plant();
    }

    /// Gate 4's inputs: small integers, so every fp32 partial is exact in every order.

    /// fp16 overflows past 65,504, and bf16's quantum of 1,024 at 2^16 drops the unit the whole call is about.

    /// The mix mirrors its halves (x[h+m] the nudge of x[m], fnw negated), so each dot is the nudges' sum, 156.

    /// A lane's running total runs past 2,048, where fp16's quantum is 2; the mixes then sit near 78.

    /// The Sinkhorn's exp reports a flushed total: exp(z - zmax) moves by e per unit, so one comb entry moves a factor.

    /// The collapse and the expand hold 2^16 while the small term is still to be added.
    fn stress(i: *In) void {
        const big: f32 = std.math.pow(f32, 2, 16);
        const h = i.dim / 2;
        @memset(i.x, toBf(0));
        @memset(i.res, toBf(0));
        switch (i.kind) {
            .mix => {
                for (0..h) |q| { // 16..23 mirrored: a lane's rising total peaks near 3,680, where fp16's quantum is 2
                    const v: f32 = @floatFromInt(16 + (q % 8));
                    i.x[q] = toBf(v);
                    i.x[h + q] = toBf(v);
                }
                for (0..mix_hc) |m| {
                    const f = i.fnw[m * i.dim ..][0..i.dim];
                    for (0..h) |q| { // the second half cancels the first term by term but for 8 unit nudges: dot = 156
                        const w: f32 = @floatFromInt(1 + ((q + m) % 7));
                        f[q] = w;
                        f[h + q] = -w + if (q < 8) @as(f32, 1) else @as(f32, 0);
                    }
                }
                for (i.scale) |*v| v.* = 1;
                for (i.base) |*v| v.* = 0;
            },
            .pre => {
                for (i.mixf) |*v| v.* = 1;
                for (0..i.rows_n) |r| for (0..i.dim) |q| {
                    const s: f32 = @floatFromInt(1 + (q % 7));
                    i.x[(r * mult + 0) * i.dim + q] = toBf(big);
                    i.x[(r * mult + 1) * i.dim + q] = toBf(s); // added while the running total is 2^16
                    i.x[(r * mult + 2) * i.dim + q] = toBf(-big);
                };
            },
            .post => {
                for (i.mixf) |*v| v.* = 1; // comb and post all 1: the branch's own term is the 0 below
                for (i.postf) |*v| v.* = 1;
                for (i.comb) |*v| v.* = 1;
                for (0..i.rows_n) |r| for (0..i.dim) |q| {
                    const s: f32 = @floatFromInt(1 + (q % 7));
                    i.res[(r * mult + 0) * i.dim + q] = toBf(big);
                    i.res[(r * mult + 1) * i.dim + q] = toBf(s); // held at 2^16, then cancelled by stream 2
                    i.res[(r * mult + 2) * i.dim + q] = toBf(-big);
                };
            },
        }
    }

    /// NaN and both infinities at known places, so the outputs' classes are proven, not assumed.
    fn plant(i: *In) void {
        const nf: f32 = std.math.nan(f32);
        const inf: f32 = std.math.inf(f32);
        switch (i.kind) {
            .mix => {
                i.x[0] = toBf(nf); // row 0: ss is NaN, so all 26 of its outputs are
                if (i.rows_n > 1) i.fnw[i.dim + 3] = nf; // mix[1]: row j=1's pre slot, and comb row 0 of every row
                if (i.rows_n > 2) i.fnw[2 * i.dim + 5] = nf; // mix[5]: one post slot, every row
                if (i.rows_n > 3) i.x[3 * i.dim + 1] = toBf(inf); // row 3: ss and inv infinite
            },
            .pre => {
                i.x[1] = toBf(inf); // +inf times its finite weight: +inf lands
                i.x[(mult + 2) * i.dim + 3] = toBf(-inf); // and -inf, row 1
                if (i.rows_n > 2) i.mixf[2 * mult + 1] = nf; // a finite row's output is NaN
            },
            .post => {
                i.x[i.dim + 1] = toBf(inf); // the branch, row 1: its streams go infinite
                i.res[(mult + 2) * i.dim + 3] = toBf(-inf); // a residual term, row 1 stream 2
                if (i.rows_n > 2) i.postf[2 * mult + 1] = nf; // one post weight, row 2
            },
        }
    }

    /// Every output of the prepared call in the order the rig gathers its output buffers; float64, or gate 4's sums.
    pub fn compute(i: *In, acc: Accum, order: rows.Order, want: []f64, mag: []f64) void {
        const sx = i.a.alloc(f64, i.dim) catch return;
        defer i.a.free(sx);
        const st = i.a.alloc(f64, i.dim) catch return;
        defer i.a.free(st);
        switch (i.kind) {
            .mix => {
                const n = i.rows_n;
                for (0..n) |r| i.mixRow(r, acc, order, sx, st, want[0..], want[n * mult ..], want[2 * n * mult ..], mag[0..], mag[n * mult ..], mag[2 * n * mult ..]);
            },

            .pre => for (0..i.rows_n) |r| i.preRow(r, acc, order, want[r * i.dim ..], mag[r * i.dim ..]),
            .post => for (0..i.rows_n) |r| i.postRow(r, acc, order, want[r * mult * i.dim ..], mag[r * mult * i.dim ..]),
        }
    }

    /// hc_mixes (model.py:951-957) plus hc_split_sinkhorn (kernel.py:407-462) of row r, into the call's three blocks.
    fn mixRow(i: *In, r: usize, acc: Accum, order: rows.Order, sx: []f64, st: []f64, pre: []f64, post: []f64, comb: []f64, mpre: []f64, mpost: []f64, mcomb: []f64) void {
        const x = i.x[r * i.dim ..][0..i.dim];
        for (sx, x) |*o, h| o.* = bf(h);
        // model.py:955: the statistic is the SUM OF SQUARES (the kernel's own ss), folded like every other sum
        for (st, sx) |*o, v| o.* = v * v;
        const inv = 1 / @sqrt(add(acc, order, st) / @as(f64, @floatFromInt(i.dim)) + @as(f64, i.eps));
        const u: f64 = 0x1p-24;
        var z: [mix_hc]f64 = undefined;
        var dz: [mix_hc]f64 = undefined;
        for (0..mix_hc) |m| {
            const f = i.fnw[m * i.dim ..][0..i.dim];
            var abs: f64 = 0;
            for (0..i.dim) |q| {
                const t = @as(f64, f[q]) * sx[q];
                st[q] = t;
                abs += @abs(t);
            }
            const mix = add(acc, order, st) * inv;
            const sc: f64 = @as(f64, i.scale[m / 8]);
            z[m] = mix * sc + @as(f64, i.base[m]);
            dz[m] = abs * @abs(inv) * sc + @abs(z[m]) * 2 * u;
        }
        for (0..mult) |j| {
            const s = 1 / (1 + @exp(-z[j]));
            pre[r * mult + j] = s + @as(f64, i.hc_eps);
            mpre[r * mult + j] = dz[j] + (@abs(z[j]) + 4) * @abs(pre[r * mult + j]) * u;
            const p = @as(f64, i.alpha) / (1 + @exp(-z[mult + j]));
            post[r * mult + j] = p;
            mpost[r * mult + j] = dz[mult + j] + (@abs(z[mult + j]) + 3) * @abs(p) * u;
        }
        sinkhorn(z[2 * mult ..], i.iters, @as(f64, i.hc_eps), comb[r * 16 ..][0..16]);
        const steps: f64 = @as(f64, @floatFromInt(i.iters)) * 6 + 8;
        for (0..16) |q| mcomb[r * 16 + q] = dz[2 * mult + q] * (@as(f64, @floatFromInt(i.iters)) * 4 + 6) + @abs(comb[r * 16 + q]) * (steps + @abs(z[2 * mult + q])) * u;
    }

    /// hc_pre (model.py:959-963) of row r: the streams added in order 0, 1, 2, 3 (O4), one round to bf16.
    fn preRow(i: *In, r: usize, acc: Accum, order: rows.Order, want: []f64, mag: []f64) void {
        var t: [mult]f64 = undefined;
        for (0..i.dim) |q| {
            var abs: f64 = 0;
            for (0..mult) |j| {
                t[j] = @as(f64, i.mixf[r * mult + j]) * bf(i.x[(r * mult + j) * i.dim + q]);
                abs += @abs(t[j]);
            }
            want[q] = add(acc, order, &t);
            mag[q] = abs;
        }
    }

    /// hc_post of row r in O5's order: stream k sums comb[j][k]*r_j for j = 0..3, adds post[k]*branch last, then bf16.
    fn postRow(i: *In, r: usize, acc: Accum, order: rows.Order, want: []f64, mag: []f64) void {
        var t: [5]f64 = undefined;
        for (0..mult) |k| for (0..i.dim) |q| {
            const o = (k * i.dim) + q;
            for (0..mult) |j| t[j] = @as(f64, i.comb[r * 16 + 4 * j + k]) * bf(i.res[(r * mult + j) * i.dim + q]);
            t[4] = @as(f64, i.postf[r * mult + k]) * bf(i.x[r * i.dim + q]);
            want[o] = add(acc, order, &t);
            var abs: f64 = 0;
            for (&t) |v| abs += @abs(v);
            mag[o] = abs;
        };
    }
};
