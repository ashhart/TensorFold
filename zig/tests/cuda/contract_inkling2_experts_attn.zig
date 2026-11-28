//! prefill_attention.cu's vision_kernel<64, 8> under the contract: Inkling 2's tower attention over one picture's rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const aref = core.attention_ref;
const threads = core.contract.reference_threads;

const D = ink.vision.head_dim;

/// The P x V chain of a row: every key, a 64-key tile at a time.
fn keyChain(s: c.Shape) usize {
    return s.k + 64;
}

fn scale(wide: bool) f32 {
    return 1.0 / @sqrt(@as(f32, if (wide) 128 else D));
}

/// Shape: heads (rows), head dim (n), sequence rows W (k): 16 heads at 1, 4 and 32 tokens, two at 256 to 4,096.
fn spec(wide: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 2, .n = D, .k = 1025 }, .{ .rows = 16, .n = D, .k = 5 }, .{ .rows = 16, .n = D, .k = 17 }, .{ .rows = 16, .n = D, .k = 129 }, .{ .rows = 2, .n = D, .k = 4097 }, .{ .rows = 2, .n = D, .k = 16385 } };
    const name = if (wide) "inkling2 experts control: vision_kernel<64, 8> scaled for head dim 128" else "inkling2 vit vision_kernel<64, 8> (full attention, head dim 64)";
    return .{ .name = name, .accum = .f32, .depth = keyChain, .ops = 1, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 128 } }, .served = served, .sums = .{ .why = attn.why } };
}

/// Gate 4's values: odd integers, every other one times 4, the second half cancelling the first (fp32 exact at 16K).
fn mirrored(rng: std.Random, out: []f32) void {
    const h = out.len / 2;
    for (0..h) |i| {
        const v: f32 = @floatFromInt((1 + 2 * rng.uintLessThan(u32, 128)) * @as(u32, if (i % 2 == 1) 4 else 1));
        out[i] = v;
        out[h + i] = (if (i < 8) @as(f32, 1) else 0) - v;
    }
    if (out.len % 2 == 1) out[out.len - 1] = 0;
}

pub const Attention = struct {
    rig: rig_.Rig,
    att: *const cuda.vit.Attention,
    wide: bool = false, // control: the score scale of head dim 128
    shape: c.Shape = undefined,
    qkv: []u16 = &.{}, // [W][3 heads 64]: q heads, then k, then v
    f: []f64 = &.{}, // [3][heads][W][64]: the same in float64
    keys: []usize = &.{},
    stressed: bool = false,
    dev: u64 = 0,
    out: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Attention) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attention) void {
        const a = self.rig.gpu.gpa;
        a.free(self.qkv);
        a.free(self.f);
        a.free(self.keys);
        self.qkv = &.{};
        self.f = &.{};
        self.keys = &.{};
    }

    pub fn kernel(self: *Attention) g.Kernel {
        return .{ .spec = spec(self.wide), .ctx = self, .vt = &vt, .control = if (self.wide) .shapes else null };
    }

    const vt = g.hooks(Attention);

    fn heads(self: *const Attention) usize {
        return self.shape.rows;
    }

    fn at(self: *const Attention, row: usize, which: usize, head: usize) []u16 {
        return self.qkv[row * 3 * self.heads() * D + (which * self.heads() + head) * D ..][0..D];
    }

    /// Float64 rows of q, k or v for one head: [W][64].
    fn rows(self: *const Attention, which: usize, head: usize) []const f64 {
        const w = self.shape.k;
        return self.f[(which * self.heads() + head) * w * D ..][0 .. w * D];
    }

    pub fn prepare(self: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != D or shape.rows == 0 or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        self.free();
        self.shape = shape;
        self.stressed = fill == .stress;
        const W = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.qkv = try a.alloc(u16, W * 3 * shape.rows * D);
        for (self.qkv) |*e| e.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        self.f = try a.alloc(f64, self.qkv.len);
        for (0..W) |j| for (0..3) |which| for (0..shape.rows) |h| {
            for (self.f[((which * shape.rows + h) * W + j) * D ..][0..D], self.at(j, which, h)) |*y, x| y.* = c.fromBf16(x);
        };
        self.keys = try a.alloc(usize, W);
        for (self.keys, 0..) |*k, i| k.* = i;
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
        self.out = try self.rig.output(W * shape.rows * D * 2);
        return true;
    }

    /// Head 0: NaN in row 0's query, +inf and -inf in values, both in one column; head 1: NaN in one key.
    fn plant(self: *Attention) void {
        const L = self.shape.k;
        self.at(0, 0, 0)[3] = c.toBf16(std.math.nan(f32));
        self.at(L - 1, 2, 0)[7] = c.toBf16(std.math.inf(f32));
        self.at(L / 2, 2, 0)[9] = c.toBf16(-std.math.inf(f32));
        if (L >= 3) {
            self.at(1, 2, 0)[11] = c.toBf16(std.math.inf(f32));
            self.at(2, 2, 0)[11] = c.toBf16(-std.math.inf(f32));
        }
        if (self.heads() > 1) self.at(L / 3, 1, 1)[5] = c.toBf16(std.math.nan(f32));
    }

    /// Gate 4: every key one vector (each p is 1); each value column mirrored over the keys.
    fn stress(self: *Attention, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const L = self.shape.k;
        var key: [D]u16 = undefined;
        for (&key) |*e| e.* = attn.normal(rng);
        const col = try a.alloc(f32, L);
        defer a.free(col);
        for (0..self.heads()) |h| {
            for (0..L) |j| @memcpy(self.at(j, 1, h), &key);
            for (0..D) |e| {
                mirrored(rng, col);
                for (col, 0..) |y, j| self.at(j, 2, h)[e] = c.toBf16(y);
            }
        }
    }

    pub fn run(self: *Attention) !void {
        try self.att.full(self.rig.s, self.dev, self.out, self.shape.k, self.heads(), D, scale(self.wide));
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attention) usize {
        return self.shape.k * self.heads() * D;
    }

    /// Output rows [lo, hi) (row-major over sequence row and head) checked against attention_ref on one thread.
    const Job = struct {
        t: *const Attention,
        out: []const u8,
        lo: usize,
        hi: usize,
        gate: g.Gate,
        planted: usize = 0,
        bad: ?usize = null, // first failing output
        ratio: f64 = 0,

        pub fn run(j: *Job) void {
            var want: [D]f64 = undefined;
            var tol: [D]f64 = undefined;
            const t = j.t;
            const s: f64 = scale(false);
            for (j.lo..j.hi) |i| {
                const row = i / t.heads();
                const head = i % t.heads();
                aref.head(t.rows(0, head)[row * D ..][0..D], t.rows(1, head), t.rows(2, head), t.keys, D, s, &want, &tol);
                for (want, tol, 0..) |w, tl, e| {
                    const got = c.read(.bf16, j.out, i * D + e);
                    if (c.classOf(w) != .finite) j.planted += 1;
                    const off = c.classOf(got) != c.classOf(w) or (j.gate != .nonfinite and std.math.isFinite(w) and !(@abs(got - w) <= tl));
                    if (off and j.bad == null) {
                        j.bad = i * D + e;
                        j.ratio = @abs(got - w) / tl;
                    }
                }
                if (j.bad != null) return;
            }
        }
    };

    /// Gates 1 and 3: every output's class equals float64's, and a finite one is within attention_ref's bound.
    pub fn check(self: *Attention, gate: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const n = self.shape.k * self.heads();
        var jobs: [threads.maximum]Job = undefined;
        const parts = threads.count(threads.maximum, n);
        for (jobs[0..parts], 0..) |*j, p| j.* = .{ .t = self, .out = out, .lo = n * p / parts, .hi = n * (p + 1) / parts, .gate = gate };
        threads.run(Job, jobs[0..parts], .normal);
        var planted: usize = 0;
        for (jobs[0..parts]) |j| {
            planted += j.planted;
            if (j.bad) |b| {
                const got = c.read(.bf16, out, b);
                return try std.fmt.bufPrint(&self.msg, "output {d} (row {d}, head {d}, dim {d}) is {e:.6}: {t}, {d:.2} of attention_ref's bound", .{ b, b / D / self.heads(), b / D % self.heads(), b % D, got, c.classOf(got), j.ratio });
            }
        }
        if (gate == .nonfinite and (planted == 0 or planted == n * D)) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ planted, n * D });
        return null;
    }

    /// Float64 softmax(s q.k) v of one output row, and its terms' magnitude sum p |v| / sum p.
    fn one(self: *const Attention, i: usize, p: []f64, want: []f64, mag: []f64) void {
        const row = i / self.heads();
        const head = i % self.heads();
        const q = self.rows(0, head)[row * D ..][0..D];
        const k = self.rows(1, head);
        const v = self.rows(2, head);
        var m: f64 = -std.math.inf(f64);
        for (p, 0..) |*x, j| {
            var s: f64 = 0;
            for (0..D) |e| s += q[e] * k[j * D + e];
            x.* = s * scale(false);
            m = @max(m, x.*);
        }
        var l: f64 = 0;
        for (p) |*x| {
            x.* = @exp(x.* - m);
            l += x.*;
        }
        for (0..D) |e| {
            var s: f64 = 0;
            var abs: f64 = 0;
            for (p, 0..) |x, j| {
                s += x * v[j * D + e];
                abs += x * @abs(v[j * D + e]);
            }
            want[i * D + e] = s / l;
            mag[i * D + e] = abs / l;
        }
    }

    /// Gate 4 reads these; under its stress every key scores alike, so a head's rows share one output row.
    pub fn reference(self: *Attention, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const p = try a.alloc(f64, self.shape.k);
        defer a.free(p);
        const n = self.shape.k * self.heads();
        for (0..n) |i| {
            if (self.stressed and i >= self.heads()) {
                @memcpy(want[i * D ..][0..D], want[(i % self.heads()) * D ..][0..D]);
                @memcpy(mag[i * D ..][0..D], mag[(i % self.heads()) * D ..][0..D]);
            } else self.one(i, p, want, mag);
        }
    }

    /// Gate 4's sums in an accumulator: its keys score alike, so every p is 1 and the sums are V's terms and l's ones.
    pub fn simulate(self: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!self.stressed) return error.SumsNeedStress;
        const a = self.rig.gpu.gpa;
        const L = self.shape.k;
        const terms = try a.alloc(f64, L);
        defer a.free(terms);
        const ones = try a.alloc(f64, L);
        defer a.free(ones);
        @memset(ones, 1);
        const l = core.contract.rows.sumIn(acc, order, ones);
        for (0..self.heads()) |h| for (0..D) |e| {
            const v = self.rows(2, h);
            for (terms, 0..) |*x, j| x.* = v[j * D + e];
            out[h * D + e] = core.contract.rows.sumIn(acc, order, terms) / l;
        };
        for (self.heads()..L * self.heads()) |i| @memcpy(out[i * D ..][0..D], out[(i % self.heads()) * D ..][0..D]);
    }
};
