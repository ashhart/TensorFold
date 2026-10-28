//! MiniMax-M3.1-Flash's CUDA norms under the contract: tf_mm3_norm, tf_mm3_add_norm and tf_mm3_prep (norm and RoPE).
const std = @import("std");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const kern = mm3.kernels;
const Iv = ref.Iv;
const Sums = ref.Sums;
const Kernels = kern.Kernels;

const hidden = 6144;
const D = 128;
const eps: f32 = 1e-6;
const theta: f32 = 1e7;
const rotary = 64;

/// A thread's fma chain (width / 1024), the warp's 5 shuffles and the 32 warps' partials in order.
fn rowDepth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 5 + 32;
}

/// Gemma's RMSNorm of one row's values `x` (ranges) scaled by 1 + w: fp32 sum of squares, rsqrtf, one bf16 rounding.
fn normRow(s: Sums, x: []const Iv, w: []const u16, out: []Iv, depth: usize, scratch: []f64, sq: []Iv) void {
    for (sq, x) |*q, v| {
        const lo = @min(@abs(v.lo), @abs(v.hi));
        const hi = @max(@abs(v.lo), @abs(v.hi));
        q.* = .{ .m = v.m * v.m, .lo = if (v.lo <= 0 and v.hi >= 0) 0 else lo * lo, .hi = hi * hi };
    }
    const total = s.sum(sq, depth, scratch);
    const inv = total.div(Iv.of(@floatFromInt(x.len))).add(Iv.of(eps)).rsqrt();
    for (out, x, w) |*o, v, wv| o.* = v.mul(inv).mul(Iv.of(1).add(Iv.of(ref.bf(wv)))).bf();
}

/// Rows x hidden x hidden: one block a row, so a prompt chunk's rows are the same launch per row as a decode window's.
pub fn normSpec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = hidden, .k = hidden }, .{ .rows = 1, .n = hidden, .k = hidden }, .{ .rows = 7, .n = hidden, .k = hidden }, .{ .rows = 32, .n = hidden, .k = hidden }, .{ .rows = 512, .n = hidden, .k = hidden } };
    return .{ .name = "cuda minimax tf_mm3_norm (1 + w, rounded once)", .accum = .f32, .depth = rowDepth, .out = .bf16, .served = served, .sums = .{} };
}

/// Norm weights as the checkpoint's 1 + w offsets: mostly in [-1, 3), one in 97 ten times larger.
fn weights(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const w = try a.alloc(u16, n);
    for (w, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 4 - 1) * (if (i % 97 == 3) @as(f32, 10) else 1));
    return w;
}

/// Residual rows (bf16 values in fp32): one in 64 twenty times larger; stress: integers in [-32, 32].
fn residual(a: std.mem.Allocator, rng: std.Random, n: usize, fill: g.Fill) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x, 0..) |*v, i| v.* = if (fill == .stress) ref.int(rng, -32, 32) else c.fromBf16(c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1)));
    return x;
}

/// NaN, +Inf and -Inf at known places: a row each, row 3 holding both infinities.
fn plant(x: []f32, rows: usize, n: usize) void {
    const at = [_][2]usize{ .{ 0, n / 3 }, .{ 1, n / 2 }, .{ 2, n - 1 }, .{ 3, 0 }, .{ 3, n - 1 } };
    const put = [_]f32{ ref.nan, ref.inf, -ref.inf, ref.inf, -ref.inf };
    for (at, put) |p, v| if (p[0] < rows) {
        x[p[0] * n + p[1]] = v;
    };
}

pub const Norm = struct {
    kern: *const Kernels,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []u16 = &.{},
    in: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(_: *const Norm) c.Spec {
        return normSpec();
    }

    pub fn control(_: *const Norm) ?g.Gate {
        return null;
    }

    pub fn free(self: *Norm, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.w);
        self.x = &.{};
        self.w = &.{};
    }

    pub fn prepare(self: *Norm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != shape.k or shape.rows == 0 or shape.n == 0) return false;
        self.free(a);
        self.shape = shape;
        self.x = try residual(a, rng, shape.rows * shape.n, fill);
        self.w = try weights(a, rng, shape.n);
        if (fill == .nonfinite) plant(self.x, shape.rows, shape.n);
        self.in = .{ try ref.input(rig, self.x), try ref.input(rig, self.w) };
        self.out = try rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(self: *Norm, rig: *rig_.Rig) !void {
        try kern.norm(self.kern, rig.s, self.in[0], self.in[1], self.out, self.shape.rows, self.shape.n, eps);
    }

    pub fn count(self: *const Norm) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const Norm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        for (0..self.shape.rows) |r| {
            for (buf[0..n], self.x[r * n ..][0..n]) |*v, x| v.* = Iv.of(x);
            normRow(s, buf[0..n], self.w, out[r * n ..][0..n], rowDepth(self.shape), scratch, buf[n..]);
        }
    }
};

/// Which output an add_norm registration checks: the residual written back, or the next norm's rows.
pub const View = enum { resid, normed };

/// Rows x hidden x (ranks x slots): k 2 is the attention's and dense MLP's add, k 4 the MoE's, k 1 one rank's.
pub fn addSpec(v: View) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = hidden, .k = 2 }, .{ .rows = 64, .n = hidden, .k = 4 }, .{ .rows = 1, .n = hidden, .k = 4 }, .{ .rows = 7, .n = hidden, .k = 2 }, .{ .rows = 3, .n = hidden, .k = 1 }, .{ .rows = 512, .n = hidden, .k = 4 } };
    return .{
        .name = if (v == .resid) "cuda minimax tf_mm3_add_norm, residual (last layer: no norm)" else "cuda minimax tf_mm3_add_norm, the next norm's rows",
        .accum = .f32,
        .depth = rowDepth,
        .out = if (v == .resid) .f32 else .bf16,
        .in_place = true,
        .served = served,
        .sums = .{},
    };
}

/// R = bf16(R + X), X the ranks' partials summed (slots 2: bf16(bf16(2 routed) + bf16(shared))), then the next norm.
pub const AddNorm = struct {
    kern: *const Kernels,
    view: View,
    shape: c.Shape = undefined,
    world: usize = 0,
    slots: usize = 0,
    x: []f32 = &.{},
    parts: []f32 = &.{},
    w: []u16 = &.{},
    resid: u64 = 0,
    in: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const AddNorm) c.Spec {
        return addSpec(self.view);
    }

    pub fn control(_: *const AddNorm) ?g.Gate {
        return null;
    }

    pub fn free(self: *AddNorm, a: std.mem.Allocator) void {
        a.free(self.x);
        a.free(self.parts);
        a.free(self.w);
        self.x = &.{};
        self.parts = &.{};
        self.w = &.{};
    }

    fn scale(self: *const AddNorm) f32 {
        return if (self.slots == 2) 2.0 else 1.0; // the config's routed_scale on the MoE's add
    }

    /// Partials in [-2, 2); stress: rank 0's integers near 2^16 cancelled by rank 1's (fp16 overflows, bf16 loses).
    pub fn prepare(self: *AddNorm, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or (shape.k != 1 and shape.k != 2 and shape.k != 4)) return false;
        self.free(a);
        self.shape = shape;
        self.world = if (shape.k == 1) 1 else 2;
        self.slots = shape.k / self.world;
        const rows = shape.rows;
        const n = shape.n;
        self.x = try residual(a, rng, rows * n, fill);
        self.parts = try ref.f32s(a, rng, self.world * rows * self.slots * n, -2, 2);
        if (fill == .stress) {
            const per = rows * self.slots * n;
            for (0..per) |i| {
                const big = ref.int(rng, 40000, 120000);
                self.parts[i] = if (self.world == 2) big else ref.int(rng, -8, 8);
                if (self.world == 2) self.parts[per + i] = ref.int(rng, -8, 8) - big;
            }
        }
        self.w = try weights(a, rng, n);
        if (fill == .nonfinite) {
            plant(self.x, rows, n);
            self.parts[(rows * self.slots - 1) * n + 5] = ref.nan; // the last row's last slot, rank 0
        }
        self.in = .{ try ref.input(rig, self.parts), if (self.view == .normed) try ref.input(rig, self.w) else 0 };
        if (self.view == .resid) {
            self.resid = try rig.output(rows * n * 4);
            self.out = try rig.scratch(rows * n * 2);
        } else {
            self.resid = try rig.scratch(rows * n * 4);
            self.out = try rig.output(rows * n * 2);
        }
        return true;
    }

    pub fn run(self: *AddNorm, rig: *rig_.Rig) !void {
        try ref.restore(rig, self.resid, std.mem.sliceAsBytes(self.x)); // read and written in place: every run the same pass
        try kern.addNorm(self.kern, rig.s, self.resid, self.in[0], self.world, self.slots, self.scale(), self.in[1], self.out, self.shape.rows, self.shape.n, eps);
    }

    pub fn count(self: *const AddNorm) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn solve(self: *const AddNorm, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const n = self.shape.n;
        const rows = self.shape.rows;
        const buf = try a.alloc(Iv, 2 * n);
        defer a.free(buf);
        const scratch = try a.alloc(f64, n);
        defer a.free(scratch);
        var terms: [2]Iv = undefined;
        for (0..rows) |r| {
            const row = buf[0..n];
            for (row, 0..) |*v, i| {
                var y: [2]Iv = undefined;
                for (0..self.slots) |slot| {
                    for (terms[0..self.world], 0..) |*t, q| t.* = Iv.of(self.parts[((q * rows + r) * self.slots + slot) * n + i]);
                    y[slot] = s.sum(terms[0..self.world], self.world, scratch);
                }
                const x = if (self.slots == 2) y[0].mul(Iv.of(self.scale())).bf().add(y[1].bf()).bf() else y[0].bf();
                v.* = Iv.of(self.x[r * n + i]).add(x).bf();
            }
            if (self.view == .resid) @memcpy(out[r * n ..][0..n], row) else normRow(s, row, self.w, out[r * n ..][0..n], rowDepth(.{ .rows = 1, .n = n, .k = n }), scratch, buf[n..]);
        }
    }
};

/// The forward's prep calls: a layer's (q, k, v, the index pair), a draft block's (no index), a context absorb.
pub const Form = enum { full, draft, absorb };

const Heads = struct { hq: usize, hk: usize, hi: usize, ik: bool };

fn heads(f: Form) Heads {
    return switch (f) {
        .full => .{ .hq = 32, .hk = 2, .hi = 2, .ik = true }, // a TP=2 rank: 32 query, 2 key and 2 index heads
        .draft => .{ .hq = 32, .hk = 2, .hi = 0, .ik = false },
        .absorb => .{ .hq = 0, .hk = 2, .hi = 0, .ik = false },
    };
}

fn width(f: Form) usize {
    const h = heads(f);
    return (h.hq + h.hi + 2 * h.hk + @intFromBool(h.ik)) * D;
}

fn headDepth(_: c.Shape) usize {
    return 4 + 5; // a lane's 4 squares, the warp's 5 shuffles
}

/// The forward's calls: a prompt chunk at 0, decode windows deep in the context and the 1M window's last position.
fn prepServed(comptime f: Form) []const c.Shape {
    const w = width(f);
    return switch (f) {
        .full => &[_]c.Shape{ .{ .rows = 64, .n = w, .k = 64 }, .{ .rows = 16, .n = w, .k = 100_016 }, .{ .rows = 1, .n = w, .k = 4097 }, .{ .rows = 32, .n = w, .k = 262_144 }, .{ .rows = 1, .n = w, .k = 1_048_576 }, .{ .rows = 512, .n = w, .k = 2560 } },
        .draft => &[_]c.Shape{ .{ .rows = 7, .n = w, .k = 3007 }, .{ .rows = 8, .n = w, .k = 70_008 } },
        .absorb => &[_]c.Shape{ .{ .rows = 16, .n = w, .k = 2016 }, .{ .rows = 512, .n = w, .k = 512 } },
    };
}

/// Rows x the row's outputs (q, index q, k, v, index k heads) x the window's end position (pos + rows).
pub fn prepSpec(f: Form) c.Spec {
    const served: []const c.Shape = switch (f) {
        inline else => |x| comptime prepServed(x),
    };
    const name = switch (f) {
        .full => "cuda minimax tf_mm3_prep, a layer's q, k, v and index heads",
        .draft => "cuda minimax tf_mm3_prep, a draft block's stacked q, k, v (no index)",
        .absorb => "cuda minimax tf_mm3_prep, a drafter's absorbed keys and values",
    };
    return .{ .name = name, .accum = .f32, .depth = headDepth, .out = .bf16, .served = served, .sums = .{} };
}

/// Each head Gemma-normed (1 + w, bf16), its first 64 dims NeoX-rotated at pos + row (bf16 cos, sin and products).
pub const Prep = struct {
    kern: *const Kernels,
    form: Form,
    /// control: the device rotates every row one position late (its cache rows land where they are due)
    late: bool = false,
    shape: c.Shape = undefined,
    pos: usize = 0,
    x: [4][]u16 = .{ &.{}, &.{}, &.{}, &.{} }, // q, k, v, index (full); or one stacked q | k | v buffer in x[0]
    w: [4][]u16 = .{ &.{}, &.{}, &.{}, &.{} }, // q, k, index q, index k norms
    dev: [8]u64 = @splat(0),
    outs: [5]u64 = @splat(0), // q, index q, keys, values, index keys

    pub const sums = {};

    pub fn spec(self: *const Prep) c.Spec {
        var s = prepSpec(self.form);
        if (self.late) s.name = "control: MiniMax prep rotates its rows one position late";
        return s;
    }

    pub fn control(self: *const Prep) ?g.Gate {
        return if (self.late) .shapes else null;
    }

    pub fn free(self: *Prep, a: std.mem.Allocator) void {
        for (&self.x) |*v| {
            a.free(v.*);
            v.* = &.{};
        }
        for (&self.w) |*v| {
            a.free(v.*);
            v.* = &.{};
        }
    }

    /// A row's elements in x[0] (stacked forms) or in each projection's own buffer (full).
    fn stride(self: *const Prep, which: usize) usize {
        const h = heads(self.form);
        if (self.form != .full) return (h.hq + 2 * h.hk) * D;
        const per = [4]usize{ h.hq, h.hk, h.hk, h.hi + 1 };
        return per[which] * D;
    }

    /// The element of projection `which` (0 q, 1 k, 2 v, 3 index) at row r, head h, dim d.
    fn at(self: *const Prep, which: usize, r: usize, h: usize, d: usize) f64 {
        const hh = heads(self.form);
        if (self.form == .full) return ref.bf(self.x[which][r * self.stride(which) + h * D + d]);
        const firsts = [3]usize{ 0, hh.hq, hh.hq + hh.hk };
        const first = firsts[which];
        return ref.bf(self.x[0][r * self.stride(0) + (first + h) * D + d]);
    }

    /// Heads in [-2, 2) (one row in 9 ten times smaller); stress: integers in [-64, 64] (the squares pass 2^16).
    pub fn prepare(self: *Prep, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.n != width(self.form) or shape.rows == 0 or shape.k < shape.rows) return false;
        self.free(a);
        self.shape = shape;
        self.pos = shape.k - shape.rows;
        const rows = shape.rows;
        const nx: usize = if (self.form == .full) 4 else 1;
        for (self.x[0..nx], 0..) |*x, i| {
            x.* = try a.alloc(u16, rows * self.stride(i));
            for (x.*, 0..) |*v, e| v.* = c.toBf16(if (fill == .stress) ref.int(rng, -64, 64) else (rng.float(f32) * 4 - 2) * (if (e / D % 9 == 4) @as(f32, 0.1) else 1));
        }
        if (fill == .nonfinite) {
            self.x[0][3] = c.toBf16(ref.nan); // row 0's first head: its every output NaN
            self.x[0][self.stride(0) + D + 9] = c.toBf16(ref.inf);
            const kx: usize = if (self.form == .full) 1 else 0;
            const k0: usize = if (self.form == .full) 0 else heads(self.form).hq * D;
            self.x[kx][2 * self.stride(kx) + k0 + 70] = c.toBf16(-ref.inf); // row 2's first key head, past the rotary dims
        }
        for (&self.w) |*w| w.* = try weights(a, rng, D);
        const h = heads(self.form);
        const shift: usize = @intFromBool(self.late);
        for (self.x[0..nx], 0..) |x, i| self.dev[i] = try ref.input(rig, x);
        for (self.w, 0..) |w, i| self.dev[4 + i] = try ref.input(rig, w);
        const slot0 = (self.pos + shift) * 2; // the caches' base, so slot pos + row lands on the buffer's row
        self.outs = @splat(0); // allocated in the order the outputs gather: q, index q, keys, values, index keys
        if (h.hq > 0) self.outs[0] = try rig.output(rows * h.hq * D * 2);
        if (h.hi > 0) self.outs[1] = try rig.output(rows * h.hi * D * 2);
        self.outs[2] = try rig.output(rows * h.hk * D * 2) - slot0 * h.hk * D;
        self.outs[3] = try rig.output(rows * h.hk * D * 2) - slot0 * h.hk * D;
        if (h.ik) self.outs[4] = try rig.output(rows * D * 2) - slot0 * D;
        return true;
    }

    pub fn run(self: *Prep, rig: *rig_.Rig) !void {
        const h = heads(self.form);
        const d = self.dev;
        const rows: [4]kern.Rows = if (self.form == .full)
            .{ kern.Rows.of(d[0], self.stride(0)), kern.Rows.of(d[1], self.stride(1)), kern.Rows.of(d[2], self.stride(2)), kern.Rows.of(d[3], self.stride(3)) }
        else
            kern.stacked(d[0], self.stride(0), h.hq, h.hk, D);
        const o = self.outs;
        const full = self.form == .full;
        try kern.prep(self.kern, rig.s, .{ .xq = rows[0], .xk = rows[1], .xv = rows[2], .xi = rows[3], .qw = if (h.hq > 0) d[4] else 0, .kw = d[5], .iqw = if (full) d[6] else 0, .ikw = if (full) d[7] else 0, .q = o[0], .iq = o[1], .kc = o[2], .vc = o[3], .ik = o[4], .rows = self.shape.rows, .pos = self.pos + @intFromBool(self.late), .hq = h.hq, .hk = h.hk, .hi = h.hi, .rotary = rotary, .eps = eps, .theta = theta });
    }

    pub fn count(self: *const Prep) usize {
        return self.shape.rows * self.shape.n;
    }

    /// One head's 128 outputs: the norm, then RoPE at `position` on its first 64 dims.
    fn head(self: *const Prep, s: Sums, which: usize, r: usize, hd: usize, w: []const u16, out: []Iv, scratch: []f64) void {
        var x: [D]Iv = undefined;
        var sq: [D]Iv = undefined;
        var nrm: [D]Iv = undefined;
        for (&x, 0..) |*v, d| v.* = Iv.of(self.at(which, r, hd, d));
        for (&sq, x) |*q, v| q.* = Iv.of(v.m * v.m);
        const total = s.sum(&sq, headDepth(self.shape), scratch);
        const inv = total.div(Iv.of(D)).add(Iv.of(eps)).rsqrt();
        for (&nrm, x, w) |*o, v, wv| o.* = v.mul(inv).mul(Iv.of(1).add(Iv.of(ref.bf(wv)))).bf();
        const position: f64 = @floatFromInt(self.pos + r);
        for (out, 0..) |*o, d| {
            if (d >= rotary) {
                o.* = nrm[d];
                continue;
            }
            const i = d % (rotary / 2);
            const e = @as(f64, @floatFromInt(2 * i)) / rotary;
            const am = position * std.math.pow(f64, theta, -e);
            // the power in double rounded once to fp32 (1 at i 0), the fp32 reciprocal, the product: half an ulp each
            const rel: f64 = if (i == 0) 0 else 3 * ref.unit + 0x1p-50;
            const angle: Iv = .{ .m = am, .lo = am - am * rel, .hi = am + am * rel };
            const cs = angle.trig(false).bf();
            const sn = angle.trig(true).bf();
            const rot = if (d < rotary / 2) nrm[d + rotary / 2].neg() else nrm[d - rotary / 2];
            o.* = nrm[d].mul(cs).bf().add(rot.mul(sn).bf()).bf();
        }
    }

    pub fn solve(self: *const Prep, _: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const h = heads(self.form);
        const rows = self.shape.rows;
        var scratch: [D]f64 = undefined;
        var o: usize = 0;
        for (0..rows) |r| for (0..h.hq) |hd| {
            self.head(s, 0, r, hd, self.w[0], out[o..][0..D], &scratch);
            o += D;
        };
        for (0..rows) |r| for (0..h.hi) |hd| {
            self.head(s, 3, r, hd, self.w[2], out[o..][0..D], &scratch);
            o += D;
        };
        for (0..rows) |r| for (0..h.hk) |hd| {
            self.head(s, 1, r, hd, self.w[1], out[o..][0..D], &scratch);
            o += D;
        };
        for (0..rows) |r| for (0..h.hk) |hd| for (0..D) |d| {
            out[o] = Iv.of(self.at(2, r, hd, d)); // the values as they are
            o += 1;
        };
        if (h.ik) for (0..rows) |r| {
            self.head(s, 3, r, h.hi, self.w[3], out[o..][0..D], &scratch);
            o += D;
        };
        std.debug.assert(o == out.len);
    }
};
