//! Inkling 2's DSpark block attention under the contract: tf_ink2_dsp_attn with its merge, and the merge alone.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const Iv = ref.Iv;
const Sums = ref.Sums;
const D = 128;
const T = 8; // a block's rows
const hk = 4; // a TP=2 rank's key heads, 6 query heads each
const hq = 24;
const grp = hq / hk;
const scale: f32 = 1.0 / @sqrt(@as(f32, D)); // as the drafter passes it
const soft_why = "online softmax over 32-key tiles in fp32 (expf rescales): no input makes its sums exact in fp32";

/// A key's 128 products, then the row's keys, each tile's rescale and the chunks' merge.
fn attnDepth(s: c.Shape) usize {
    return D + s.k + s.k / 32 + 16;
}

/// T rows x 24 heads x 128 dims x the keys every row sees: a first block, a sliding window, a global prefix, 1M's edge.
pub fn attnSpec(late: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = T, .n = hq * D, .k = 8 }, .{ .rows = T, .n = hq * D, .k = 1032 }, .{ .rows = T, .n = hq * D, .k = 3000 }, .{ .rows = T, .n = hq * D, .k = 2500 } };
    const name = if (late) "inkling2 dspark control: block attention starts a key late" else "inkling2 dspark tf_ink2_dsp_attn + tf_ink2_dsp_merge (a block over keys [lo, hi])";
    return .{ .name = name, .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
}

/// The window a shape's key count stands for: [lo, hi] and the ring the cache holds its keys in.
const Window = struct { lo: usize, hi: usize, ring: usize };

fn windowOf(keys: usize) Window {
    return switch (keys) {
        1032 => .{ .lo = 68_976, .hi = 70_007, .ring = 4096 }, // a sliding layer at 70,000: the window wraps the ring
        2500 => .{ .lo = 1_046_076, .hi = 1_048_575, .ring = 8192 }, // the global layer's last chunks at 1M
        else => .{ .lo = 0, .hi = keys - 1, .ring = 4096 },
    };
}

/// One block's queries over a key and value cache (the ring as the draft slot holds it).
pub const Attend = struct {
    rig: rig_.Rig,
    k: *const dk.Kernels,
    /// control: the window starts one key late on the device (the reference keeps its first key)
    late: bool = false,
    shape: c.Shape = undefined,
    win: Window = undefined,
    q: []u16 = &.{},
    keys: []u16 = &.{}, // [ring][hk][D]
    values: []u16 = &.{},
    tol: []f64 = &.{},
    in: [3]u64 = @splat(0),
    po: u64 = 0,
    pml: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Attend) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attend) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "keys", "values" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
        a.free(self.tol);
        self.tol = &.{};
    }

    pub fn kernel(self: *Attend) g.Kernel {
        return .{ .spec = attnSpec(self.late), .ctx = self, .vt = &vt, .control = if (self.late) .shapes else null };
    }

    const vt = g.hooks(Attend);

    /// q in [-4, 4) (peaked: a lost key shows), k and v in [-1, 1); stress: 1/8 steps; non-finite: NaN q, inf k.
    pub fn prepare(self: *Attend, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != hq * D or shape.rows == 0 or shape.rows > T or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.win = windowOf(shape.k);
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.q = try a.alloc(u16, shape.rows * hq * D);
        self.keys = try a.alloc(u16, self.win.ring * hk * D);
        self.values = try a.alloc(u16, self.win.ring * hk * D);
        for (self.q) |*v| v.* = c.toBf16(if (fill == .stress) 4 * grid(rng) else (rng.float(f32) * 2 - 1) * 4);
        for ([_][]u16{ self.keys, self.values }) |x| for (x) |*v| {
            v.* = c.toBf16(if (fill == .stress) grid(rng) else rng.float(f32) * 2 - 1);
        };
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(ref.nan); // row 0, head 0
            self.keys[((self.win.lo + 2) % self.win.ring * hk + 1) * D + 5] = c.toBf16(ref.inf); // key head 1, key 3
        }
        self.in = .{ try self.rig.input(std.mem.sliceAsBytes(self.q)), try self.rig.input(std.mem.sliceAsBytes(self.keys)), try self.rig.input(std.mem.sliceAsBytes(self.values)) };
        const n = dk.chunks(self.win.lo, self.win.hi);
        self.po = try self.rig.scratch(n * shape.rows * hq * D * 4);
        self.pml = try self.rig.scratch(n * shape.rows * hq * 2 * 4);
        self.out = try self.rig.output(shape.rows * hq * D * 2);
        return true;
    }

    /// A value on a 1/8 grid in [-1, 1]: bf16-exact, so stress sums start from exact inputs.
    fn grid(rng: std.Random) f32 {
        return @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8)) / 8;
    }

    pub fn run(self: *Attend) !void {
        const w = self.win;
        try dk.attend(self.k, self.rig.s, .{ .q = self.in[0], .kc = self.in[1], .vc = self.in[2], .po = self.po, .pml = self.pml, .T = @intCast(self.shape.rows), .hq = hq, .hk = hk, .ring = @intCast(w.ring), .lo = @intCast(w.lo + @intFromBool(self.late)), .hi = @intCast(w.hi), .c0 = 0, .scale = scale }, self.out);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attend) usize {
        return self.shape.rows * hq * D;
    }

    fn at(self: *const Attend, x: []const u16, key: usize, h: usize, d: usize) f64 {
        return c.fromBf16(x[((key % self.win.ring) * hk + h) * D + d]);
    }

    /// Row r, head h's softmax(q k scale) v over [lo, hi], sums in `acc` (null: f64); `tol`: fp32 dots, exp, rescales.
    fn attend(self: *Attend, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const kv = h / grp;
        const lo = self.win.lo;
        const n = self.win.hi + 1 - lo;
        const s = try a.alloc(f64, 4 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n .. 3 * n];
        const col = s[3 * n ..];
        var terms: [D]f64 = undefined;
        var most = -std.math.inf(f64);
        const sc: f64 = scale;
        const qr = self.q[(r * hq + h) * D ..][0..D];
        for (0..n) |i| {
            var abs: f64 = 0;
            for (&terms, qr, 0..) |*x, qv, d| {
                x.* = @as(f64, c.fromBf16(qv)) * self.at(self.keys, lo + i, kv, d);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| sumIn(k, order, &terms) else sumIn(.none, .sequential, &terms)) * sc;
            ds[i] = (abs * sc * (D + 2) + @abs(s[i]) * 2) * 0x1p-24 + 0x1p-60;
            most = maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| sumIn(k, order, p) else sumIn(.none, .sequential, p);
        // fp32 sums of p and p v, a rescale a 32-key tile and one a chunk, each expf (2 ulps) of a rounded gap
        const chain = @as(f64, @floatFromInt(n + 12 * (n / 32 + 1) + 12 * dk.chunks(lo, self.win.hi) + 32)) * 0x1p-24;
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, ds, s[0..n], 0..) |*x, w, e, sv, i| {
                const v = self.at(self.values, lo + i, kv, d);
                x.* = w * v;
                spread += w * @abs(v) * (2 * e + @abs(sv - most) * 0x1p-23 + 0x1p-22 + chain);
            }
            od.* = (if (acc) |k| sumIn(k, order, col) else sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * (0x1p-8 + chain) + spread / l + 0x1p-120;
        }
    }

    /// float64 o = sum p v / sum p, p = exp(s - max), over every key in [lo, hi]; bound as attend states.
    pub fn reference(self: *Attend, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..hq) |h| {
            const o = (r * hq + h) * D;
            try self.attend(r, h, null, .sequential, want[o..][0..D], self.tol[o..][0..D]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attend, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Attend, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..hq) |h| {
            const o = (r * hq + h) * D;
            try self.attend(r, h, acc, order, out[o..][0..D], null);
            for (out[o..][0..D]) |*v| v.* = c.fromBf16(c.toBf16(@floatCast(v.*)));
        };
    }
};

/// max that keeps a NaN, as float64 arithmetic on the formula would.
fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}

/// (row, head) pairs x 128 dims x chunks: a block's 192 pairs over 2-3 chunks, a 1M context's 1,025, one chunk.
pub fn mergeSpec(short: bool) c.Spec {
    const served = &[_]c.Shape{ .{ .rows = T * hq, .n = D, .k = 2 }, .{ .rows = T * hq, .n = D, .k = 1025 }, .{ .rows = T * hq, .n = D, .k = 3 }, .{ .rows = T * hq, .n = D, .k = 1 } };
    const name = if (short) "inkling2 dspark control: the merge folds one chunk short" else "inkling2 dspark tf_ink2_dsp_merge (chunks folded in order)";
    return .{ .name = name, .accum = .f32, .depth = mergeDepth, .out = .bf16, .served = served, .sums = .{} };
}

fn mergeDepth(s: c.Shape) usize {
    return s.k + 1;
}

/// out = bf16(sum_c po_c w_c / sum_c l_c w_c), w_c = exp(m_c - max) (0 for an empty chunk, m -inf), max by fmaxf.
pub const Merge = struct {
    kern: *const dk.Kernels,
    /// control: launched over one chunk fewer
    short: bool = false,
    shape: c.Shape = undefined,
    po: []f32 = &.{},
    pml: []f32 = &.{},
    in: [2]u64 = @splat(0),
    out: u64 = 0,

    pub const sums = {};

    pub fn spec(self: *const Merge) c.Spec {
        return mergeSpec(self.short);
    }

    pub fn control(self: *const Merge) ?g.Gate {
        return if (self.short) .shapes else null;
    }

    pub fn free(self: *Merge, a: std.mem.Allocator) void {
        a.free(self.po);
        a.free(self.pml);
        self.po = &.{};
        self.pml = &.{};
    }

    /// Chunk maxima in [-4, 4) (one chunk in 7 empty), totals in [1, 64), sums within them; stress: maxima 0, integers.
    pub fn prepare(self: *Merge, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != D or shape.k == 0) return false;
        self.free(a);
        self.shape = shape;
        const nqt = shape.rows;
        const ch = shape.k;
        self.po = try a.alloc(f32, ch * nqt * D);
        self.pml = try a.alloc(f32, ch * nqt * 2);
        const half = (ch - 1) / 2;
        for (0..ch) |cc| for (0..nqt) |rh| {
            const ml = self.pml[(cc * nqt + rh) * 2 ..][0..2];
            const row = self.po[(cc * nqt + rh) * D ..][0..D];
            if (fill == .stress) {
                ml.* = .{ 0, ref.int(rng, 1, 1000) };
                for (row) |*v| v.* = ref.int(rng, 1, 1000);
                continue;
            }
            const empty = ch > 1 and cc % 7 == 3;
            ml.* = if (empty) .{ -ref.inf, 0 } else .{ rng.float(f32) * 8 - 4, 1 + rng.float(f32) * 63 };
            for (row) |*v| v.* = if (empty) 0 else (rng.float(f32) * 2 - 1) * ml[1];
        };
        if (fill == .stress) for (0..half) |cc| for (0..nqt) |rh| for (0..D) |d| {
            const mirror = half + (cc + 1) % half; // one step apart, so no replay cancels exactly
            self.po[(mirror * nqt + rh) * D + d] = -self.po[(cc * nqt + rh) * D + d] + @as(f32, if (cc == 0 and d % 5 == 1) 1 else 0);
        };
        if (fill == .nonfinite) {
            self.po[3] = ref.nan; // pair 0, chunk 0
            self.pml[1 * 2] = ref.nan; // pair 1's chunk 0 maximum: its weight NaN
            if (ch > 1) self.po[(nqt + 2) * D + 5] = ref.inf; // pair 2, chunk 1
            self.pml[3 * 2 + 1] = ref.inf; // pair 3's chunk 0 total: its every output 0
        }
        self.in = .{ try ref.input(rig, self.po), try ref.input(rig, self.pml) };
        self.out = try rig.output(nqt * D * 2);
        return true;
    }

    pub fn run(self: *Merge, rig: *rig_.Rig) !void {
        var m: cuda.Args = .{};
        m.add(self.in[0]);
        m.add(self.in[1]);
        m.add(@as(i32, @intCast(self.shape.k - @intFromBool(self.short))));
        m.add(@as(i32, @intCast(self.shape.rows)));
        m.add(self.out);
        try cuda.launch.launch(self.kern.merge, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 128 } }, rig.s, &m);
    }

    pub fn count(self: *const Merge) usize {
        return self.shape.rows * D;
    }

    pub fn solve(self: *const Merge, a: std.mem.Allocator, s: Sums, out: []Iv) !void {
        const nqt = self.shape.rows;
        const ch = self.shape.k;
        const buf = try a.alloc(Iv, 3 * ch);
        defer a.free(buf);
        const wt = buf[0..ch];
        const terms = buf[ch .. 2 * ch];
        const ls = buf[2 * ch ..];
        const scratch = try a.alloc(f64, ch);
        defer a.free(scratch);
        for (0..nqt) |rh| {
            var mx = -std.math.inf(f64);
            for (0..ch) |cc| {
                const m: f64 = self.pml[(cc * nqt + rh) * 2];
                if (!std.math.isNan(m)) mx = @max(mx, m); // fmaxf: a NaN maximum is passed over
            }
            for (wt, 0..) |*w, cc| {
                const m: f64 = self.pml[(cc * nqt + rh) * 2];
                w.* = if (m == -std.math.inf(f64)) Iv.of(0) else if (m == mx) Iv.of(1) else Iv.of(m).sub(Iv.of(mx)).exp();
            }
            for (ls, wt, 0..) |*t, w, cc| t.* = weigh(Iv.of(self.pml[(cc * nqt + rh) * 2 + 1]), w);
            const l = s.sum(ls, ch, scratch);
            for (0..D) |d| {
                for (terms, wt, 0..) |*t, w, cc| t.* = weigh(Iv.of(self.po[(cc * nqt + rh) * D + d]), w);
                out[rh * D + d] = s.sum(terms, ch, scratch).div(l).bf();
            }
        }
    }

    /// x w in fp32: exact where w is 0 or 1 (exp(0) and an empty chunk), else one rounding.
    fn weigh(x: Iv, w: Iv) Iv {
        if (w.lo == w.hi and (w.m == 0 or w.m == 1)) return Iv.of(x.m * w.m);
        return x.mul(w);
    }
};
