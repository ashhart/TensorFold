//! Inkling 2's HIP attention inputs under the contract: tf_ink2_prep (q, rel, k, v rows) and tf_ink2_rel_table.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_hip");
const Rig = @import("contract_rig.zig").Rig;
const fill_ = @import("contract_inkling2_fill.zig");

const contract = core.contract;
const ref = contract.inkling2_attn_ref;
const c = contract.spec;
const g = contract.gates;
const D = ref.D;
const heads = fill_.heads;
const kv = fill_.kv;
const R = ref.d_rel;
const u: f64 = 0x1p-24;

fn sh(rows: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = heads * D, .k = k };
}

const prep_shapes = [_]c.Shape{ sh(16, 300000), sh(2048, 2048), sh(1, 1048576), sh(128, 1048576), sh(2048, 1048576), sh(7, 8192), sh(4, 8193), sh(2, 8194) };
const apart_shapes = [_]c.Shape{ sh(16, 300000), sh(128, 1048576), sh(9, 20000) };
const nan_rule = "a NaN in a q, k or rel head stays NaN and leaves the head's other values unclipped (fminf drops the NaN " ++
    "norm, where transformers' clamp gives NaN); an infinity makes its head 0 and itself NaN; v rows propagate";

fn depthPrep(_: c.Shape) usize {
    return 4 + 5; // a lane's four squares, then the wave's five butterfly adds
}

/// rows x 24 heads x 128 (q); k: the first stream's context, so global rows meet tau (floor 8,192) at 1M positions.
pub fn prepSpec(global: bool, apart: bool) c.Spec {
    const name = if (apart) "inkling2 hip attn tf_ink2_prep global: three streams' rows and caches in one launch" else if (global) "inkling2 hip attn tf_ink2_prep global (q, rel scaled by tau; k, v into a 4,096-slot ring)" else "inkling2 hip attn tf_ink2_prep sliding (q, rel, k, v rows into the 3,072-slot ring)";
    const why = "a head's sum of squares only scales it: a bf16 sum's error stays under the bf16 output's rounding";
    var s: c.Spec = .{ .name = name, .accum = .f32, .depth = depthPrep, .ops = 16, .out = .bf16, .served = &prep_shapes, .nan = .{ .stated = nan_rule }, .sums = .{ .f16 = 100, .bf16 = 0, .why = why } };
    if (apart) {
        s.served = &apart_shapes;
        s.replaces = "tf_ink2_prep one stream at a time, at each stream's row and cache offsets";
    }
    return s;
}

/// A prep control's fault: alpha 0 on the device, or stream 1's rows written at stream 0's cache offset.
pub const Fault = enum { none, tau_off, cross_write };

/// tf_ink2_prep as the forward calls it: raw projections in, prepared q and rel out, k and v into the rows' slots.
pub const Prep = struct {
    rig: Rig,
    k: *const K.Kernels,
    global: bool,
    apart: bool = false,
    fault: Fault = .none,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    lay: fill_.Layout = undefined,
    raw: [4][]u16 = .{ &.{}, &.{}, &.{}, &.{} }, // q [rows][24][128], k, v [rows][8][128], rel [rows][24][16]
    scales: [2][]u16 = .{ &.{}, &.{} }, // q_scale [24][128], rel_scale [24][16]
    pos: []usize = &.{},
    stream: []usize = &.{},
    want: []f64 = &.{},
    lo: []f64 = &.{},
    hi: []f64 = &.{},
    zeros: []u8 = &.{},
    dev: [5]u64 = @splat(0), // pos, seg_of, segs (the device's: a control's faulted copy), zeros' seg_of, the base
    alone: [3]u64 = @splat(0), // each stream's segment from row 0
    in: [6]u64 = @splat(0),
    out: [4]u64 = @splat(0), // q, rel, kc, vc
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Prep) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Prep) void {
        const a = self.rig.a;
        for (&self.raw) |*x| a.free(x.*);
        for (&self.scales) |*x| a.free(x.*);
        self.raw = .{ &.{}, &.{}, &.{}, &.{} };
        self.scales = .{ &.{}, &.{} };
        inline for (.{ "pos", "stream", "want", "lo", "hi", "zeros" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
    }

    pub fn kernel(self: *Prep) g.Kernel {
        var s = prepSpec(self.global, self.apart);
        switch (self.fault) {
            .none => {},
            .tau_off => s.name = "control: inkling2 hip attn tf_ink2_prep global rows without tau",
            .cross_write => s.name = "control: inkling2 hip attn tf_ink2_prep writing stream 1's k, v into stream 0's slots",
        }
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Prep);

    fn sizes(self: *const Prep) [4]usize {
        const rows = self.shape.rows;
        const cache = self.lay.streams * kv * @as(usize, self.lay.layer.ring) * D;
        return .{ rows * heads * D, rows * heads * R, cache, cache };
    }

    pub fn count(self: *const Prep) usize {
        const s = self.sizes();
        return s[0] + s[1] + s[2] + s[3];
    }

    /// Projections as N(0, 3) (most heads clip), scales in [0.5, 3); stress: integers to 256, every partial sum exact.
    pub fn prepare(self: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != heads * D or shape.rows == 0 or shape.k < shape.rows) return false;
        if (self.apart and shape.rows < 3) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.a;
        const rows = shape.rows;
        const layer: K.Layer = if (self.global) .{ .window = 0, .ring = fill_.global_ring, .chunk = 0, .tau = true } else .{ .window = 1024, .ring = K.Layer.sliding(1024, 2048).ring, .chunk = 0, .tau = false };
        const streams: usize = if (self.apart) 3 else if (rows >= 2 and rows <= 128) 2 else 1;
        self.lay = fill_.Layout.of(layer, rows, shape.k, streams);
        self.pos = try a.alloc(usize, rows);
        self.stream = try a.alloc(usize, rows);
        const pos32 = try a.alloc(i32, 2 * rows);
        defer a.free(pos32);
        for (0..rows) |r| {
            self.stream[r] = self.lay.streamOf(r);
            self.pos[r] = self.lay.posOf(r);
            pos32[r] = @intCast(self.pos[r]);
            pos32[rows + r] = @intCast(self.stream[r]);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const lens = [4]usize{ rows * heads * D, rows * kv * D, rows * kv * D, rows * heads * R };
        for (&self.raw, lens, 0..) |*x, n, w| {
            x.* = try a.alloc(u16, n);
            for (x.*, 0..) |*v, i| v.* = ref.bits(switch (fill) {
                .stress => @floatFromInt(if (i % 64 == w) @as(i32, 256) else @as(i32, @intCast(rng.uintLessThan(u32, 511))) - 255),
                else => rng.floatNorm(f64) * 3,
            });
        }
        for (&self.scales, [2]usize{ heads * D, heads * R }) |*x, n| {
            x.* = try a.alloc(u16, n);
            for (x.*) |*v| v.* = ref.bits(if (fill == .stress) 1 else 0.5 + 2.5 * rng.float(f64));
        }
        if (fill == .nonfinite) {
            const nan = c.toBf16(std.math.nan(f32));
            const inf = c.toBf16(std.math.inf(f32));
            self.raw[0][(0 * heads + 1) * D + 3] = nan;
            if (rows > 1) self.raw[1][(1 * kv + 0) * D + 5] = inf;
            if (rows > 2) self.raw[2][(2 * kv + 1) * D + 0] = inf | 0x8000;
            if (rows > 2) self.raw[2][(2 * kv + 1) * D + 1] = inf;
            if (rows > 3) self.raw[3][(3 * heads + 0) * R + 2] = nan;
        }
        var segs = self.lay.segs;
        if (self.fault == .cross_write) segs[1].off = segs[0].off;
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(pos32[0..rows]));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(pos32[rows..]));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(segs[0..streams]));
        for (self.raw, 0..) |x, i| self.in[i] = try self.rig.input(std.mem.sliceAsBytes(x));
        for (self.scales, 0..) |x, i| self.in[4 + i] = try self.rig.input(std.mem.sliceAsBytes(x));
        const sz = self.sizes();
        for (&self.out, sz) |*o, n| o.* = try self.rig.output(n * 2);
        self.zeros = try a.alloc(u8, sz[2] * 2);
        @memset(self.zeros, 0);
        if (self.apart) {
            @memset(pos32[rows..], 0);
            self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(pos32[rows..]));
            self.dev[4] = try self.rig.scratch(self.count() * 2);
            for (self.lay.segs, &self.alone) |s, *x| {
                var one = s;
                one.row0 = 0;
                x.* = try self.rig.input(std.mem.asBytes(&one));
            }
        }
        return true;
    }

    fn config(self: *const Prep) K.Prep {
        const alpha: f32 = if (self.fault == .tau_off) 0 else @floatCast(ref.alpha);
        return .{ .qk_clip = @floatCast(ref.qk_clip), .rel_clip = @floatCast(ref.rel_clip), .floor = @floatCast(fill_.floor_pos), .alpha = alpha, .eps = @floatCast(ref.eps) };
    }

    pub fn run(self: *Prep) !void {
        try self.rig.put(self.out[2], self.zeros); // slots no row writes stay 0, so a stray write shows
        try self.rig.put(self.out[3], self.zeros);
        const r: K.Rows = .{ .rows = @intCast(self.shape.rows), .heads = heads, .kv_heads = kv, .pos = self.dev[0], .seg_of = self.dev[1], .segs = self.dev[2], .tiles = 0, .n_tiles = 0, .last_pos = @intCast(self.shape.k - 1), .streams = @intCast(self.lay.streams) };
        try K.prep(self.k, self.rig.s, r, self.lay.layer, self.config(), self.in[0..4].*, self.in[4..6].*, self.out[0], self.out[1], .{ .k = self.out[2], .v = self.out[3] });
        try self.rig.s.synchronize();
    }

    /// Each stream's rows alone, at its row offsets into the inputs and outputs and its segment's cache offset.
    pub fn replaced(self: *Prep) ![]const u8 {
        const sz = self.sizes();
        const base = self.dev[4];
        const at = [4]u64{ base, base + sz[0] * 2, base + (sz[0] + sz[1]) * 2, base + (sz[0] + sz[1] + sz[2]) * 2 };
        try self.rig.put(at[2], self.zeros);
        try self.rig.put(at[3], self.zeros);
        for (self.lay.segs, self.alone) |s, one| {
            const r0: u64 = @intCast(s.row0);
            const r: K.Rows = .{ .rows = @intCast(s.n), .heads = heads, .kv_heads = kv, .pos = self.dev[0] + r0 * 4, .seg_of = self.dev[3], .segs = one, .tiles = 0, .n_tiles = 0, .last_pos = @intCast(s.p0 + s.n - 1), .streams = 1 };
            const raw = [4]u64{ self.in[0] + r0 * heads * D * 2, self.in[1] + r0 * kv * D * 2, self.in[2] + r0 * kv * D * 2, self.in[3] + r0 * heads * R * 2 };
            try K.prep(self.k, self.rig.s, r, self.lay.layer, self.config(), raw, self.in[4..6].*, at[0] + r0 * heads * D * 2, at[1] + r0 * heads * R * 2, .{ .k = at[2], .v = at[3] });
        }
        return self.rig.bytes(base, self.count() * 2);
    }

    /// One head's outputs: x (bf16 products, exact in fp32) times its clip factor, rounded; tau after a first rounding.
    fn head(self: *Prep, x: []f64, thr: f64, t: ?f64, at: usize, acc: ?c.Accum, order: g.Order) void {
        var sq: [D]f64 = undefined;
        for (sq[0..x.len], x) |*s, v| s.* = v * v;
        const ss = if (acc) |k| contract.rows.sumIn(k, order, sq[0..x.len]) else contract.rows.sumIn(.none, .sequential, sq[0..x.len]);
        const f = ref.minNum(thr / (@sqrt(ss) + 1e-6), 1); // the L2 clip's factor: fp32 sums, sqrt, add, divide, fminf
        for (x, 0..) |v, i| self.put(at + i, v * f, t, 12 * u);
    }

    /// Output i from its fp32 value v (relative error eps before rounding): rn(v), or rn(rn(v) tau) on global rows.
    fn put(self: *Prep, i: usize, v: f64, t: ?f64, eps: f64) void {
        const one = [2]f64{ ref.rn(v - @abs(v) * eps), ref.rn(v + @abs(v) * eps) }; // bf16 of each end: rounding is monotone
        if (t) |tv| {
            self.want[i] = ref.rn(ref.rn(v) * tv);
            const ends = [2]f64{ one[0] * tv, one[1] * tv }; // tau in fp32 (logf, divide, fmaxf, product, add): 8u wider
            self.lo[i] = ref.rn(ends[0] - @abs(ends[0]) * 8 * u);
            self.hi[i] = ref.rn(ends[1] + @abs(ends[1]) * 8 * u);
        } else {
            self.want[i] = ref.rn(v);
            self.lo[i] = one[0];
            self.hi[i] = one[1];
        }
    }

    /// float64 prep at the device's rounding points (want), and the range its fp32 steps allow (lo, hi).
    fn compute(self: *Prep, acc: ?c.Accum, order: g.Order) !void {
        const a = self.rig.a;
        const n = self.count();
        inline for (.{ "want", "lo", "hi" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = try a.alloc(f64, n);
            @memset(@field(self, f), 0);
        }
        const sz = self.sizes();
        const ring: usize = self.lay.layer.ring;
        var x: [D]f64 = undefined;
        for (0..self.shape.rows) |r| {
            const t: ?f64 = if (self.global) ref.tauAt(self.pos[r], fill_.floor_pos) else null;
            for (0..heads) |h| {
                for (x[0..D], 0..) |*v, i| v.* = ref.bf(self.raw[0][(r * heads + h) * D + i]) * ref.bf(self.scales[0][h * D + i]);
                self.head(x[0..D], ref.qk_clip, t, (r * heads + h) * D, acc, order);
                for (x[0..R], 0..) |*v, i| v.* = ref.bf(self.raw[3][(r * heads + h) * R + i]) * ref.bf(self.scales[1][h * R + i]);
                self.head(x[0..R], ref.rel_clip, t, sz[0] + (r * heads + h) * R, acc, order);
            }
            for (0..kv) |h| {
                const slot = ((self.stream[r] * kv + h) * ring + self.pos[r] % ring) * D;
                for (x[0..D], 0..) |*v, i| v.* = ref.bf(self.raw[1][(r * kv + h) * D + i]);
                self.head(x[0..D], ref.qk_clip, null, sz[0] + sz[1] + slot, acc, order); // k: the clip alone
                var sq: [D]f64 = undefined;
                for (&sq, x[0..D], 0..) |*s, *v, i| {
                    v.* = ref.bf(self.raw[2][(r * kv + h) * D + i]);
                    s.* = v.* * v.*;
                }
                const ss = if (acc) |k| contract.rows.sumIn(k, order, &sq) else contract.rows.sumIn(.none, .sequential, &sq);
                const inv = 1 / @sqrt(ss / D + ref.eps); // v's RMS norm: fp32 sums, then rsqrtf within 2 ulp
                for (x[0..D], 0..) |v, i| self.put(sz[0] + sz[1] + sz[2] + slot + i, v * inv, null, 16 * u);
            }
        }
    }

    pub fn reference(self: *Prep, want: []f64, mag: []f64) !void {
        try self.compute(null, .sequential);
        @memcpy(want, self.want);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Prep, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, self.lo, self.hi) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@abs(lo - w), @abs(hi - w)) + 0x1p-140 else std.math.floatMax(f64);
    }

    pub fn intervals(self: *Prep, low: []f64, high: []f64) !void {
        for (low, high, self.lo, self.hi) |*l, *h, lo, hi| {
            l.* = @min(lo, hi);
            h.* = @max(lo, hi);
        }
    }

    pub fn simulate(self: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.compute(acc, order);
        @memcpy(out, self.want);
        try self.compute(null, .sequential); // the kernel's run is then checked against float64's ranges, not the replay's
    }

    /// The stated rule: classes as float64 prep with fminf's NaN rule; at shapes, each output inside its range.
    pub fn check(self: *Prep, at: g.Gate) !?[]const u8 {
        try self.compute(null, .sequential);
        const out = try self.rig.outputBytes();
        var planted: usize = 0;
        for (self.want, self.lo, self.hi, 0..) |w, lo, hi, i| {
            const got = c.read(.bf16, out, i);
            if (c.classOf(w) != .finite) planted += 1;
            if (c.classOf(got) != c.classOf(w)) return std.fmt.bufPrint(&self.msg, "output {d} is {e} where the rule gives {e}", .{ i, got, w }) catch "a class differs";
            if (at == .shapes and std.math.isFinite(w) and !(got >= @min(lo, hi) and got <= @max(lo, hi)))
                return std.fmt.bufPrint(&self.msg, "output {d} is {e}, outside [{e}, {e}] (float64 {e})", .{ i, got, @min(lo, hi), @max(lo, hi), w }) catch "an output leaves its range";
        }
        if (at == .nonfinite and planted == 0) return "the planted values give no non-finite output: no proof";
        return null;
    }
};

const table_shapes = [_]c.Shape{ .{ .rows = 16, .n = heads * ref.extent, .k = R }, .{ .rows = 2048, .n = heads * ref.extent, .k = R }, .{ .rows = 1, .n = heads * ref.extent, .k = R }, .{ .rows = 128, .n = heads * ref.extent, .k = R } };

fn depthTable(_: c.Shape) usize {
    return R; // a product and an add a rel value, in tap order
}

pub fn tableSpec() c.Spec {
    return .{ .name = "inkling2 hip attn tf_ink2_rel_table (bf16 bias of each (row, head) at distances 0..1023)", .accum = .f32, .depth = depthTable, .out = .bf16, .served = &table_shapes };
}

/// tf_ink2_rel_table: tab[rh][d] = bf16(sum_j rel[rh][j] proj[j][d]), proj the layer's rel_logits_proj [16][1024].
pub const Table = struct {
    rig: Rig,
    k: *const K.Kernels,
    late: bool = false, // control: the device reads proj one column late
    shape: c.Shape = undefined,
    rel: []u16 = &.{},
    proj: []u16 = &.{},
    in: [3]u64 = @splat(0), // rel, proj, the device's proj
    out: u64 = 0,

    pub fn deinit(self: *Table) void {
        const a = self.rig.a;
        a.free(self.rel);
        a.free(self.proj);
        self.rig.deinit();
    }

    pub fn kernel(self: *Table) g.Kernel {
        var s = tableSpec();
        if (self.late) s.name = "control: inkling2 hip attn tf_ink2_rel_table reading rel_logits_proj one column late";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.late) .shapes else null };
    }

    const vt = g.hooks(Table);

    pub fn count(self: *const Table) usize {
        return self.shape.rows * heads * ref.extent;
    }

    /// rel as prep leaves it (clipped to 8), proj N(0, 0.5); stress: 512 x odd mirrored a step along, proj 1 or 2.
    pub fn prepare(self: *Table, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != heads * ref.extent or shape.k != R or shape.rows == 0) return false;
        const a = self.rig.a;
        a.free(self.rel);
        a.free(self.proj);
        self.rel = &.{};
        self.proj = &.{};
        self.rig.reset();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows * heads;
        self.rel = try a.alloc(u16, n * R);
        self.proj = try a.alloc(u16, R * ref.extent);
        for (self.proj, 0..) |*p, i| p.* = ref.bits(if (fill == .stress) projStress(i / ref.extent, i % ref.extent) else rng.floatNorm(f64) * 0.5);
        var x: [R]f64 = undefined;
        for (0..n) |rh| {
            for (&x, 0..) |*v, j| v.* = if (fill == .stress) relStress(rh, j) else rng.floatNorm(f64);
            const f = if (fill == .stress) 1 else ref.clipFactor(&x, ref.rel_clip);
            for (self.rel[rh * R ..][0..R], x) |*o, v| o.* = ref.bits(v * f);
        }
        if (fill == .nonfinite) {
            self.rel[0 * R + 3] = c.toBf16(std.math.nan(f32));
            self.rel[1 * R + 5] = c.toBf16(std.math.inf(f32));
            self.rel[2 * R + 0] = c.toBf16(std.math.inf(f32));
            self.rel[2 * R + 1] = c.toBf16(-std.math.inf(f32));
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.rel));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.proj));
        self.in[2] = self.in[1];
        if (self.late) {
            const t = try a.alloc(u16, self.proj.len);
            defer a.free(t);
            for (t, 0..) |*v, i| v.* = if (i % ref.extent + 1 < ref.extent) self.proj[i + 1] else 0;
            self.in[2] = try self.rig.input(std.mem.sliceAsBytes(t));
        }
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    fn projStress(j: usize, d: usize) f64 {
        const jj = if (j >= 8) (j - 8 + 1) % 8 else j;
        return @floatFromInt(1 + (jj * 7 + d) % 2);
    }

    fn relStress(rh: usize, j: usize) f64 {
        if (j < 8) return 512 * @as(f64, @floatFromInt(2 * ((rh * 5 + j * 3) % 64) + 1));
        return -relStress(rh, (j - 8 + 1) % 8) + @as(f64, if (j == 8) 512 else 0);
    }

    pub fn run(self: *Table) !void {
        const r: K.Rows = .{ .rows = @intCast(self.shape.rows), .heads = heads, .kv_heads = kv, .pos = 0, .seg_of = 0, .segs = 0, .tiles = 0, .n_tiles = 0, .last_pos = 0, .streams = 1 };
        try K.relTable(self.k, self.rig.s, r, self.in[0], self.in[2], self.out);
        try self.rig.s.synchronize();
    }

    fn terms(self: *const Table, o: usize, t: *[R]f64) void {
        const rh = o / ref.extent;
        const d = o % ref.extent;
        for (t, 0..) |*v, j| v.* = ref.bf(self.rel[rh * R + j]) * ref.bf(self.proj[j * ref.extent + d]); // exact in fp32
    }

    pub fn reference(self: *Table, want: []f64, mag: []f64) !void {
        var t: [R]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            self.terms(o, &t);
            w.* = 0;
            m.* = 0;
            for (t) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Table, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [R]f64 = undefined;
        for (out, 0..) |*v, o| {
            self.terms(o, &t);
            v.* = contract.rows.sumIn(acc, order, &t);
        }
    }
};
