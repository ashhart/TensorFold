//! qk_rows' tf_qk_rows_l2 (Dory: RoPE, then each head's L2 norm, q by a scale vector, v as is) and its controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const bfr = core.bf16_round;
const ref = core.l2_ref;
const qk = cuda.qk_rows;

/// A registration; a control keeps the rounding points in `keep` (qk_l2.cuh's ROPE 1, NORM 2).
pub const Form = struct { name: []const u8, keep: ?i32 = null };

pub const forms = [_]Form{
    .{ .name = "cuda qk_rows tf_qk_rows_l2 (Dory: dim 256, rotation from inv_freq, L2 heads, q by (sqk f)^2, v as is)" },
    .{ .name = "control: tf_qk_rows_l2 without the rotated head's rounding", .keep = 2 },
    .{ .name = "control: tf_qk_rows_l2 without q's normalized rounding before its scale", .keep = 1 },
};

const dim = 256;
const group = 4;
/// cosf and sinf within 2 ulps of their value either way.
const trig = 0x1p-22;

/// The sum of squares' chain: a warp's xor tree (5), then the head's warps in order.
fn depth(_: c.Shape) usize {
    return 5 + dim / 32;
}

/// An fp32 inverse norm's relative spread from float64's: the sum's gamma halved by the root, the fp32 steps.
pub fn spread() f64 {
    return c.gamma(5 + dim / 32, 0x1p-24) / 2 + 4 * 0x1p-24;
}

const why = "an output follows its head's sum of squares as ss^-1/2 through bf16 rounding points: a bf16 sum's scale cancels in the norm, and an fp16 sum overflows to a zero output";

/// Shape: rows x query heads (n) x head dim (k); Dory's 16 query heads over 4 key heads.
pub fn spec(f: Form) c.Spec {
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = depth,
        .ops = 12,
        .mids = 4,
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = &.{ .{ .rows = 1, .n = 16, .k = dim }, .{ .rows = 300, .n = 16, .k = dim } },
        .sums = .{ .f16 = 40, .bf16 = 0, .why = why },
    };
}

const len = 600;

const Sim = struct {
    acc: c.Accum,
    order: rows.Order,
    buf: []f64,

    pub fn squares(s: Sim, v: []const f64) f64 {
        for (s.buf[0..v.len], v) |*t, x| t.* = x * x;
        return rows.sumIn(s.acc, s.order, s.buf[0..v.len]);
    }
};

fn inRange(got: u16, r: ref.Iv) bool {
    const v: f64 = bfr.value(got);
    if (std.math.isNan(r[0]) or std.math.isNan(r[1])) return std.math.isNan(v);
    if (@as(u64, @bitCast(r[0])) == @as(u64, @bitCast(r[1]))) return got == bfr.bits(@floatCast(r[0]));
    return (r[0] < v or r[0] == v) and (v < r[1] or v == r[1]);
}

pub const Qk = struct {
    rig: rig_.Rig,
    form: Form,
    k: *const qk.QkRows,
    ctl: ?cuda.Function = null,
    arena: std.heap.ArenaAllocator,
    r: usize = 0,
    h: usize = 0,
    hk: usize = 0,
    x: []f64 = &.{}, // [r, (h + 2 hk) dim]
    qs: []f64 = &.{},
    cs: []f64 = &.{}, // [r, dim]: cos and sin of the fp32 angle pos * inv
    sn: []f64 = &.{},
    dx: u64 = 0,
    sh: qk.Shape = .{},
    dpos: u64 = 0,
    dslot: u64 = 0,
    out: qk.Out = undefined,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Qk) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Qk) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.keep != null) .shapes else null };
    }

    const vt = g.hooks(Qk);

    fn heads(self: *const Qk) usize {
        return self.h + 2 * self.hk;
    }

    pub fn prepare(self: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.k != dim or shape.n % group != 0 or shape.n == 0) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.r = shape.rows;
        self.h = shape.n;
        self.hk = shape.n / group;
        const ld = self.heads() * dim;
        const xb = try a.alloc(u16, self.r * ld);
        for (xb, 0..) |*h, i| h.* = bfr.bits(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
        const qsf = try a.alloc(f32, self.h * dim);
        for (qsf) |*v| v.* = 0.5 + 30 * rng.float(f32); // (sqk f)^2
        const inv = try a.alloc(f32, dim / 2); // vLLM's fp32 inv_freq at theta 1e4
        for (inv, 0..) |*v, j| v.* = 1 / std.math.pow(f32, 1e4, @as(f32, @floatFromInt(2 * j)) / dim);
        const pos = try a.alloc(i32, self.r);
        const slot = try a.alloc(i32, self.r);
        for (pos, slot, 0..) |*p, *sl, i| { // distinct (slot, pos % len) a row; a third of them past 199,800
            p.* = @intCast((i / 2) % len + len * (i % 5) + (if (i % 3 == 0) @as(usize, 199_800) else 0));
            sl.* = @intCast(i % 2);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            const at = [_][2]usize{ .{ 0, ld / 3 }, .{ 1, ld / 2 }, .{ 2, ld - 1 }, .{ 3, 0 }, .{ 3, dim - 1 } };
            for (at, put) |p, v| if (p[0] < self.r) {
                xb[p[0] * ld + p[1]] = bfr.bits(v);
            };
        }
        if (fill == .stress) { // integers 32 to 64 of either sign: squares sum past fp16's range, exactly in fp32; no rotation
            for (xb) |*h| h.* = bfr.bits(@as(f32, @floatFromInt(32 + rng.uintLessThan(u32, 33))) * (if (rng.boolean()) @as(f32, 1) else -1));
            @memset(inv, 0);
        }
        self.x = try a.alloc(f64, xb.len);
        for (self.x, xb) |*v, h| v.* = bfr.value(h);
        self.qs = try a.alloc(f64, qsf.len);
        for (self.qs, qsf) |*v, q| v.* = q;
        self.cs = try a.alloc(f64, self.r * dim);
        self.sn = try a.alloc(f64, self.r * dim);
        for (0..self.r) |row| for (0..dim) |i| {
            const ang: f64 = @as(f32, @floatFromInt(pos[row])) * inv[i % (dim / 2)]; // the kernel's fp32 angle
            self.cs[row * dim + i] = @cos(ang);
            self.sn[row * dim + i] = @sin(ang);
        };
        self.dx = try self.rig.input(std.mem.sliceAsBytes(xb));
        self.sh = .{ .l2 = true, .q_scale = try self.rig.input(std.mem.sliceAsBytes(qsf)), .inv = try self.rig.input(std.mem.sliceAsBytes(inv)), .rot = dim };
        self.dpos = try self.rig.input(std.mem.sliceAsBytes(pos));
        self.dslot = try self.rig.input(std.mem.sliceAsBytes(slot));
        const kv = self.r * self.hk * dim * 2;
        const cache = 2 * len * self.hk * dim * 2;
        self.out = .{ .q = try self.rig.output(self.r * self.h * dim * 2), .k = try self.rig.output(kv), .v = try self.rig.output(kv), .kc = try self.rig.scratch(cache), .vc = try self.rig.scratch(cache), .len = len };
        return true;
    }

    pub fn run(self: *Qk) !void {
        const ld: i64 = @intCast(self.heads() * dim);
        const kp = self.dx + self.h * dim * 2;
        const src: qk.Src = .{ .q = self.dx, .q_ld = ld, .q_hs = dim, .k = kp, .k_ld = ld, .k_hs = dim, .v = kp + self.hk * dim * 2, .v_ld = ld, .v_hs = dim };
        try self.k.rowsVia(self.ctl, 0, self.rig.s, src, .{ .h = self.h, .hk = self.hk, .dim = dim }, self.sh, self.dpos, self.dslot, self.out, self.r);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Qk) usize {
        return self.r * self.heads() * dim;
    }

    /// The output index of head hd's first value: q [r, h, dim], then k and v [r, hk, dim].
    fn outAt(self: *const Qk, row: usize, hd: usize) usize {
        if (hd < self.h) return (row * self.h + hd) * dim;
        const base = self.r * self.h * dim;
        if (hd < self.h + self.hk) return base + (row * self.hk + hd - self.h) * dim;
        return base + self.r * self.hk * dim + (row * self.hk + hd - self.h - self.hk) * dim;
    }

    /// Every head's outputs as ranges under `st` (trig: cosf's and sinf's stray), sums of squares by `sums`.
    fn ranges(self: *Qk, sums: anytype, st: ref.Stray, tr: f64, out: []ref.Iv) !void {
        const a = self.arena.allocator();
        const small = try a.alloc(f64, dim);
        const big = try a.alloc(f64, dim);
        const pts: ref.QkPoints = .{}; // the served kernel's points: a control's dropped one leaves the ranges
        for (0..self.r) |row| for (0..self.heads()) |hd| {
            const x = self.x[(row * self.heads() + hd) * dim ..][0..dim];
            const o = out[self.outAt(row, hd)..][0..dim];
            if (hd >= self.h + self.hk) {
                for (o, x) |*e, v| e.* = ref.point(v);
                continue;
            }
            const qs: ?[]const f64 = if (hd < self.h) self.qs[hd * dim ..][0..dim] else null;
            ref.qkHead(x, self.cs[row * dim ..][0..dim], self.sn[row * dim ..][0..dim], dim, qs, pts, 1e-6, sums, st, tr, o, small, big);
        };
    }

    fn points(self: *Qk, sums: anytype, want: []f64) !void {
        const iv = try self.arena.allocator().alloc(ref.Iv, want.len);
        try self.ranges(sums, .{}, 0, iv);
        for (want, iv) |*w, v| w.* = v[0];
    }

    pub fn reference(self: *Qk, want: []f64, mag: []f64) !void {
        try self.points(ref.Exact{}, want);
        for (mag, want) |*m, v| m.* = @abs(v) / 2;
    }

    pub fn simulate(self: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try self.arena.allocator().alloc(f64, dim);
        try self.points(Sim{ .acc = acc, .order = order, .buf = buf }, out);
    }

    /// Every output in float64's range over the fp32 run (a dropped point leaves it); v as stored.
    pub fn check(self: *Qk, _: g.Gate) !?[]const u8 {
        const n = self.count();
        const a = self.arena.allocator();
        const got = try a.alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(got), try self.rig.outputBytes());
        const iv = try a.alloc(ref.Iv, n);
        try self.ranges(ref.Exact{}, .{ .spread = spread(), .u = 0x1p-24 }, trig, iv);
        for (got, iv, 0..) |h, r, i| if (!inRange(h, r))
            return try std.fmt.bufPrint(&self.msg, "out[{d}] = {e} outside float64's [{e}, {e}]", .{ i, bfr.value(h), r[0], r[1] });
        return null;
    }
};

test "the inverse's spread stays far below a bf16 ulp, and each control drops exactly one point" {
    try std.testing.expect(spread() < 0x1p-18);
    for (forms) |f| if (f.keep) |k| try std.testing.expectEqual(@as(u32, 1), @popCount(3 & ~@as(u32, @intCast(k))));
}
