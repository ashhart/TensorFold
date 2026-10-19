//! qk_rows' tf_qk_rows_once and tf_qk_rows_round (dims to 512, v's unit norm, MLX's rows) and their controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const qr = @import("qk_round_check.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bfr = core.bf16_round;
const ref = core.qk_ref;
const qk = cuda.qk_rows;

pub const Rope = enum { table, inv, proportional };

/// A registration: head dim, query heads a key head, the options; a control keeps the bits in `keep` (qk_round.cuh).
pub const Form = struct { name: []const u8, dim: usize, group: usize, rope: Rope, round: bool, v_norm: bool, v_from_k: bool = false, keep: ?i32 = null };

const row_pt = 1;
const out_pt = 2;
const vnorm_pt = 4;

/// Gemma 4 31B's global (k's columns as v) and sliding layers, the 12B's global rounded once; controls drop one each.
pub const forms = [_]Form{
    .{ .name = "cuda qk_rows tf_qk_rows_round (MLX's rows, dim 512, k's columns as v by a unit norm, 64 of 256 pairs rotated)", .dim = 512, .group = 8, .rope = .proportional, .round = true, .v_norm = true, .v_from_k = true },
    .{ .name = "cuda qk_rows tf_qk_rows_round (MLX's rows, dim 256, unit v norm, rotation from inv_freq)", .dim = 256, .group = 2, .rope = .inv, .round = true, .v_norm = true },
    .{ .name = "cuda qk_rows tf_qk_rows_once (dim 512, norm, rotation from a table, v as is)", .dim = 512, .group = 16, .rope = .table, .round = false, .v_norm = false },
    .{ .name = "control: MLX's qk rows without the normalized row's rounding", .dim = 512, .group = 8, .rope = .proportional, .round = true, .v_norm = true, .v_from_k = true, .keep = out_pt | vnorm_pt },
    .{ .name = "control: MLX's qk rows without the weighted row's rounding before RoPE", .dim = 512, .group = 8, .rope = .proportional, .round = true, .v_norm = true, .v_from_k = true, .keep = row_pt | vnorm_pt },
    .{ .name = "control: MLX's qk rows without v's unit norm", .dim = 512, .group = 8, .rope = .proportional, .round = true, .v_norm = true, .v_from_k = true, .keep = row_pt | out_pt },
};

fn depth(shape: c.Shape) usize {
    return qr.depth(shape.k);
}

const why = "an output follows its sum of squares as ss^-1/2 through bf16 rounding points: a bf16 sum's error is a few output ulps, and an fp16 sum overflows to a zero output, 51x the bound that carries the points";

/// Shape: rows x query heads (n) x head dim (k).
pub fn spec(f: Form) c.Spec {
    const served = switch (f.group) {
        8 => &[_]c.Shape{ .{ .rows = 1, .n = 32, .k = 512 }, .{ .rows = 300, .n = 32, .k = 512 } },
        2 => &[_]c.Shape{ .{ .rows = 1, .n = 32, .k = 256 }, .{ .rows = 300, .n = 32, .k = 256 } },
        else => &[_]c.Shape{ .{ .rows = 1, .n = 16, .k = 512 }, .{ .rows = 300, .n = 16, .k = 512 } },
    };
    return .{
        .name = f.name,
        .accum = .f32,
        .depth = depth,
        .ops = 12, // the division, eps, the root, the reciprocal, the products and RoPE's
        .mids = 4, // a rounding point decided the other way by the fp32 inverse: two output half-ulps a point
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = served,
        .sums = .{ .f16 = 40, .bf16 = 0, .why = why },
    };
}

const positions = 4096;
const len = 600;

/// Sums of squares in an accumulator and order (gate 4's simulation), or float64's.
const Sim = struct { acc: c.Accum, order: rows.Order };

pub const Qk = struct {
    rig: rig_.Rig,
    form: Form,
    k: *const qk.QkRows,
    ctl: ?cuda.Function = null,
    arena: std.heap.ArenaAllocator,
    case: qr.Case = undefined,
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

    pub fn prepare(self: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const f = self.form;
        if (shape.rows == 0 or shape.k != f.dim or shape.n % f.group != 0 or shape.n == 0) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const hk = shape.n / f.group;
        const ld = (shape.n + (if (f.v_from_k) hk else 2 * hk)) * f.dim;
        const x = try a.alloc(u16, r * ld);
        for (x, 0..) |*h, i| h.* = bfr.bits(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
        // Gemma 4 12B's norm weights: q 0.95 to 1.05, k 0.06 to 0.13
        const wb = try a.alloc(u16, 2 * f.dim);
        for (wb, 0..) |*v, i| v.* = bfr.bits(if (i < f.dim) 0.9 + 0.2 * rng.float(f32) else 0.05 + 0.1 * rng.float(f32));
        const rot = f.dim;
        const inv = try a.alloc(f32, rot / 2);
        for (inv, 0..) |*v, j| v.* = if (f.rope == .proportional and j >= 64) 0 else @floatCast(1 / std.math.pow(f64, if (f.rope == .proportional) 1e6 else 1e4, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rot))));
        const pos = try a.alloc(i32, r);
        const slot = try a.alloc(i32, r);
        for (pos, slot, 0..) |*p, *sl, i| { // distinct (slot, pos % len) a row: no row's cache entry overwrites another's
            p.* = @intCast((i / 2) % len + len * (i % 5));
            sl.* = @intCast(i % 2);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            const at = [_][2]usize{ .{ 0, ld / 3 }, .{ 1, ld / 2 }, .{ 2, ld - 1 }, .{ 3, 0 }, .{ 3, f.dim - 1 } };
            for (at, put) |p, v| if (p[0] < r) {
                x[p[0] * ld + p[1]] = bfr.bits(v);
            };
        }
        if (fill == .stress) { // integers 32 to 64 of either sign: squares sum past fp16's range, exactly in fp32; no rotation
            for (x) |*h| h.* = bfr.bits(@as(f32, @floatFromInt(32 + rng.uintLessThan(u32, 33))) * (if (rng.boolean()) @as(f32, 1) else -1));
            @memset(inv, 0);
        }
        const cosv = try a.alloc(f32, positions * rot);
        const sinv = try a.alloc(f32, positions * rot);
        if (f.rope == .table) for (0..positions) |p| for (0..rot) |i| {
            const ang = @as(f64, @floatFromInt(p)) * inv[i % (rot / 2)];
            cosv[p * rot + i] = @floatCast(@cos(ang));
            sinv[p * rot + i] = @floatCast(@sin(ang));
        };
        // the host's cos and sin of each row: the table's, or of fp32 angle pos * inv as the kernel forms it
        const cs = try a.alloc(f64, r * rot);
        const sn = try a.alloc(f64, r * rot);
        for (0..r) |row| for (0..rot) |i| {
            const p: usize = @intCast(pos[row]);
            if (f.rope == .table) {
                cs[row * rot + i] = cosv[p * rot + i];
                sn[row * rot + i] = sinv[p * rot + i];
            } else {
                const ang: f64 = @as(f32, @floatFromInt(p)) * inv[i % (rot / 2)];
                cs[row * rot + i] = @cos(ang);
                sn[row * rot + i] = @sin(ang);
            }
        };
        const wv = try a.alloc(f64, 2 * f.dim);
        for (wv, wb) |*v, h| v.* = bfr.value(h);
        const pts: ref.Points = if (f.round) .{ .row = true, .out = true } else .{};
        self.case = .{ .rows = r, .h = shape.n, .hk = hk, .dim = f.dim, .rot = rot, .x = x, .ld = ld, .v_from_k = f.v_from_k, .wq = wv[0..f.dim], .wk = wv[f.dim..], .cos = cs, .sin = sn, .computed = f.rope != .table, .points = pts, .v_norm = f.v_norm };
        self.dx = try self.rig.input(std.mem.sliceAsBytes(x));
        const nw = try self.rig.input(std.mem.sliceAsBytes(wb));
        if (f.rope == .table) {
            self.sh = .{ .norms = nw, .norms_bf16 = true, .cos = try self.rig.input(std.mem.sliceAsBytes(cosv)), .sin = try self.rig.input(std.mem.sliceAsBytes(sinv)), .rot = rot, .round = f.round, .v_norm = f.v_norm };
        } else self.sh = .{ .norms = nw, .norms_bf16 = true, .inv = try self.rig.input(std.mem.sliceAsBytes(inv)), .rot = rot, .round = f.round, .v_norm = f.v_norm };
        self.dpos = try self.rig.input(std.mem.sliceAsBytes(pos));
        self.dslot = try self.rig.input(std.mem.sliceAsBytes(slot));
        const kv = r * hk * f.dim * 2;
        const cache = 2 * len * hk * f.dim * 2;
        self.out = .{ .q = try self.rig.output(r * shape.n * f.dim * 2), .k = try self.rig.output(kv), .v = try self.rig.output(kv), .kc = try self.rig.scratch(cache), .vc = try self.rig.scratch(cache), .len = len };
        return true;
    }

    pub fn run(self: *Qk) !void {
        const k = self.case;
        const ld: i64 = @intCast(k.ld);
        const dim: i64 = @intCast(k.dim);
        const kp = self.dx + k.h * k.dim * 2;
        const src: qk.Src = .{ .q = self.dx, .q_ld = ld, .q_hs = dim, .k = kp, .k_ld = ld, .k_hs = dim, .v = if (k.v_from_k) kp else kp + k.hk * k.dim * 2, .v_ld = ld, .v_hs = dim };
        try self.k.rowsVia(self.ctl, self.form.keep orelse 0, self.rig.s, src, .{ .h = k.h, .hk = k.hk, .dim = k.dim }, self.sh, self.dpos, self.dslot, self.out, k.rows);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Qk) usize {
        return self.case.rows * self.case.heads() * self.case.dim;
    }

    /// Every head's float64 point at its rounding points, its sums of squares in `sim` (null: float64).
    fn outputs(self: *Qk, sim: ?Sim, want: []f64, mag: []f64) !void {
        const k = self.case;
        const a = self.arena.allocator();
        const x = try a.alloc(f64, k.dim);
        const sq = try a.alloc(f64, k.dim);
        const hi = try a.alloc(f64, k.dim);
        for (0..k.rows) |row| for (0..k.heads()) |hd| {
            for (x, sq, 0..) |*v, *q, i| {
                v.* = bfr.value(k.x[k.src(row, hd) + i]);
                q.* = v.* * v.*;
            }
            const ss: ?f64 = if (sim) |s| rows.sumIn(s.acc, s.order, sq) else null;
            const at = qr.outAt(k, row, hd);
            qr.head(k, row, hd, .{}, ss, x, want[at..][0..k.dim], hi);
            for (mag[at..][0..k.dim], want[at..][0..k.dim]) |*m, v| m.* = @abs(v) / 2; // v ~ ss^-1/2
        };
    }

    pub fn reference(self: *Qk, want: []f64, mag: []f64) !void {
        return self.outputs(null, want, mag);
    }

    pub fn simulate(self: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        return self.outputs(.{ .acc = acc, .order = order }, out, mag);
    }

    /// Every output in float64's range at the rounding points (a dropped point leaves it).
    pub fn check(self: *Qk, _: g.Gate) !?[]const u8 {
        const n = self.count();
        const out = try self.arena.allocator().alloc(u16, n);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.outputBytes());
        var stats: qr.Stats = .{};
        return qr.verify(self.rig.gpu.gpa, self.case, out, &stats, &self.msg);
    }
};

test {
    _ = qr;
}

test "every control keeps all but one option of the form it controls" {
    for (forms) |f| if (f.keep) |k| {
        try std.testing.expectEqual(@as(u32, 1), @popCount((row_pt | out_pt | vnorm_pt) & ~k));
    };
}
