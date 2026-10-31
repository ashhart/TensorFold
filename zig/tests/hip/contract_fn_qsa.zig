//! Flash Next's QSA decode prep under the contract on HIP: one launch, bit-equal to the seven launches it replaces.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// Flash Next's QSA layer: 24 query heads and 2 kv heads of 256, 4 index heads of 128 and the raw index key after them.
const heads = 24;
const kv_heads = 2;
const hd = 256;
const ih = 4;
const di = 128;
const width = (ih + 1) * di;
const ratio = 4;
const rotary = 64;
const theta: f32 = 10_000_000;
const eps: f32 = 1e-6;
/// The first row's position: rows from it complete blocks at 1023, 1027 and on, each block's first rows cached before.
const first = 1021;
const cap = 1088;
const slots = cap / ratio;

/// The launch's fp32 q and index q rows, or the caches it writes (keys, values, raw index keys, pooled keys).
pub const Part = enum { wide, caches };

/// A lane's 16 squares of a 512-wide head and the wave's xor tree; a pool's block tree is no deeper.
fn normDepth(_: c.Shape) usize {
    return 1 + 512 / 32 + 5;
}

const why = "the outputs pass two roundings to the activation kind: a bf16 sum of squares misses by a few of " ++
    "their ulps (fp16's overflows)";

const names = [2][2][]const u8{
    .{ "hip flashnext tf_fn_qsa_prep (q and index q rows, fp16)", "hip flashnext tf_fn_qsa_prep (q and index q rows, bf16)" },
    .{ "hip flashnext tf_fn_qsa_prep (caches and pools, fp16)", "hip flashnext tf_fn_qsa_prep (caches and pools, bf16)" },
};

/// A decode row, a draft window, a verify window under the prompt path's 16 rows.
const served = [_]c.Shape{
    .{ .rows = 1, .n = heads, .k = hd },
    .{ .rows = 4, .n = heads, .k = hd },
    .{ .rows = 15, .n = heads, .k = hd },
};

pub fn spec(p: Part, k: kit.Kind) c.Spec {
    return .{
        .name = names[@backingInt(p)][@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = normDepth,
        .ops = 8,
        .mids = 1,
        .out = if (p == .wide) .f32 else kit.elem(k),
        .replaces = "tf_qk_rope (q, index q, k), tf_kv_write_at (v, raw key), tf_copy_cols and tf_fn_idx_pool_at",
        // the rows' block ends: 4 positions a pooled block
        .tiles = .{ .{ .size = ratio }, .{}, .{} },
        .served = &served,
        .sums = .{ .bf16 = 0, .why = why },
    };
}

/// shape.rows rows at positions first ..: their projections of the kind, the caches' earlier contents prepared.
pub const Qsa = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    qg: []f64 = &.{}, // [rows][24][512]: q, then the gate
    kr: []f64 = &.{}, // [rows][2][256]
    vr: []f64 = &.{},
    idx: []f64 = &.{}, // [rows][640]: the index q heads, then the raw key
    w: [4][]f64 = undefined, // q, index q, k and pool norms
    before: [4][]f64 = undefined, // K [2][cap][256], V, raw [cap][128], pooled [slots][128] before the launch
    before_bytes: [4][]u8 = undefined,
    tol: []f64 = &.{},
    dev: [9]u64 = @splat(0), // qg, k, v, idx, the four norms, positions
    outs: [6]u64 = @splat(0), // q rows, index q rows, K, V, raw, pooled
    base: [7]u64 = @splat(0), // the same for the replaced launches, and the copied raw keys

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, p: Part) Qsa {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .part = p, .arena = .init(a) };
    }

    pub fn deinit(self: *Qsa) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Qsa) g.Kernel {
        return .{ .spec = spec(self.part, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Qsa);

    fn lens(self: *const Qsa) [6]usize {
        const n = self.shape.rows;
        return .{ n * heads * hd, n * ih * di, kv_heads * cap * hd, kv_heads * cap * hd, cap * di, slots * di };
    }

    fn values(self: *Qsa, rng: std.Random, n: usize, stress: bool) ![]f64 {
        const v = try self.arena.allocator().alloc(f64, n);
        for (v) |*x| x.* = if (stress)
            @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 129))) - 64)
        else
            kit.round(self.k, rng.float(f64) * 4 - 2);
        return v;
    }

    /// Projections in [-2, 2) of the kind, norms in [1, 1.5); gate 4: integers in [-64, 64] and unit norms.
    pub fn prepare(self: *Qsa, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > cap - first or shape.n != heads or shape.k != hd) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.rows;
        const stress = fill == .stress;
        self.qg = try self.values(rng, n * heads * 2 * hd, stress);
        self.kr = try self.values(rng, n * kv_heads * hd, stress);
        self.vr = try self.values(rng, n * kv_heads * hd, stress);
        self.idx = try self.values(rng, n * width, stress);
        const l = self.lens();
        for (&self.before, l[2..]) |*x, len| x.* = try self.values(rng, len, stress);
        for (&self.w, [4]usize{ hd, di, hd, di }) |*x, len| {
            x.* = try a.alloc(f64, len);
            for (x.*) |*v| v.* = if (stress) 1 else @as(f32, @floatCast(1 + rng.float(f64) * 0.5));
        }
        if (fill == .nonfinite) self.plant();
        const kb = kit.size(self.k);
        const ins = [4][]f64{ self.qg, self.kr, self.vr, self.idx };
        for (ins, self.dev[0..4]) |vals, *at| at.* = try self.rig.input(try self.kindBytes(vals));
        for (self.w, self.dev[4..8]) |vals, *at| at.* = try self.rig.input(try kit.f32Bytes(a, vals));
        const pos = try a.alloc(i32, n);
        for (pos, 0..) |*p, r| p.* = @intCast(first + r);
        self.dev[8] = try self.rig.input(std.mem.sliceAsBytes(pos));
        for (&self.before_bytes, self.before) |*b, vals| b.* = try self.kindBytes(vals);
        const sizes = [6]usize{ l[0] * 4, l[1] * 4, l[2] * kb, l[3] * kb, l[4] * kb, l[5] * kb };
        for (&self.outs, sizes, 0..) |*at, len, i| {
            const mine = (i < 2) == (self.part == .wide);
            at.* = if (mine) try self.rig.output(len) else try self.rig.scratch(len);
        }
        for (self.base[0..6], sizes) |*at, len| at.* = try self.rig.scratch(len);
        self.base[6] = try self.rig.scratch(n * di * kb);
        return true;
    }

    fn kindBytes(self: *Qsa, vals: []const f64) ![]u8 {
        const b = try self.arena.allocator().alloc(u8, vals.len * kit.size(self.k));
        for (vals, 0..) |v, i| kit.put(self.k, b, i, v);
        return b;
    }

    /// q row 0 a NaN; k row 1 a +Inf; row 2's raw index key both infinities (its block completes); v row 0 a -Inf.
    fn plant(self: *Qsa) void {
        const inf = std.math.inf(f64);
        self.qg[3] = std.math.nan(f64);
        self.vr[hd + 2] = -inf;
        if (self.shape.rows > 1) self.kr[kv_heads * hd + 5] = inf;
        if (self.shape.rows > 2) {
            self.idx[2 * width + ih * di + 1] = inf;
            self.idx[2 * width + ih * di + 2] = -inf;
        }
    }

    /// The caches back to their prepared contents.
    fn restore(self: *Qsa, at: []const u64) !void {
        for (at, self.before_bytes) |p, b| try self.rig.put(p, b);
    }

    pub fn run(self: *Qsa) !void {
        try self.restore(self.outs[2..6]);
        const d = self.dev;
        const o = self.outs;
        const n: c_int = @intCast(self.shape.rows);
        const Rope = hip.launches.flashnext.QsaRope;
        try self.l.fn_qsa_prep(.{
            .q = Rope{ .src = d[0], .s_row = 2 * heads * hd, .s_head = 2 * hd, .heads = heads, .width = hd, .weight = d[4], .wide = o[0] },
            .iq = Rope{ .src = d[3], .s_row = width, .s_head = di, .heads = ih, .width = di, .weight = d[5], .wide = o[1] },
            .k = .{ .src = d[1], .s_row = kv_heads * hd, .s_head = hd, .heads = kv_heads, .width = hd, .weight = d[6], .cache = o[2] },
            .v = d[2],
            .v_cache = o[3],
            .v_width = hd,
            .raw = d[3],
            .raw_stride = width,
            .raw_offset = ih * di,
            .raw_cache = o[4],
            .raw_width = di,
            .pooled = o[5],
            .pool_scale = d[7],
            .ratio = ratio,
            .pos = d[8],
            .seg_pos = d[8],
            .rows = n,
            .n = n,
            .total = cap,
            .kind = @backingInt(self.k),
            .eps = eps,
            .rotary = rotary,
            .theta = theta,
        }, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// tf_qk_rope for q, index q and k, v and the raw index key written at their positions, then the pools.
    pub fn replaced(self: *Qsa) ![]const u8 {
        try self.restore(self.base[2..6]);
        const d = self.dev;
        const b = self.base;
        const n: c_int = @intCast(self.shape.rows);
        const kind = @backingInt(self.k);
        const st = self.rig.s.handle;
        const C = ?*const anyopaque;
        const P = ?*anyopaque;
        const CF = ?[*]const f32;
        const CI = ?[*]const i32;
        const F = ?[*]f32;
        const pos = kit.ptr(CI, d[8]);
        const l = self.l;
        try l.tf_qk_rope(kit.ptr(C, d[0]), kind, 2 * heads * hd, 2 * hd, kit.ptr(CF, d[4]), eps, n, heads, hd, rotary, theta, pos, kit.ptr(F, b[0]), null, 0, st);
        try l.tf_qk_rope(kit.ptr(C, d[3]), kind, width, di, kit.ptr(CF, d[5]), eps, n, ih, di, rotary, theta, pos, kit.ptr(F, b[1]), null, 0, st);
        try l.tf_qk_rope(kit.ptr(C, d[1]), kind, kv_heads * hd, hd, kit.ptr(CF, d[6]), eps, n, kv_heads, hd, rotary, theta, pos, null, kit.ptr(P, b[2]), cap, st);
        try l.tf_kv_write_at(kit.ptr(C, d[2]), kit.ptr(P, b[3]), kind, n, kv_heads, hd, cap, pos, st);
        try l.tf_copy_cols(kit.ptr(C, d[3]), width, ih * di, kit.ptr(P, b[6]), kind, n, di, st);
        try l.tf_kv_write_at(kit.ptr(C, b[6]), kit.ptr(P, b[4]), kind, n, 1, di, cap, pos, st);
        try l.fn_idx_pool_at(kit.ptr(C, b[4]), kit.ptr(P, b[5]), kit.ptr(CF, d[7]), kind, pos, n, di, ratio, rotary, theta, eps, st);
        const sizes = self.byteLens();
        const a = self.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        const lo: usize = if (self.part == .wide) 0 else 2;
        const hi: usize = if (self.part == .wide) 2 else 6;
        for (b[lo..hi], sizes[lo..hi]) |at, len| try all.appendSlice(a, try self.rig.bytes(at, len));
        return all.items;
    }

    fn byteLens(self: *const Qsa) [6]usize {
        const l = self.lens();
        const kb = kit.size(self.k);
        return .{ l[0] * 4, l[1] * 4, l[2] * kb, l[3] * kb, l[4] * kb, l[5] * kb };
    }

    pub fn count(self: *Qsa) usize {
        const l = self.lens();
        return if (self.part == .wide) l[0] + l[1] else l[2] + l[3] + l[4] + l[5];
    }

    /// Gate 4's accumulator and order of every sum (null: float64).
    const How = struct { c.Accum, g.Order };

    fn sum(sim: ?How, t: []const f64) f64 {
        return if (sim) |how| rows.sumIn(how[0], how[1], t) else rows.sumIn(.none, .sequential, t);
    }

    /// The kind's unit roundoff: half its ulp at 1.
    fn unit(self: *const Qsa) f64 {
        return if (self.k == .bf16) 0x1p-8 else 0x1p-11;
    }

    /// RMS norm with weight, then the rotation at position p over the first 64 values (tf_qk_rope's formula).
    fn rope(self: *const Qsa, x: []const f64, w: []const f64, p: f64, sim: ?How, out: []f64, tol: ?[]f64, extra: f64) void {
        var sq: [512]f64 = undefined;
        for (sq[0..x.len], x) |*s, v| s.* = v * v;
        const inv = 1 / @sqrt(sum(sim, sq[0..x.len]) / @as(f64, @floatFromInt(x.len)) + eps);
        var nrm: [512]f64 = undefined;
        for (nrm[0..x.len], x, w) |*y, v, s| y.* = v * inv * s;
        const half = rotary / 2;
        const u = self.unit();
        for (0..x.len) |i| {
            if (i >= rotary) {
                out[i] = nrm[i];
                if (tol) |t| t[i] = u * 2 * @abs(nrm[i]) * (1 + 0x1p-6) + (0x1p-20 + extra) * @abs(nrm[i]) + 0x1p-120;
                continue;
            }
            const f = i % half;
            const ang = p / std.math.pow(f64, theta, @as(f64, @floatFromInt(f)) / half);
            const cs = @cos(ang);
            const sn = @sin(ang);
            const x1 = nrm[f];
            const x2 = nrm[f + half];
            out[i] = if (i < half) x1 * cs - x2 * sn else x1 * sn + x2 * cs;
            const m = if (i < half) @abs(x1 * cs) + @abs(x2 * sn) else @abs(x1 * sn) + @abs(x2 * cs);
            const amp = @abs(x1) + @abs(x2);
            // the normed values' and the output's roundings; fp32's angle (powf, p freq), cos, sin and products
            if (tol) |t| t[i] = u * (m + @abs(out[i])) * (1 + 0x1p-6) + (p * 0x1p-21 + 0x1p-20 + extra) * amp + 0x1p-120;
        }
    }

    /// Every output of the launch in float64, the part's slice into `out`; tol the gate-4 allowance of each.
    fn outputs(self: *Qsa, out: []f64, tol: ?[]f64, sim: ?How) !void {
        const a = self.arena.allocator();
        const l = self.lens();
        var total: usize = 0;
        for (l) |x| total += x;
        const all = try a.alloc(f64, total);
        const tl = try a.alloc(f64, total);
        @memset(tl, 0x1p-120);
        var at: [6]usize = undefined;
        var off: usize = 0;
        for (&at, l) |*x, len| {
            x.* = off;
            off += len;
        }
        for (self.before, at[2..]) |vals, o| @memcpy(all[o..][0..vals.len], vals);
        const n = self.shape.rows;
        const t = if (tol != null) tl else null;
        for (0..n) |r| {
            const p: f64 = @floatFromInt(first + r);
            for (0..heads) |h| {
                const o = at[0] + (r * heads + h) * hd;
                self.rope(self.qg[(r * heads + h) * 2 * hd ..][0..hd], self.w[0], p, sim, all[o..][0..hd], if (t) |x| x[o..][0..hd] else null, 0);
            }
            for (0..ih) |h| {
                const o = at[1] + (r * ih + h) * di;
                self.rope(self.idx[r * width + h * di ..][0..di], self.w[1], p, sim, all[o..][0..di], if (t) |x| x[o..][0..di] else null, 0);
            }
            for (0..kv_heads) |h| {
                const o = at[2] + (h * cap + first + r) * hd;
                self.rope(self.kr[(r * kv_heads + h) * hd ..][0..hd], self.w[2], p, sim, all[o..][0..hd], if (t) |x| x[o..][0..hd] else null, 0);
                @memcpy(all[at[3] + (h * cap + first + r) * hd ..][0..hd], self.vr[(r * kv_heads + h) * hd ..][0..hd]);
            }
            @memcpy(all[at[4] + (first + r) * di ..][0..di], self.idx[r * width + ih * di ..][0..di]);
        }
        // the blocks the rows complete: the mean of their 4 raw keys (rounded to the kind), then pooled as a row
        var mean: [di]f64 = undefined;
        var four: [ratio]f64 = undefined;
        for (0..n) |r| {
            const end = first + r + 1;
            if (end % ratio != 0) continue;
            const b = end / ratio - 1;
            for (&mean, 0..) |*m, d| {
                for (&four, 0..) |*v, j| v.* = all[at[4] + (b * ratio + j) * di + d];
                m.* = sum(sim, &four) / ratio;
            }
            const o = at[5] + b * di;
            const p: f64 = @floatFromInt(b * ratio);
            self.rope(&mean, self.w[3], p, sim, all[o..][0..di], if (t) |x| x[o..][0..di] else null, 2 * self.unit());
        }
        const lo = if (self.part == .wide) 0 else at[2];
        @memcpy(out, all[lo..][0..out.len]);
        if (tol) |x| @memcpy(x, tl[lo..][0..out.len]);
    }

    pub fn reference(self: *Qsa, want: []f64, mag: []f64) !void {
        self.tol = try self.arena.allocator().alloc(f64, want.len);
        try self.outputs(want, self.tol, null);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    /// Two roundings to the kind, fp32's norm, angle and rotation, and for a pool its mean's rounding.
    pub fn bounds(self: *Qsa, _: []const f64, _: []const f64, allow: []f64) !void {
        @memcpy(allow, self.tol);
    }

    pub fn simulate(self: *Qsa, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.outputs(out, null, .{ acc, order });
    }
};
