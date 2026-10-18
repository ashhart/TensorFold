//! qk_rows.cu (#797) under the contract on CUDA: QK norm, rotate-half RoPE and v, against core/qk_ref.zig.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const qk = cuda.qk_rows;

pub const Norm = enum { none, f32w, bf16w };
pub const Rope = enum { none, table, inv };

/// One registered form: head dim, query heads a key head, rotated dims, norm weights and where cos and sin come from.
pub const Form = struct { name: []const u8, dim: usize, group: usize, rot: usize, norm: Norm, rope: Rope, h: usize };

/// Kolibri (48/4 x 128, sliding and full), a warp a head without the norm, the 27B (24/4 x 256, rotary 64), dim 64.
pub const forms = [_]Form{
    .{ .name = "cuda qk_rows tf_qk_rows_w (norm, rotation from a table)", .dim = 128, .group = 12, .rot = 128, .norm = .f32w, .rope = .table, .h = 48 },
    .{ .name = "cuda qk_rows tf_qk_rows_w (norm, no rotation)", .dim = 128, .group = 12, .rot = 0, .norm = .f32w, .rope = .none, .h = 48 },
    .{ .name = "cuda qk_rows tf_qk_rows_w (bf16 norm, rotation from inv_freq)", .dim = 128, .group = 12, .rot = 128, .norm = .bf16w, .rope = .inv, .h = 48 },
    .{ .name = "cuda qk_rows tf_qk_rows_w (no norm, rotation from a table)", .dim = 128, .group = 8, .rot = 128, .norm = .none, .rope = .table, .h = 16 },
    .{ .name = "cuda qk_rows tf_qk_rows (dim 256, norm, rotary 64 from a table)", .dim = 256, .group = 6, .rot = 64, .norm = .f32w, .rope = .table, .h = 24 },
    .{ .name = "cuda qk_rows tf_qk_rows (dim 256, bf16 norm, rotary 64 from inv_freq)", .dim = 256, .group = 6, .rot = 64, .norm = .bf16w, .rope = .inv, .h = 24 },
    .{ .name = "cuda qk_rows tf_qk_rows (dim 64, no norm, rotary 32 from a table)", .dim = 64, .group = 4, .rot = 32, .norm = .none, .rope = .table, .h = 8 },
};

/// A rotated value: x rstd, times w, times cos (cosf and sinf: 2 ulp, 4 units more), then the add; unrotated fewer.
fn depthTable(_: c.Shape) usize {
    return 4;
}

fn depthInv(_: c.Shape) usize {
    return 8;
}

fn depthBare(_: c.Shape) usize {
    return 2;
}

const norm_why = "these stress inputs do not separate bf16 square sums beyond the normed output bound; fp16 overflows";

/// Shape: rows x query heads (n) x head dim (k).
pub fn spec(f: Form) c.Spec {
    const served = switch (f.dim) {
        128 => if (f.h == 48) &[_]c.Shape{ .{ .rows = 1, .n = 48, .k = 128 }, .{ .rows = 300, .n = 48, .k = 128 } } else &[_]c.Shape{ .{ .rows = 1, .n = 16, .k = 128 }, .{ .rows = 300, .n = 16, .k = 128 } },
        256 => &[_]c.Shape{ .{ .rows = 1, .n = 24, .k = 256 }, .{ .rows = 300, .n = 24, .k = 256 } },
        else => &[_]c.Shape{ .{ .rows = 1, .n = 8, .k = 64 }, .{ .rows = 300, .n = 8, .k = 64 } },
    };
    const depth: *const fn (c.Shape) usize = if (f.norm == .none) &depthBare else if (f.rope == .inv) &depthInv else &depthTable;
    // the norm's sum of squares (14 deep, half of it in rsqrt), the division, eps, the root and the reciprocal
    const ops: f64 = if (f.norm == .none) 0 else 12;
    const sums: c.Sums = if (f.norm == .none) .{} else .{ .bf16 = 0, .why = norm_why };
    return .{ .name = f.name, .accum = .f32, .depth = depth, .ops = ops, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = served, .sums = sums };
}

const positions = 4096;
const len = 600;

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const Qk = struct {
    rig: rig_.Rig,
    form: Form,
    k: *const qk.QkRows,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    w: []f64 = &.{},
    cos: []f32 = &.{},
    sin: []f32 = &.{},
    inv: []f32 = &.{},
    pos: []i32 = &.{},
    dx: u64 = 0,
    sh: qk.Shape = .{},
    dpos: u64 = 0,
    dslot: u64 = 0,
    out: qk.Out = undefined,

    pub fn deinit(self: *Qk) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Qk) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Qk);

    fn hk(self: *const Qk) usize {
        return self.shape.n / self.form.group;
    }

    fn heads(self: *const Qk) usize {
        return self.shape.n + 2 * self.hk();
    }

    pub fn prepare(self: *Qk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const f = self.form;
        if (shape.rows == 0 or shape.k != f.dim or shape.n % f.group != 0 or shape.n == 0) return false;
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const width = self.heads() * f.dim;
        self.x = try a.alloc(u16, r * width);
        for (self.x, 0..) |*h, i| h.* = c.toBf16(rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 8) else 1));
        const rot = @max(f.rot, 2);
        self.cos = try a.alloc(f32, positions * rot);
        self.sin = try a.alloc(f32, positions * rot);
        for (0..positions) |p| for (0..rot) |i| {
            const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * (i % (rot / 2)))) / @as(f64, @floatFromInt(rot)));
            self.cos[p * rot + i] = @floatCast(@cos(ang));
            self.sin[p * rot + i] = @floatCast(@sin(ang));
        };
        self.inv = try a.alloc(f32, rot / 2);
        for (self.inv, 0..) |*v, j| v.* = @floatCast(1 / std.math.pow(f64, 10000, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(rot))));
        const wf = try a.alloc(f32, 2 * f.dim);
        for (wf) |*v| v.* = 0.5 + rng.float(f32);
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            const at = [_][2]usize{ .{ 0, width / 3 }, .{ 1, width / 2 }, .{ 2, width - 1 }, .{ 3, 0 }, .{ 3, f.dim - 1 } };
            for (at, put) |p, v| if (p[0] < r) {
                self.x[p[0] * width + p[1]] = c.toBf16(v);
            };
        }
        if (fill == .stress) {
            if (f.norm == .none) {
                // x c - x' s with c, s = 65535, 65534, x = 255: products near 2^24, exact in fp32, a difference of 255
                @memset(self.x, c.toBf16(255));
                @memset(self.cos, 65535);
                @memset(self.sin, 65534);
            } else {
                // integers 32 to 64 of either sign: squares sum past fp16's range, exactly in fp32; cos 1, sin 0
                for (self.x) |*h| h.* = c.toBf16(@as(f32, @floatFromInt(32 + rng.uintLessThan(u32, 33))) * (if (rng.boolean()) @as(f32, 1) else -1));
                @memset(self.cos, 1);
                @memset(self.sin, 0);
                @memset(self.inv, 0);
            }
        }
        self.w = try a.alloc(f64, 2 * f.dim);
        const wb = try a.alloc(u16, 2 * f.dim);
        for (self.w, wf, wb) |*v, x, *b| {
            b.* = c.toBf16(x);
            v.* = if (f.norm == .bf16w) bf(b.*) else x;
        }
        self.pos = try a.alloc(i32, r);
        const slot = try a.alloc(i32, r);
        for (self.pos, slot, 0..) |*p, *sl, i| { // distinct (slot, pos % len) a row: no row's cache entry overwrites another's
            p.* = @intCast((i / 2) % len + len * (i % 5));
            sl.* = @intCast(i % 2);
        }
        self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
        const nw: ?u64 = switch (f.norm) {
            .none => null,
            .f32w => try self.rig.input(std.mem.sliceAsBytes(wf)),
            .bf16w => try self.rig.input(std.mem.sliceAsBytes(wb)),
        };
        const tc = try self.rig.input(std.mem.sliceAsBytes(self.cos));
        const ts = try self.rig.input(std.mem.sliceAsBytes(self.sin));
        const ti = try self.rig.input(std.mem.sliceAsBytes(self.inv));
        self.sh = .{ .norms = nw, .norms_bf16 = f.norm == .bf16w, .cos = if (f.rope == .table) tc else null, .sin = if (f.rope == .table) ts else null, .inv = if (f.rope == .inv) ti else null, .rot = f.rot };
        self.dpos = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.dslot = try self.rig.input(std.mem.sliceAsBytes(slot));
        const kv = r * self.hk() * f.dim * 2;
        const cache = 2 * len * self.hk() * f.dim * 2;
        self.out = .{ .q = try self.rig.output(r * shape.n * f.dim * 2), .k = try self.rig.output(kv), .v = try self.rig.output(kv), .kc = try self.rig.scratch(cache), .vc = try self.rig.scratch(cache), .len = len };
        return true;
    }

    pub fn run(self: *Qk) !void {
        try self.k.rows(self.rig.s, self.dx, .{ .h = self.shape.n, .hk = self.hk(), .dim = self.form.dim }, self.sh, self.dpos, self.dslot, self.out, self.shape.rows);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Qk) usize {
        return self.shape.rows * self.heads() * self.form.dim;
    }

    /// Output o's row and head (q heads, k heads, v heads as the outputs lie: all q, then all k, then all v).
    fn locate(self: *const Qk, o: usize) [2]usize {
        const r = self.shape.rows;
        const h = self.shape.n;
        const d = self.form.dim;
        const i = o / d;
        if (i < r * h) return .{ i / h, i % h };
        const j = i - r * h;
        const hk_ = self.hk();
        if (j < r * hk_) return .{ j / hk_, h + j % hk_ };
        return .{ (j - r * hk_) / hk_, h + hk_ + (j - r * hk_) % hk_ };
    }

    /// One head's outputs and term magnitudes; its sums in `acc` and `order` (null: float64, core/qk_ref.head).
    fn head(self: *Qk, r: usize, hd: usize, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const f = self.form;
        const d = f.dim;
        const a = self.arena.allocator();
        const x = try a.alloc(f64, d);
        for (x, 0..) |*v, i| v.* = bf(self.x[(r * self.heads() + hd) * d + i]);
        if (hd >= self.shape.n + self.hk()) {
            @memcpy(want, x);
            for (mag, x) |*m, v| m.* = @abs(v);
            return;
        }
        const w: ?[]const f64 = if (f.norm == .none) null else self.w[(if (hd >= self.shape.n) d else 0)..][0..d];
        const rot = @max(f.rot, 2);
        const p: usize = @intCast(self.pos[r]);
        const cv = try a.alloc(f64, rot);
        const sv = try a.alloc(f64, rot);
        for (0..rot) |i| {
            if (f.rope == .inv) {
                const ang: f64 = @as(f32, @floatFromInt(p)) * self.inv[i % (rot / 2)]; // the angle in fp32, as served
                cv[i] = @cos(ang);
                sv[i] = @sin(ang);
            } else {
                cv[i] = self.cos[p * rot + i];
                sv[i] = self.sin[p * rot + i];
            }
        }
        const roped = f.rope != .none;
        var ss: f64 = 0;
        const sq = try a.alloc(f64, d);
        for (sq, x) |*q, v| {
            q.* = v * v;
            ss += q.*;
        }
        if (acc) |ac| ss = rows.sumIn(ac, order, sq);
        const inv = if (w != null) 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + 1e-6) else 1;
        const z = try a.alloc(f64, d);
        for (z, x, 0..) |*o, v, i| o.* = v * inv * (if (w) |ww| ww[i] else 1);
        if (acc == null) core.qk_ref.head(x, w, 1e-6, if (roped) cv else null, if (roped) sv else null, f.rot, want);
        const half = f.rot / 2;
        for (want, mag, 0..) |*o, *m, i| {
            if (!roped or i >= f.rot) {
                if (acc != null) o.* = z[i];
                m.* = @abs(z[i]);
                continue;
            }
            const pi = if (i < half) i + half else i - half;
            const t = [2]f64{ z[i] * cv[i], (if (i < half) -z[pi] else z[pi]) * sv[i] };
            if (acc) |ac| o.* = rows.sumIn(ac, order, &t);
            m.* = @abs(t[0]) + @abs(t[1]);
        }
    }

    fn outputs(self: *Qk, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const d = self.form.dim;
        var o: usize = 0;
        while (o < want.len) : (o += d) {
            const at = self.locate(o);
            try self.head(at[0], at[1], acc, order, want[o..][0..d], mag[o..][0..d]);
        }
    }

    pub fn reference(self: *Qk, want: []f64, mag: []f64) !void {
        return self.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Qk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        return self.outputs(acc, order, out, mag);
    }
};
