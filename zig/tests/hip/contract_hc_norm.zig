//! Flash Next's HC norms under the contract on HIP: the 16-byte norm and the MoE combine in it, bit-equal to chains.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const mix = @import("contract_hc_mix.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const C = ?*const anyopaque;
const P = ?*anyopaque;
const CF = ?[*]const f32;
const nan = std.math.nan(f64);
const inf = std.math.inf(f64);

/// tf_fn_hc_norm_v with a branch added to h in place, or without (x only); tf_fn_hc_norm_moe and its 16-byte form.
pub const Mode = enum { v_branch, v_plain, moe, moe_v };

const dims = 2560; // Flash Next's stream width, the one the 16-byte kernels take
const slots = 11; // Flash Next's 10 routed experts and the shared one
const eps: f32 = 1e-6;

/// The old norm's sum of squares: up to 4 a thread, the wave's xor tree, the block's 32 waves; the combine's slots.
fn depth(s: c.Shape) usize {
    return (s.k + 1023) / 1024 + 5 + 32 + slots;
}

const why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const rule = "the chain's bytes and every float64 output class";

pub fn spec(m: Mode, k: kit.Kind) c.Spec {
    var s = c.Spec{
        .name = switch (m) {
            .v_branch => "hip tf_fn_hc_norm_v (bf16, a branch added in place)",
            .v_plain => "hip tf_fn_hc_norm_v (bf16, no branch)",
            .moe => if (k == .bf16) "hip tf_fn_hc_norm_moe (bf16)" else "hip tf_fn_hc_norm_moe (fp16)",
            .moe_v => "hip tf_fn_hc_norm_moe_v (bf16, a prompt's rows)",
        },
        .accum = .f32,
        .depth = depth,
        // rsqrt, the mean and eps, two products
        .ops = 8,
        .out = kit.elem(k),
        .replaces = if (m == .moe or m == .moe_v) "tf_moe_combine then tf_fn_hc_norm" else "tf_fn_hc_norm",
        .nan = .{ .stated = rule },
        // one to four streams, Flash Next's four the widest
        .tiles = .{ .{}, .{ .size = 4 }, .{} },
        .served = switch (m) {
            .v_branch, .v_plain => &.{
                .{ .rows = 1, .n = 4, .k = dims },
                .{ .rows = 16, .n = 4, .k = dims },
                .{ .rows = 300, .n = 4, .k = dims },
            },
            .moe => &.{
                .{ .rows = 1, .n = 4, .k = dims },
                .{ .rows = 8, .n = 4, .k = dims },
                .{ .rows = 16, .n = 4, .k = dims },
            },
            .moe_v => &.{
                .{ .rows = 1, .n = 4, .k = dims },
                .{ .rows = 64, .n = 4, .k = dims },
                .{ .rows = 300, .n = 4, .k = dims },
            },
        },
        .sums = .{ .bf16 = 0, .why = why },
    };
    // the 1,024-thread block's edges in the width it takes, up to 4,096
    if (m == .moe) s.tiles[2] = .{ .size = 1024 };
    return s;
}

/// h [rows][streams][k] of the kind, x [rows][streams][k]: h += round(branch * inject) (a branch), x = h rinv scale.
pub const Norm = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    mode: Mode,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    h: []f64 = &.{},
    br: []f64 = &.{}, // [rows][k], the v modes' branch
    y: []f64 = &.{}, // [rows][slots][k], the MoE's expert rows
    wts: []f64 = &.{},
    inj: []f64 = &.{},
    scale: []f64 = &.{},
    h_bytes: []u8 = &.{},
    in: [5]u64 = @splat(0), // branch or y, wts, inject, scale, h when not in place
    out: [2]u64 = @splat(0), // h in place, x
    base: [3]u64 = @splat(0), // the chain's h, x and combined branch

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) Norm {
        const k: kit.Kind = if (m == .moe) kit.act() else .bf16;
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .mode = m, .k = k, .arena = .init(a) };
    }

    pub fn deinit(self: *Norm) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Norm) g.Kernel {
        return .{ .spec = spec(self.mode, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Norm);

    fn moe(self: *const Norm) bool {
        return self.mode == .moe or self.mode == .moe_v;
    }

    fn inPlace(self: *const Norm) bool {
        return self.mode != .v_plain;
    }

    fn at(self: *const Norm, r: usize, s: usize, d: usize) usize {
        return (r * self.shape.n + s) * self.shape.k + d;
    }

    /// h in [-3, 3), branch in [-1, 1), inject in [-2, 2), y in [-2, 2) by weights in [0, 0.5), scale in [1, 1.25).
    pub fn prepare(self: *Norm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 4 or shape.k == 0) return false;
        if (if (self.mode == .moe) shape.k > 4096 else shape.k != dims) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = self.k;
        const n = shape.rows * shape.n * shape.k;
        self.h = (try kit.values(a, rng, k, n, -3, 3))[0];
        self.br = (try kit.values(a, rng, k, shape.rows * shape.k, -1, 1))[0];
        self.inj = (try kit.values(a, rng, k, shape.rows * shape.n, -2, 2))[0];
        self.y = if (self.moe()) try kit.floats(a, rng, shape.rows * slots * shape.k, -2, 2) else &.{};
        self.wts = try kit.floats(a, rng, shape.rows * slots, 0, 0.5);
        self.scale = try kit.floats(a, rng, shape.n * shape.k, 1, 1.25);
        if (fill == .stress) try self.stress(seed);
        if (fill == .nonfinite) self.plant();
        self.h_bytes = try a.alloc(u8, n * kit.size(k));
        for (self.h, 0..) |v, i| kit.put(k, self.h_bytes, i, v);
        const branch = if (self.moe()) try kit.f32Bytes(a, self.y) else blk: {
            const b = try a.alloc(u8, self.br.len * kit.size(k));
            for (self.br, 0..) |v, i| kit.put(k, b, i, v);
            break :blk b;
        };
        const ib = try a.alloc(u8, self.inj.len * kit.size(k));
        for (self.inj, 0..) |v, i| kit.put(k, ib, i, v);
        const ins = [_][]const u8{ branch, try kit.f32Bytes(a, self.wts), ib, try kit.f32Bytes(a, self.scale) };
        for (ins, self.in[0..4]) |b, *p| p.* = try self.rig.input(b);
        if (self.inPlace()) self.out[0] = try self.rig.output(self.h_bytes.len) else self.in[4] = try self.rig.input(self.h_bytes);
        self.out[1] = try self.rig.output(self.h_bytes.len);
        self.base = .{ try self.rig.scratch(self.h_bytes.len), try self.rig.scratch(self.h_bytes.len), try self.rig.scratch(shape.rows * shape.k * kit.size(k)) };
        return true;
    }

    /// Gate 4: the norm's rows of integers (rows.zig's) and nothing added, so its sums are exact in fp32.
    fn stress(self: *Norm, seed: u64) !void {
        const s = self.shape;
        const r = try rows.Rows.init(self.arena.allocator(), .rms, .{ .rows = s.rows * s.n, .n = s.k, .k = s.k });
        r.fill(.stress, seed);
        for (self.h, r.x) |*v, b| v.* = c.fromBf16(b);
        @memset(self.br, 0);
        @memset(self.y, 0);
    }

    /// h: NaN, +Inf, -Inf, both infinities in one stream (rows 0-3); the branch +Inf and an inject NaN in row 4.
    fn plant(self: *Norm) void {
        const s = self.shape;
        const at_ = [_][3]usize{ .{ 0, 0, s.k / 3 }, .{ 1, 1, s.k / 2 }, .{ 2, 2, s.k - 1 }, .{ 3, 3, 0 }, .{ 3, 3, s.k - 1 } };
        const put = [_]f64{ nan, inf, -inf, inf, -inf };
        for (at_, put) |p, v| if (p[0] < s.rows) {
            self.h[self.at(p[0], p[1] % s.n, p[2])] = v;
        };
        if (s.rows < 5) return;
        if (self.moe()) self.y[(4 * slots + 3) * s.k + 7 % s.k] = inf else self.br[4 * s.k + 7 % s.k] = inf;
        self.inj[4 * s.n + s.n - 1] = nan;
    }

    fn ptr(at_: u64) ?*anyopaque {
        return @ptrFromInt(at_);
    }

    /// h back to the prepared rows (in place), then the launch.
    pub fn run(self: *Norm) !void {
        const s = self.shape;
        const st = self.rig.s.handle;
        const rn: c_int = @intCast(s.rows);
        const sn: c_int = @intCast(s.n);
        if (self.inPlace()) try self.rig.put(self.out[0], self.h_bytes);
        const h = self.out[0];
        const x = ptr(self.out[1]);
        const sc = kit.ptr(CF, self.in[3]);
        const y = kit.ptr(CF, self.in[0]);
        const w = kit.ptr(CF, self.in[1]);
        const inj = kit.ptr(C, self.in[2]);
        switch (self.mode) {
            // the launcher takes tf_fn_hc_norm_v for one rank's bf16 streams of 2,560
            .v_branch => try self.l.fn_hc_norm(kit.ptr(C, h), kit.ptr(C, self.in[0]), inj, ptr(h), x, sc, 2, rn, sn, dims, eps, st),
            .v_plain => try self.l.fn_hc_norm(kit.ptr(C, self.in[4]), null, null, null, x, sc, 2, rn, sn, dims, eps, st),
            .moe => try self.l.fn_hc_norm_moe(kit.ptr(C, h), y, w, inj, ptr(h), x, sc, @backingInt(self.k), rn, sn, @intCast(s.k), eps, slots, st),
            .moe_v => try self.l.fn_hc_norm_moe_v(kit.ptr(C, h), y, w, inj, ptr(h), x, sc, rn, sn, eps, slots, st),
        }
        try self.rig.s.synchronize();
    }

    /// tf_moe_combine into the kind (the MoE modes), then tf_fn_hc_norm's 1,024-thread blocks on the same rows.
    pub fn replaced(self: *Norm) ![]const u8 {
        const s = self.shape;
        const st = self.rig.s.handle;
        const kind = @backingInt(self.k);
        var branch = self.in[0];
        if (self.moe()) {
            const y = kit.ptr(CF, self.in[0]);
            try self.l.tf_moe_combine(y, kit.ptr(CF, self.in[1]), ptr(self.base[2]), kind, @intCast(s.rows), slots, @intCast(s.k), st);
            branch = self.base[2];
        }
        if (self.inPlace()) try self.rig.put(self.base[0], self.h_bytes);
        const h = if (self.inPlace()) self.base[0] else self.in[4];
        const add = self.mode != .v_plain;
        var a: hip.launch.Args = .{};
        const ptrs = [_]u64{ h, if (add) branch else 0, if (add) self.in[2] else 0, if (add) h else 0, self.base[1], self.in[3] };
        for (ptrs) |v| try a.add(v);
        for ([_]c_int{ kind, @intCast(s.n), @intCast(s.k) }) |v| try a.add(v);
        try a.add(eps);
        try a.add(@as(c_int, 0)); // no tensor-parallel shares
        try a.add(@as(c_longlong, 0));
        const cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast(s.n), .y = @intCast(s.rows) }, .block = .{ .x = 1024 } };
        try hip.launch.launch(self.l.fnx.hc_norm, cfg, self.rig.s, &a);
        const len = self.h_bytes.len;
        const x = try self.arena.allocator().dupe(u8, try self.rig.bytes(self.base[1], len));
        if (!self.inPlace()) return x;
        return std.mem.concat(self.arena.allocator(), u8, &.{ try self.rig.bytes(self.base[0], len), x });
    }

    pub fn check(self: *Norm, at_: c.Gate) !?[]const u8 {
        if (at_ != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        return mix.classes(kit.elem(self.k), out, try self.replaced(), want);
    }

    pub fn count(self: *Norm) usize {
        return (if (self.inPlace()) @as(usize, 2) else 1) * self.h.len;
    }

    /// The branch at (r, d) in the kind: the v modes' row, or the experts' rows combined in float64 and rounded.
    fn branchAt(self: *const Norm, r: usize, d: usize) f64 {
        const k = self.shape.k;
        if (!self.moe()) return self.br[r * k + d];
        var acc: f64 = 0;
        for (0..slots) |j| acc += self.y[(r * slots + j) * k + d] * self.wts[r * slots + j];
        return kit.round(self.k, acc);
    }

    /// h after the branch, rounded where the kernels round: h + round(branch * inject), rounded to the kind.
    fn residual(self: *const Norm, r: usize, s: usize, d: usize) f64 {
        const v = self.h[self.at(r, s, d)];
        if (self.mode == .v_plain) return v;
        const add = kit.round(self.k, self.branchAt(r, d) * self.inj[r * self.shape.n + s]);
        return kit.round(self.k, v + add);
    }

    fn norms(self: *Norm, acc: c.Accum, order: g.Order, want: []f64, mag: ?[]f64) !void {
        const s = self.shape;
        const t = try self.arena.allocator().alloc(f64, 2 * s.k);
        const off = if (self.inPlace()) self.h.len else 0;
        for (0..s.rows) |r| for (0..s.n) |si| {
            const v = t[0..s.k];
            for (v, t[s.k..], 0..) |*o, *sq, d| {
                o.* = self.residual(r, si, d);
                sq.* = o.* * o.*;
            }
            const ss = rows.sumIn(acc, order, t[s.k..]);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.k)) + eps);
            for (v, 0..) |o, d| {
                const i = self.at(r, si, d);
                if (self.inPlace()) want[i] = o;
                want[off + i] = o * inv * self.scale[si * s.k + d];
                if (mag) |m| {
                    if (self.inPlace()) m[i] = 0;
                    m[off + i] = @abs(want[off + i]) / 2; // x ~ ss^-1/2
                }
            }
        };
    }

    /// h at its rounding points, then rms(h) * scale in float64 (the h rows first when written in place).
    pub fn reference(self: *Norm, want: []f64, mag: []f64) !void {
        return self.norms(.none, .sequential, want, mag);
    }

    pub fn simulate(self: *Norm, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.norms(acc, order, out, null);
    }
};
