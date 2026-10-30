//! The planned decode attention walk under the contract on HIP: score, softmax stats, values and the tiles' sum.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const d = 128; // Nemotron's head dim
const page = 64;

/// The score's lane products and wave tree, softmax's exp, sum and divide, the values' tile and the tiles' sum.
fn depth(s: c.Shape) usize {
    return 22 + 4 + s.k / 256 + 8 + 1 + 2 * 128 + s.k / 128 + 4;
}

const names = [2][2][]const u8{
    .{ "hip planned decode walk planCausal (fp16 cache)", "hip planned decode walk planCausal (bf16 cache)" },
    .{
        "hip planned decode walk planCausalAs (q and rows fp16)",
        "hip planned decode walk planCausalAs (q and rows bf16)",
    },
};

const attend_names = [2][2][]const u8{
    .{ "hip tf_plan_attend_f16 (q and rows fp32)", "hip tf_plan_attend_bf16 (q and rows fp32)" },
    .{ "hip tf_plan_attend_f16 (q and rows fp16)", "hip tf_plan_attend_bf16 (q and rows bf16)" },
};

/// Spans the one-block walk takes: a decode row at 4k keys, a draft window, and a short span on a tp 2 rank.
const attend_served = [_]c.Shape{
    .{ .rows = 1, .n = 32, .k = 4096 },
    .{ .rows = 4, .n = 32, .k = 1000 },
    .{ .rows = 1, .n = 16, .k = 512 },
};

/// The walk, q and rows fp32 or (`act`) the cache's kind as casts around it give them; `attend`: the one-block walk.
pub fn spec(k: kit.Kind, act: bool, attend: bool) c.Spec {
    var s = plain(k);
    s.name = (if (attend) attend_names else names)[@intFromBool(act)][@intFromBool(k == .bf16)];
    if (act) s.out = kit.elem(k);
    if (attend) {
        s.replaces = if (act) "planCausalAs's four launches" else "planCausal's four launches";
        s.served = &attend_served;
        return s;
    }
    if (act) s.replaces = "tf_cast, planCausal and tf_cast";
    return s;
}

fn plain(_: kit.Kind) c.Spec {
    return .{
        .name = "",
        .accum = .f32,
        .depth = depth,
        .out = .f32,
        .tiles = .{ .{}, .{}, .{ .size = 128 } },
        // a decode row at 4k keys, a draft window, and one row at 64k keys on a tp 2 rank (16 query heads, one KV head)
        .served = &.{
            .{ .rows = 1, .n = 32, .k = 4096 },
            .{ .rows = 4, .n = 32, .k = 1000 },
            .{ .rows = 1, .n = 16, .k = 65536 },
        },
    };
}

/// shape.rows rows of shape.n query heads (sixteen a KV head), each over the slot's first shape.k keys.
pub const Walk = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    q: []f64 = &.{}, // [rows][heads][d]
    kc: []f64 = &.{}, // [kv][keys][d]
    vc: []f64 = &.{},
    plan: hip.launches.PlanRef = undefined,
    /// q and the rows in the cache's kind.
    act: bool = false,
    /// The one-block walk (tf_plan_attend_*) in place of the four launches.
    attend: bool = false,
    q32: u64 = 0,
    wide: u64 = 0, // the baseline's fp32 rows
    base: u64 = 0,
    scratch: [3]u64 = @splat(0), // scores, stats, partials
    qd: u64 = 0,
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Walk {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Walk) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Walk) g.Kernel {
        return .{ .spec = spec(self.k, self.act, self.attend), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Walk);

    fn kv(self: *const Walk) usize {
        return @max(self.shape.n / 16, 1);
    }

    pub fn prepare(self: *Walk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.attend and shape.k > 4096) return false;
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.n % @max(shape.n / 16, 1) != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = self.k;
        const keys = shape.k;
        const pages = (keys + page - 1) / page;
        const n_kv = self.kv();
        self.q = try a.alloc(f64, shape.rows * shape.n * d);
        self.kc = try a.alloc(f64, n_kv * pages * page * d);
        self.vc = try a.alloc(f64, self.kc.len);
        for (self.q) |*v| v.* = if (fill == .stress) 0 else kit.round(k, rng.float(f64) * 2 - 1);
        for (self.kc, self.vc) |*x, *y| {
            x.* = kit.round(k, rng.float(f64) * 2 - 1);
            y.* = kit.round(k, rng.float(f64) * 2 - 1);
        }
        if (fill == .stress) self.stress(rng);
        if (fill == .nonfinite) {
            self.q[0] = std.math.nan(f64); // row 0, head 0
            self.kc[(0 * pages * page + keys / 2) * d + 5] = std.math.inf(f64);
            self.vc[(0 * pages * page + keys / 3) * d + 7] = -std.math.inf(f64);
        }
        // the pages under a permuted table, as decode/pages.hpp lays a pool out
        const table = try a.alloc(u32, pages);
        for (table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % pages);
        if (pages % 5 == 0) for (table, 0..) |*p, i| {
            p.* = @intCast(pages - 1 - i);
        };
        const pool_k = try a.alloc(u8, self.kc.len * 2);
        const pool_v = try a.alloc(u8, self.kc.len * 2);
        for (0..n_kv) |h| for (0..pages * page) |pos| for (0..d) |j| {
            const from = (h * pages * page + pos) * d + j;
            const to = ((h * pages + table[pos / page]) * page + pos % page) * d + j;
            kit.put(k, pool_k, to, self.kc[from]);
            kit.put(k, pool_v, to, self.vc[from]);
        };
        const pk = try self.rig.input(pool_k);
        const pv = try self.rig.input(pool_v);
        // one slot: descriptor [cached, kept, k pool, v pool], then its page table
        const words = 4;
        const desc = try a.alloc(u64, words + (pages + 1) / 2);
        @memset(desc, 0);
        desc[2] = pk;
        desc[3] = pv;
        @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. pages * 4], std.mem.sliceAsBytes(table));
        const dev_desc = try self.rig.input(std.mem.sliceAsBytes(desc));
        const descs = try self.rig.input(std.mem.asBytes(&dev_desc));
        const pos = try a.alloc(i32, shape.rows);
        @memset(pos, @intCast(keys - 1));
        const zeros = try a.alloc(i32, shape.rows);
        @memset(zeros, 0);
        const many = [1]i32{@intCast(shape.rows)};
        self.plan = .{ .rows = shape.rows, .slots = 1, .args = .{
            .pos = try self.rig.input(std.mem.sliceAsBytes(pos)),
            .slot = try self.rig.input(std.mem.sliceAsBytes(zeros)),
            .first = try self.rig.input(std.mem.sliceAsBytes(zeros[0..1])),
            .count = try self.rig.input(std.mem.asBytes(&many)),
            .desc = descs,
            .pages = words,
            .pool = @intCast(pages),
        } };
        self.q32 = try self.rig.input(try kit.f32Bytes(a, self.q));
        self.qd = self.q32;
        if (self.act) {
            const qb = try a.alloc(u8, self.q.len * 2);
            for (self.q, 0..) |v, i| kit.put(k, qb, i, v);
            self.qd = try self.rig.input(qb);
        }
        const heads = shape.rows * shape.n;
        const tiles = (keys + 127) / 128;
        const lens = [_]usize{ heads * keys * 4, heads * 2 * 4, heads * tiles * d * 4 };
        for (lens, &self.scratch) |len, *at| at.* = try self.rig.scratch(len);
        self.out = try self.rig.output(heads * d * @as(usize, if (self.act) 2 else 4));
        if (self.act or self.attend) {
            self.wide = try self.rig.scratch(heads * d * 4);
            self.base = try self.rig.scratch(heads * d * 2);
        }
        return true;
    }

    /// Gate 4: q = 0, so every weight is 1 / keys (a power of two); the second half's values undo the first but for 8.
    fn stress(self: *Walk, rng: std.Random) void {
        const keys = self.shape.k;
        const stride = self.kc.len / self.kv();
        for (0..self.kv()) |h| for (0..keys / 2) |key| for (0..d) |j| {
            const v: f64 = @floatFromInt(1 + rng.uintLessThan(u32, 64));
            self.vc[h * stride + key * d + j] = v;
            self.vc[h * stride + (keys / 2 + key) * d + j] = -v + @as(f64, if (key < 8) 1 else 0);
        };
    }

    pub fn run(self: *Walk) !void {
        const s = self.shape;
        const scale: f32 = 1 / @sqrt(@as(f32, d));
        if (self.attend) {
            const kind = @backingInt(self.k);
            const qk: c_int = if (self.act) kind else 0;
            const st = self.rig.s.handle;
            const ran = try self.l.planAttend(
                self.qd,
                qk,
                self.out,
                qk,
                self.plan,
                0,
                s.n,
                self.kv(),
                d,
                s.k,
                scale,
                kind,
                st,
            );
            if (!ran) return error.Refused;
            return self.rig.s.synchronize();
        }
        if (self.act) {
            try self.l.planCausalAs(
                self.qd,
                self.out,
                self.scratch[0],
                self.scratch[1],
                self.scratch[2],
                self.plan,
                0,
                s.n,
                self.kv(),
                d,
                s.k,
                scale,
                @backingInt(self.k),
                true,
                self.rig.s.handle,
            );
            return self.rig.s.synchronize();
        }
        try self.l.planCausal(
            self.qd,
            self.out,
            self.scratch[0],
            self.scratch[1],
            self.scratch[2],
            self.plan,
            0,
            s.n,
            self.kv(),
            d,
            s.k,
            scale,
            @backingInt(self.k),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// The fp32 walk on q widened (the cast's exact values), then its rows cast to the kind; or the four launches.
    pub fn replaced(self: *Walk) ![]const u8 {
        const s = self.shape;
        const scale: f32 = 1 / @sqrt(@as(f32, d));
        const st = self.rig.s.handle;
        if (self.attend) {
            const out = if (self.act) self.base else self.wide;
            try self.l.planCausalAs(
                self.qd,
                out,
                self.scratch[0],
                self.scratch[1],
                self.scratch[2],
                self.plan,
                0,
                s.n,
                self.kv(),
                d,
                s.k,
                scale,
                @backingInt(self.k),
                self.act,
                st,
            );
            return self.rig.bytes(out, s.rows * s.n * d * @as(usize, if (self.act) 2 else 4));
        }
        try self.l.planCausal(
            self.q32,
            self.wide,
            self.scratch[0],
            self.scratch[1],
            self.scratch[2],
            self.plan,
            0,
            s.n,
            self.kv(),
            d,
            s.k,
            scale,
            @backingInt(self.k),
            st,
        );
        const n = s.rows * s.n * d;
        try self.l.tf_cast(@ptrFromInt(self.wide), 0, @ptrFromInt(self.base), @backingInt(self.k), @intCast(n), st);
        return self.rig.bytes(self.base, n * 2);
    }

    pub fn count(self: *Walk) usize {
        return self.shape.rows * self.shape.n * d;
    }

    /// softmax(q k * scale) v in float64 a row and head, each output's magnitude through its scores' and sums' errors.
    pub fn reference(self: *Walk, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const a = self.rig.a;
        const p = try a.alloc(f64, 2 * s.k);
        defer a.free(p);
        const sens = p[s.k..];
        const scale: f64 = @as(f32, 1 / @sqrt(@as(f32, d)));
        const stride = self.kc.len / self.kv();
        for (0..s.rows) |r| for (0..s.n) |h| {
            const qv = self.q[(r * s.n + h) * d ..][0..d];
            const base = (h / (s.n / self.kv())) * stride;
            var top = -std.math.inf(f64);
            for (0..s.k) |key| {
                var dot: f64 = 0;
                var abs: f64 = 0;
                for (0..d) |j| {
                    dot += qv[j] * self.kc[base + key * d + j];
                    abs += @abs(qv[j] * self.kc[base + key * d + j]);
                }
                p[key] = dot * scale;
                sens[key] = 2 + abs * scale;
                top = if (std.math.isNan(p[key]) or std.math.isNan(top)) std.math.nan(f64) else @max(top, p[key]);
            }
            var sum: f64 = 0;
            for (p[0..s.k]) |*x| {
                x.* = @exp(x.* - top);
                sum += x.*;
            }
            const out = want[(r * s.n + h) * d ..][0..d];
            const m = mag[(r * s.n + h) * d ..][0..d];
            @memset(out, 0);
            for (0..s.k) |key| for (0..d) |j| {
                out[j] += p[key] / sum * self.vc[base + key * d + j];
            };
            @memset(m, 0);
            for (0..s.k) |key| for (0..d) |j| {
                m[j] += p[key] / sum * (@abs(self.vc[base + key * d + j]) + @abs(out[j])) * sens[key];
            };
        };
    }

    /// Gate 4's outputs: every weight 1 / keys, the values' products summed in the accumulator and order.
    pub fn simulate(self: *Walk, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const t = try self.rig.a.alloc(f64, s.k);
        defer self.rig.a.free(t);
        const w = 1 / @as(f64, @floatFromInt(s.k));
        const stride = self.kc.len / self.kv();
        for (0..s.rows) |r| for (0..s.n) |h| for (0..d) |j| {
            const base = (h / (s.n / self.kv())) * stride;
            for (t, 0..) |*x, key| x.* = w * self.vc[base + key * d + j];
            out[(r * s.n + h) * d + j] = rows.sumIn(acc, order, t);
        };
    }
};
