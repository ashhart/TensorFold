//! A DeltaNet prompt's prework in one launch under the contract on HIP: tf_gdn_prompt_prep, the chain's bytes.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const gdn = @import("contract_gdn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const taps = 4; // the conv's kernel (Flash Next's)
const eps: f32 = 1e-6;

/// A q or k output: the conv's four adds, then a head's 128 squares, four a lane, and the wave's xor tree.
fn depth(_: c.Shape) usize {
    return 4 + 4 + 5;
}

/// Flash Next's 48 value heads (16 key heads): a prompt of 40 tokens, one token, and 300.
const served = [_]c.Shape{
    .{ .rows = 40, .n = 48, .k = 128 },
    .{ .rows = 1, .n = 48, .k = 128 },
    .{ .rows = 300, .n = 48, .k = 128 },
};

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = if (k == .bf16) "hip DeltaNet tf_gdn_prompt_prep (bf16 in)" else "hip DeltaNet tf_gdn_prompt_prep (fp16 in)",
        .accum = .f32,
        .depth = depth,
        // silu, the mean's eps, rsqrt and two products; the gate's add, softplus and two exps
        .ops = 16,
        .out = .f32,
        .replaces = "tf_conv_prefill, tf_copy_cols, tf_rms and tf_gdn_gate_prefill",
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = &served,
    };
}

/// shape.rows tokens of a layer's conv input over shape.n value heads: q, k, v, gate, beta and the new conv state.
pub const Prep = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    kw: usize = 0, // key heads x 128
    vw: usize = 0, // value heads x 128
    ch: usize = 0, // the conv's channels: q, k, v
    x: []f64 = &.{}, // [rows][ch], of the kind
    w: []f64 = &.{}, // [ch][taps]
    st: []f64 = &.{}, // [taps - 1][ch]
    qw: []f64 = &.{},
    kn: []f64 = &.{},
    av: []f64 = &.{}, // [rows][heads], of the kind
    bv: []f64 = &.{}, // [rows][heads], of the kind
    a_log: []f64 = &.{},
    dt: []f64 = &.{},
    in: [9]u64 = @splat(0), // x, w, state, q and k norm weights, a, b, a_log, dt_bias
    out: [6]u64 = @splat(0), // q, k, v, gate, beta, new state
    base: [9]u64 = @splat(0), // the chain's six, then its mixed rows and its q and k columns

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Prep {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Prep) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Prep) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Prep);

    /// Each output's elements, in the order the outputs are bound: q, k, v, gate, beta, new state.
    fn lens(self: *const Prep) [6]usize {
        const n = self.shape.rows;
        const heads = self.shape.n;
        return .{ n * self.kw, n * self.kw, n * self.vw, n * heads, n * heads, (taps - 1) * self.ch };
    }

    pub fn prepare(self: *Prep, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 128 or shape.rows == 0 or shape.n == 0 or shape.n % gdn.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const n = shape.rows;
        const heads = shape.n;
        self.kw = heads / gdn.group * 128;
        self.vw = heads * 128;
        self.ch = 2 * self.kw + self.vw;
        self.x = try a.alloc(f64, n * self.ch);
        self.w = try a.alloc(f64, self.ch * taps);
        self.st = try a.alloc(f64, (taps - 1) * self.ch);
        self.qw = try a.alloc(f64, 128);
        self.kn = try a.alloc(f64, 128);
        self.av = try a.alloc(f64, n * heads);
        self.bv = try a.alloc(f64, n * heads);
        self.a_log = try a.alloc(f64, heads);
        self.dt = try a.alloc(f64, heads);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (fill == .stress) self.stress() else self.random(rng);
        if (fill == .nonfinite) self.plant();
        const kinds = [_][]const f64{ self.x, self.av, self.bv };
        var kb: [3][]u8 = undefined;
        for (&kb, kinds) |*b, v| {
            b.* = try a.alloc(u8, v.len * kit.size(self.k));
            for (v, 0..) |e, i| kit.put(self.k, b.*, i, e);
        }
        const ins = [_][]const u8{
            kb[0],
            try kit.f32Bytes(a, self.w),
            try kit.f32Bytes(a, self.st),
            try kit.f32Bytes(a, self.qw),
            try kit.f32Bytes(a, self.kn),
            kb[1],
            kb[2],
            try kit.f32Bytes(a, self.a_log),
            try kit.f32Bytes(a, self.dt),
        };
        for (ins, &self.in) |b, *at| at.* = try self.rig.input(b);
        for (self.lens(), &self.out) |len, *at| at.* = try self.rig.output(len * 4);
        for (self.lens(), self.base[0..6]) |len, *at| at.* = try self.rig.scratch(len * 4);
        self.base[6] = try self.rig.scratch(n * self.ch * 4);
        self.base[7] = try self.rig.scratch(n * self.kw * 4);
        self.base[8] = try self.rig.scratch(n * self.kw * 4);
        return true;
    }

    fn random(self: *Prep, rng: std.Random) void {
        put(rng, self.k, self.x, -2, 2);
        put(rng, self.k, self.av, -3, 3);
        put(rng, self.k, self.bv, -3, 3);
        put(rng, .f32, self.w, -0.5, 0.5);
        put(rng, .f32, self.st, -1, 1);
        put(rng, .f32, self.qw, 1, 1.25);
        put(rng, .f32, self.kn, 1, 1.25);
        put(rng, .f32, self.a_log, 0, 1);
        put(rng, .f32, self.dt, -1, 1);
    }

    /// Gate 4: taps alternate every row, so weights A, B, -A, 1 - B leave the second tap, an integer in [40, 63].
    fn stress(self: *Prep) void {
        const n = self.shape.rows;
        for (0..n + taps - 1) |t| for (0..self.ch) |ch| {
            const v: f64 = @floatFromInt(40 + (ch + t % 2) % 24);
            if (t < taps - 1) self.st[t * self.ch + ch] = v else self.x[(t - (taps - 1)) * self.ch + ch] = v;
        };
        for (0..self.ch) |ch| {
            const w = self.w[ch * taps ..][0..taps];
            w[0] = @floatFromInt(2000 + ch % 40);
            w[1] = @floatFromInt(1990 + ch % 50);
            w[2] = -w[0];
            w[3] = 1 - w[1];
        }
        // silu of an integer past 17 is itself in fp32: the norms sum integer squares below 2^19
        @memset(self.qw, 1);
        @memset(self.kn, 1);
        @memset(self.av, 0);
        @memset(self.bv, 0);
        @memset(self.a_log, 0);
        @memset(self.dt, 0);
    }

    /// NaN, +Inf and -Inf in a q, a k and a v channel; NaN in the conv state; NaN, -Inf in a and +Inf in b.
    fn plant(self: *Prep) void {
        const heads = self.shape.n;
        if (self.shape.rows < 3 or heads < 4) return;
        self.x[0] = std.math.nan(f64);
        self.x[self.ch + self.kw + 5] = std.math.inf(f64);
        self.x[2 * self.ch + 2 * self.kw + 7] = -std.math.inf(f64);
        self.st[self.ch + 2 * self.kw + 9] = std.math.nan(f64);
        self.av[1] = std.math.nan(f64);
        self.bv[heads + 2] = std.math.inf(f64);
        self.av[2 * heads + 3] = -std.math.inf(f64);
    }

    pub fn run(self: *Prep) !void {
        const C = ?*const anyopaque;
        const CF = ?[*]const f32;
        const F = ?[*]f32;
        const in = self.in;
        const out = self.out;
        try self.l.tf_gdn_prompt_prep(
            kit.ptr(C, in[0]),
            @backingInt(self.k),
            kit.ptr(CF, in[1]),
            kit.ptr(CF, in[2]),
            kit.ptr(F, out[5]),
            kit.ptr(F, out[0]),
            kit.ptr(F, out[1]),
            kit.ptr(F, out[2]),
            kit.ptr(CF, in[3]),
            kit.ptr(CF, in[4]),
            eps,
            @intCast(self.shape.rows),
            @intCast(self.kw),
            @intCast(self.vw),
            taps,
            kit.ptr(C, in[5]),
            kit.ptr(C, in[6]),
            kit.ptr(CF, in[7]),
            kit.ptr(CF, in[8]),
            kit.ptr(F, out[3]),
            kit.ptr(F, out[4]),
            @intCast(self.shape.n),
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// The chain the launch replaces: conv, three column copies, the q and k norms, the gates.
    pub fn replaced(self: *Prep) ![]const u8 {
        const l = self.l;
        const P = ?*anyopaque;
        const C = ?*const anyopaque;
        const CF = ?[*]const f32;
        const F = ?[*]f32;
        const s = self.rig.s.handle;
        const in = self.in;
        const b = self.base;
        const n: c_int = @intCast(self.shape.rows);
        const ch: c_longlong = @intCast(self.ch);
        const kw: c_int = @intCast(self.kw);
        const kind = @backingInt(self.k);
        try l.tf_conv_prefill(kit.ptr(C, in[0]), kind, kit.ptr(CF, in[1]), kit.ptr(CF, in[2]), kit.ptr(F, b[6]), kit.ptr(F, b[5]), n, @intCast(self.ch), taps, s);
        try l.tf_copy_cols(kit.ptr(C, b[6]), ch, 0, kit.ptr(P, b[7]), 0, n, kw, s);
        try l.tf_copy_cols(kit.ptr(C, b[6]), ch, kw, kit.ptr(P, b[8]), 0, n, kw, s);
        try l.tf_copy_cols(kit.ptr(C, b[6]), ch, 2 * kw, kit.ptr(P, b[2]), 0, n, @intCast(self.vw), s);
        const heads: c_int = n * @divExact(kw, 128);
        try l.tf_rms(kit.ptr(C, b[7]), kit.ptr(CF, in[3]), kit.ptr(P, b[0]), 0, heads, 128, eps, s);
        try l.tf_rms(kit.ptr(C, b[8]), kit.ptr(CF, in[4]), kit.ptr(P, b[1]), 0, heads, 128, eps, s);
        try l.tf_gdn_gate_prefill(kit.ptr(C, in[5]), kit.ptr(C, in[6]), kind, kit.ptr(CF, in[7]), kit.ptr(CF, in[8]), kit.ptr(F, b[3]), kit.ptr(F, b[4]), n, @intCast(self.shape.n), s);
        const a = self.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        for (self.lens(), b[0..6]) |len, at| try all.appendSlice(a, try self.rig.bytes(at, len * 4));
        return all.items;
    }

    pub fn count(self: *Prep) usize {
        var n: usize = 0;
        for (self.lens()) |len| n += len;
        return n;
    }

    /// Conv tap t of channel ch: the state's rows, then x's.
    fn window(self: *const Prep, t: usize, ch: usize) f64 {
        return if (t < taps - 1) self.st[t * self.ch + ch] else self.x[(t - (taps - 1)) * self.ch + ch];
    }

    /// The conv's sum at row r of channel ch (in `sim`'s accumulator and order when given) and its terms' magnitude.
    fn conv(self: *const Prep, r: usize, ch: usize, sim: ?gdn.Sim) struct { f64, f64 } {
        var t: [taps]f64 = undefined;
        var abs: f64 = 0;
        for (&t, 0..) |*o, j| {
            o.* = self.window(r + j, ch) * self.w[ch * taps + j];
            abs += @abs(o.*);
        }
        if (sim) |how| return .{ rows.sumIn(how[0], how[1], &t), abs };
        return .{ t[0] + t[1] + t[2] + t[3], abs };
    }

    fn outputs(self: *Prep, want: []f64, mag: []f64, sim: ?gdn.Sim) !void {
        const n = self.shape.rows;
        const heads = self.shape.n;
        const kh = self.kw / 128;
        const len = self.lens();
        var at: [6]usize = undefined;
        var off: usize = 0;
        for (&at, len) |*o, l| {
            o.* = off;
            off += l;
        }
        @memset(mag, 0);
        var vals: [128]f64 = undefined;
        var sq: [128]f64 = undefined;
        for (0..n) |r| {
            // q heads then k heads: silu of the conv, then the head's RMS norm times its weight
            for (0..2 * kh) |hd| {
                for (&vals, &sq, 0..) |*v, *q, i| {
                    v.* = kit.silu(self.conv(r, hd * 128 + i, sim)[0]);
                    q.* = v.* * v.*;
                }
                const ss = if (sim) |how| rows.sumIn(how[0], how[1], &sq) else blk: {
                    var x: f64 = 0;
                    for (sq) |e| x += e;
                    break :blk x;
                };
                const inv = 1 / @sqrt(ss / 128 + @as(f64, eps));
                const is_q = hd < kh;
                const nw = if (is_q) self.qw else self.kn;
                const base = (if (is_q) at[0] else at[1]) + (r * kh + hd % kh) * 128;
                for (vals, nw, 0..) |v, wv, i| {
                    want[base + i] = v * inv * wv;
                    mag[base + i] = @abs(want[base + i]);
                }
            }
            for (0..self.vw) |i| {
                const sum, const abs = self.conv(r, 2 * self.kw + i, sim);
                want[at[2] + r * self.vw + i] = kit.silu(sum);
                mag[at[2] + r * self.vw + i] = abs * kit.siluSlope(sum);
            }
            for (0..heads) |h| {
                const i = r * heads + h;
                const xa = self.av[i] + self.dt[h];
                const sp = if (xa > 20) xa else std.math.log1p(@exp(xa));
                want[at[3] + i] = @exp(-@exp(self.a_log[h]) * sp);
                want[at[4] + i] = 1 / (1 + @exp(-self.bv[i]));
            }
        }
        for (0..taps - 1) |t| for (0..self.ch) |ch| {
            want[at[5] + t * self.ch + ch] = self.window(n + t, ch);
        };
    }

    pub fn reference(self: *Prep, want: []f64, mag: []f64) !void {
        return self.outputs(want, mag, null);
    }

    pub fn simulate(self: *Prep, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        return self.outputs(out, mag, .{ acc, order });
    }
};

/// Values uniform in [lo, hi), exact in the kind (fp16 normals only, as bf16 values are).
fn put(rng: std.Random, k: kit.Kind, v: []f64, lo: f64, hi: f64) void {
    for (v) |*x| {
        x.* = kit.round(k, lo + rng.float(f64) * (hi - lo));
        if (k == .f16 and @abs(x.*) < 0x1p-14) x.* = 0;
    }
}
