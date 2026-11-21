//! dots.mocr's tf_dots_qkv_rope (q|k|v bias, Torch-order RoPE, 12/2 heads of 128) and its control, against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows_ = contract.rows;
const bfr = core.bf16_round;

const hd = 128;
const half = hd / 2;
const h = 12; // dots.mocr's query heads
const hk = 2; // and its KV heads
const heads = h + 2 * hk;
const theta: f32 = 1e6;
const top = 131071; // the last position of max_position_embeddings 131,072
/// Rows 0 to 3's positions: the largest angle first, then HF's first rows and a page boundary.
const special = [_]i32{ top, 0, 1, 4096 };
/// CUDA's cosf and sinf (no fast math) are within 2 fp32 ulps: the reference allows 4 either side.
const trig_ulps = 4;

/// A registration: its name, symbol, and whether it is the planted control (from the contract_controls image).
pub const Form = struct { name: []const u8, sym: [:0]const u8, control: bool = false };

pub const forms = [_]Form{
    .{ .name = "cuda dots_mocr tf_dots_qkv_rope (q|k|v bias, Torch-order RoPE, 12/2 heads of 128)", .sym = "tf_dots_qkv_rope" },
    .{ .name = "control: dots.mocr's q|k|v RoPE with the rotation's two fp32 products rounded once", .sym = "tf_ctl_dots_qkv_rope", .control = true },
};

/// Shape: rows x query heads (n) x the terms each sum adds (k: acc and bias).
const served = [_]c.Shape{ .{ .rows = 1, .n = h, .k = 2 }, .{ .rows = 16, .n = h, .k = 2 }, .{ .rows = 128, .n = h, .k = 2 }, .{ .rows = 1024, .n = h, .k = 2 } };

fn depth(_: c.Shape) usize {
    return 1; // acc + bias: one fp32 add
}

/// Gates 1 and 3 check every output exactly at Torch's rounding points; gate 4 at position 0 sees t, one rounding.
pub fn spec(f: Form) c.Spec {
    return .{ .name = f.name, .accum = .f32, .depth = depth, .out = .bf16, .served = &served };
}

/// x rounded to fp32 (float64 then fp32 equals one fp32 rounding of + and x) and then to bf16, as the kernel stores.
fn b16(x: f64) f64 {
    return bfr.value(bfr.bits(@floatCast(x)));
}

/// The bf16 values a device cosf or sinf can round to: v within trig_ulps fp32 ulps, rounded at both ends.
fn trig(v: f64) [2]f64 {
    const ulp = if (v == 0) 0x1p-149 else std.math.ldexp(@as(f64, 1), std.math.frexp(v).exponent - 24);
    return .{ b16(v - trig_ulps * ulp), b16(v + trig_ulps * ulp) };
}

/// One output of a q or k head in Torch's order (TORCH) or with one rounding of the fp32 sum, from t_j, t_j+64.
fn rotate(a: f64, b: f64, upper: bool, cs: f64, sn: f64, torch: bool) f64 {
    const p = if (upper) b * cs else a * cs; // products of bf16 values: exact in float64 and fp32
    const q = if (upper) a * sn else -b * sn;
    return if (torch) b16(b16(p) + b16(q)) else b16(p + q);
}

/// The sums' accumulator and order (gate 4's simulation), or null: fp32, as the kernel adds.
const Sim = ?struct { acc: c.Accum, order: rows_.Order };

pub const Rope = struct {
    rig: rig_.Rig,
    f: Form,
    fn_: cuda.Function,
    arena: std.heap.ArenaAllocator,
    rows: usize = 0,
    acc: []f32 = &.{},
    bias: []u16 = &.{},
    pos: []i32 = &.{},
    cs: [][2]f64 = &.{}, // a row's bf16 cos candidates for each of the 64 frequencies
    sn: [][2]f64 = &.{},
    dev: struct { acc: u64 = 0, bias: u64 = 0, inv: u64 = 0, pos: u64 = 0, q: u64 = 0, kv: u64 = 0 } = .{},
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Rope) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rope) g.Kernel {
        return .{ .spec = spec(self.f), .ctx = self, .vt = &vt, .control = if (self.f.control) .shapes else null };
    }

    const vt = g.hooks(Rope);

    pub fn prepare(self: *Rope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.n != h or shape.k != 2) return false;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        self.rows = r;
        const w = heads * hd;
        const inv = try a.alloc(f32, half);
        for (inv, 0..) |*v, i| v.* = 1 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / hd); // HF's f32 inv_freq
        self.pos = try a.alloc(i32, r);
        for (self.pos, 0..) |*p, i| p.* = if (fill == .stress) 0 else if (i < special.len) special[i] else @intCast(rng.uintAtMost(u32, top));
        self.acc = try a.alloc(f32, r * w);
        self.bias = try a.alloc(u16, w);
        for (self.acc, 0..) |*v, i| v.* = rng.floatNorm(f32) * (if (i % 37 == 0) @as(f32, 64) else 2);
        for (self.bias) |*v| v.* = bfr.bits(rng.floatNorm(f32) * 0.5);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => stress(self.acc, self.bias, rng),
        }
        self.cs = try a.alloc([2]f64, r * half);
        self.sn = try a.alloc([2]f64, r * half);
        for (0..r) |row| for (0..half) |j| {
            const angle: f32 = @as(f32, @floatFromInt(self.pos[row])) * inv[j]; // __fmul_rn
            self.cs[row * half + j] = trig(@cos(@as(f64, angle)));
            self.sn[row * half + j] = trig(@sin(@as(f64, angle)));
        };
        self.dev.acc = try self.rig.input(std.mem.sliceAsBytes(self.acc));
        self.dev.bias = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.dev.inv = try self.rig.input(std.mem.sliceAsBytes(inv));
        self.dev.pos = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.dev.q = try self.rig.output(r * h * hd * 2);
        self.dev.kv = try self.rig.output(r * 2 * hk * hd * 2);
        return true;
    }

    /// NaN in a q head's first half (row 0) and a v head (row 3); both infinities in row 3's k head; -Inf in a v bias.
    fn plant(self: *Rope) void {
        const w = heads * hd;
        const put = [_]struct { usize, f32 }{
            .{ 3, std.math.nan(f32) },
            .{ w + hd + 70, std.math.inf(f32) },
            .{ 2 * w + h * hd + 5, -std.math.inf(f32) },
            .{ 3 * w + (h + 1) * hd + 20, std.math.inf(f32) },
            .{ 3 * w + (h + 1) * hd + 90, -std.math.inf(f32) },
            .{ 3 * w + (h + hk) * hd + 7, std.math.nan(f32) },
        };
        for (put) |p| if (p[0] < self.acc.len) {
            self.acc[p[0]] = p[1];
        };
        self.bias[(h + hk + 1) * hd + 9] = bfr.bits(-std.math.inf(f32));
    }

    pub fn run(self: *Rope) !void {
        var a: cuda.Args = .{};
        inline for (.{ "acc", "bias", "inv", "pos", "q", "kv" }) |name| a.add(@field(self.dev, name));
        a.add(@as(c_int, h));
        a.add(@as(c_int, hk));
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(self.rows), .y = heads, .z = 1 };
        try cuda.launch.launch(self.fn_, .{ .grid = grid, .block = .{ .x = half } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Rope) usize {
        return self.rows * heads * hd;
    }

    /// Output i's row, head (q, then k, then v) and dim: Q's rows [rows][h], then KV's [rows][k hk | v hk].
    fn at(self: *const Rope, i: usize) struct { row: usize, head: usize, d: usize } {
        const nq = self.rows * h * hd;
        if (i < nq) return .{ .row = i / (h * hd), .head = i / hd % h, .d = i % hd };
        const k = i - nq;
        return .{ .row = k / (2 * hk * hd), .head = h + k / hd % (2 * hk), .d = k % hd };
    }

    /// t = bf16(acc + bias) at a row's column, its sum in `sim`'s accumulator or fp32.
    fn t(self: *const Rope, sim: Sim, row: usize, col: usize) f64 {
        const terms = [2]f64{ self.acc[row * heads * hd + col], bfr.value(self.bias[col]) };
        return b16(if (sim) |s| rows_.sumIn(s.acc, s.order, &terms) else terms[0] + terms[1]);
    }

    /// Output i with cos and sin candidates ci and si, in Torch's order or rounded once.
    fn want(self: *const Rope, sim: Sim, i: usize, ci: usize, si: usize, torch: bool) f64 {
        const o = self.at(i);
        const base = o.head * hd;
        if (o.head >= h + hk) return self.t(sim, o.row, base + o.d);
        const j = o.d % half;
        const x = o.row * half + j;
        return rotate(self.t(sim, o.row, base + j), self.t(sim, o.row, base + j + half), o.d >= half, self.cs[x][ci], self.sn[x][si], torch);
    }

    fn outputs(self: *Rope, sim: Sim, out: []f64, mag: ?[]f64) void {
        for (out, 0..) |*v, i| v.* = self.want(sim, i, 0, 0, true);
        if (mag) |m| for (m, 0..) |*v, i| {
            const o = self.at(i);
            const col = o.row * heads * hd + o.head * hd + o.d;
            v.* = @abs(self.acc[col]) + @abs(bfr.value(self.bias[o.head * hd + o.d]));
        };
    }

    pub fn reference(self: *Rope, want_: []f64, mag: []f64) !void {
        self.outputs(null, want_, mag);
    }

    pub fn simulate(self: *Rope, acc: c.Accum, order: g.Order, out: []f64) !void {
        self.outputs(.{ .acc = acc, .order = order }, out, null);
    }

    /// Every output equals Torch's order at some cos and sin the device can round to; NaN and infinities by class.
    pub fn check(self: *Rope, gate: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        var planted: usize = 0;
        for (0..self.count()) |i| {
            const got = c.read(.bf16, out, i);
            var first: f64 = 0;
            const hit = for (0..4) |k| {
                const w = self.want(null, i, k / 2, k % 2, true);
                if (k == 0) first = w;
                if (c.classOf(got) == c.classOf(w) and (c.classOf(w) != .finite or got == w)) break true;
            } else false;
            if (c.classOf(first) != .finite) planted += 1;
            if (!hit) {
                const o = self.at(i);
                return try std.fmt.bufPrint(&self.msg, "row {d} head {d} dim {d} (position {d}): {e} where Torch's order gives {e}", .{ o.row, o.head, o.d, self.pos[o.row], got, first });
            }
        }
        if (gate == .nonfinite and (planted == 0 or planted == self.count())) return "the planted values leave no class to prove";
        return null;
    }
};

/// Gate 4 at position 0: acc s (2^20 + d) and bias -s 2^20 sum to s d in fp32; fp16 overflows, bf16 drops d (256x).
fn stress(acc: []f32, bias: []u16, rng: std.Random) void {
    const sign = [2]f32{ 1, -1 };
    for (bias, 0..) |*v, col| v.* = bfr.bits(-sign[col % 2] * 0x1p20);
    for (acc, 0..) |*v, i| v.* = sign[i % (heads * hd) % 2] * (0x1p20 + @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 127))));
}

/// The kernel and its control, registered onto `kernels`; the control from the contract_controls image `ctl`.
pub const Set = struct {
    mod: ?cuda.Module = null,
    items: [forms.len]Rope = undefined,
    n: usize = 0,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.dots_mocr);
        for (forms, &self.items) |f, *x| {
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .fn_ = try (if (f.control) ctl else self.mod.?).function(f.sym), .arena = .init(gpu.gpa) };
            self.n += 1;
        }
        for (self.items[0..self.n]) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        for (self.items[0..self.n]) |*x| x.deinit();
        if (self.mod) |*m| m.unload();
    }
};

test "Torch's order differs from one rounding of the rotation's fp32 sum, so the control cannot pass gate 3" {
    var prng = std.Random.DefaultPrng.init(0x646f7473);
    const rng = prng.random();
    var differ: usize = 0;
    for (0..4096) |_| {
        const a = b16(rng.floatNorm(f64) * 2);
        const b = b16(rng.floatNorm(f64) * 2);
        const angle: f32 = @as(f32, @floatFromInt(rng.uintAtMost(u32, top))) * 1e-3;
        const cs = b16(@cos(@as(f64, angle)));
        const sn = b16(@sin(@as(f64, angle)));
        for ([_]bool{ false, true }) |upper| differ += @intFromBool(rotate(a, b, upper, cs, sn, true) != rotate(a, b, upper, cs, sn, false));
    }
    try std.testing.expect(differ > 8192 / 20);
}

test "the bf16 cos and sin candidates: exact at 0, one value away from a rounding edge, two across it" {
    for ([_]f64{ 1, 0 }) |v| for (trig(v)) |x| try std.testing.expect(x == v);
    const edge = 1 + 0x1p-8; // a bf16 tie: the candidates straddle it
    const got = trig(edge);
    try std.testing.expect(got[0] == 1 and got[1] == 1 + 0x1p-7);
}

test "gate 4's stress sums: exact in fp32, an fp16 overflow, and bf16 losing d" {
    const terms = [2]f64{ 0x1p20 + 37, -0x1p20 };
    try std.testing.expectEqual(@as(f64, 37), rows_.sumIn(.f32, .sequential, &terms));
    try std.testing.expect(std.math.isNan(rows_.sumIn(.f16, .sequential, &terms)));
    try std.testing.expectEqual(@as(f64, 0), rows_.sumIn(.bf16, .sequential, &terms));
}

test "served shapes are dots.mocr's layout, and the special positions fit its context" {
    for (served) |s| try std.testing.expect(s.n == h and s.k == 2);
    for (special) |p| try std.testing.expect(p >= 0 and p <= top);
    try std.testing.expectEqual(@as(usize, 2048), heads * hd);
}
