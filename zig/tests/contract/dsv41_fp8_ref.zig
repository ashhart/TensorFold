//! DeepSeek V4.1 FP8 dense rows for the contract: inputs, the kernel's fp32 order as oracle, float64 reference.
const std = @import("std");
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;

pub fn e4m3(code: u8) f32 {
    const e: i32 = (code >> 3) & 15;
    const m: f32 = @floatFromInt(code & 7);
    const v: f32 = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(1 + m / 8, e - 7);
    return if (code & 0x80 != 0) -v else v;
}

pub fn e8m0(code: u8) f32 {
    return std.math.ldexp(@as(f32, 1), @as(i32, code) - 127);
}

/// Codes of the E4M3 values 1..8 (integers): gate 4's weights, so every product and sum is exact.
fn integerCodes(out: *[8]u8) void {
    var n: usize = 0;
    for (0..0x7f) |code| {
        const v = e4m3(@intCast(code));
        if (v >= 1 and v <= 8 and v == @floor(v) and n < 8) {
            out[n] = @intCast(code);
            n += 1;
        }
    }
    std.debug.assert(n == 8);
}

pub fn levelsOf(groups: usize) usize {
    var n: usize = 1;
    while (n < groups) n *= 2;
    return n;
}

/// Longest chain of roundings: the 32 FMAs of a group, the tree's levels and the output's own.
pub fn depth(s: c.Shape) usize {
    return 32 + std.math.log2_int_ceil(usize, levelsOf(s.k / 32)) + 1;
}

pub const Out = struct { oracle: f32, exact: f64, mag: f64 };

/// Host inputs: x [rows, k] bf16, codes [n, k] E4M3, scales [n/32, k/32] E8M0 (one a 32x32 block).
pub const Fp8 = struct {
    a: std.mem.Allocator,
    shape: c.Shape,
    x: []u16,
    codes: []u8,
    scales: []u8,
    wf: []f32,

    pub fn init(a: std.mem.Allocator, shape: c.Shape) !Fp8 {
        if (shape.k % 32 != 0 or shape.n % 32 != 0) return error.BlockSplitsShape;
        const x = try a.alloc(u16, shape.rows * shape.k);
        errdefer a.free(x);
        const codes = try a.alloc(u8, shape.n * shape.k);
        errdefer a.free(codes);
        const scales = try a.alloc(u8, shape.n / 32 * (shape.k / 32));
        errdefer a.free(scales);
        return .{ .a = a, .shape = shape, .x = x, .codes = codes, .scales = scales, .wf = try a.alloc(f32, shape.n * shape.k) };
    }

    pub fn deinit(f: Fp8) void {
        f.a.free(f.x);
        f.a.free(f.codes);
        f.a.free(f.scales);
        f.a.free(f.wf);
    }

    pub fn outputs(f: Fp8) usize {
        return f.shape.rows * f.shape.n;
    }

    /// The weights as the kernel forms them: the code's value times its block's power of two, exact in fp32.
    pub fn dequant(f: Fp8) void {
        const k = f.shape.k;
        for (0..f.shape.n) |j| for (0..k) |i| {
            f.wf[j * k + i] = e4m3(f.codes[j * k + i]) * e8m0(f.scales[j / 32 * (k / 32) + i / 32]);
        };
    }

    pub fn fill(f: Fp8, mode: rows.Fill, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (f.codes) |*q| {
            const b = rng.int(u8);
            q.* = if (b & 0x7f == 0x7f) b & 0xfe else b; // 0x7f is NaN
        }
        for (f.scales) |*s| s.* = 119 + rng.uintLessThan(u8, 7);
        for (f.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
        switch (mode) {
            .random => {},
            .nonfinite => {
                const k = f.shape.k;
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |p, v| if (p[0] < f.shape.rows) {
                    f.x[p[0] * k + p[1]] = c.toBf16(v);
                };
            },
            .stress => f.stress(rng),
        }
        f.dequant();
    }

    /// Gate 4's inputs: scale 1, integer weights, K's second half mirroring the first negated: exact fp32, tiny result.
    fn stress(f: Fp8, rng: std.Random) void {
        const k = f.shape.k;
        const h = k / 2;
        var ints: [8]u8 = undefined;
        integerCodes(&ints);
        @memset(f.scales, 127);
        const xmax: u32 = @min(255, ((1 << 24) - 1) / (8 * @as(u32, @intCast(k))));
        for (0..f.shape.n) |j| for (0..h) |i| {
            const q = ints[rng.uintLessThan(usize, 8)];
            f.codes[j * k + i] = q;
            f.codes[j * k + h + i] = q | 0x80;
        };
        for (0..f.shape.rows) |r| for (0..h) |i| {
            const v = 1 + rng.uintLessThan(u32, xmax);
            f.x[r * k + i] = c.toBf16(@floatFromInt(v));
            f.x[r * k + h + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
        };
        if (k % 2 == 1) unreachable;
    }

    fn xv(f: Fp8, r: usize, i: usize) f32 {
        return c.fromBf16(f.x[r * f.shape.k + i]);
    }

    /// Output (r, j): the fp32 order (32-FMA groups, then the pair tree over padded groups), float64 sum, magnitude.
    pub fn one(f: Fp8, r: usize, j: usize, leaves: []f32) Out {
        const k = f.shape.k;
        var exact: f64 = 0;
        var mag: f64 = 0;
        @memset(leaves, 0);
        for (0..k / 32) |g| {
            var acc: f32 = 0;
            for (g * 32..(g + 1) * 32) |i| {
                const w = f.wf[j * k + i];
                const x = f.xv(r, i);
                acc = @mulAdd(f32, x, w, acc);
                exact += @as(f64, x) * w;
                mag += @abs(@as(f64, x) * w);
            }
            leaves[g] = acc;
        }
        var s: usize = 1;
        while (s < leaves.len) : (s *= 2) {
            var i: usize = 0;
            while (i < leaves.len) : (i += 2 * s) leaves[i] += leaves[i + s];
        }
        return .{ .oracle = leaves[0], .exact = exact, .mag = mag };
    }

    /// The fp8_rows order's outputs as `want` (bf16-rounded if asked); it must sit inside the fp32 bound of float64.
    pub fn reference(f: Fp8, want: []f64, mag: []f64, bf16: bool) !void {
        const leaves = try f.a.alloc(f32, levelsOf(f.shape.k / 32));
        defer f.a.free(leaves);
        const d: f64 = @floatFromInt(depth(f.shape));
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const o = f.one(r, j, leaves);
            const at = r * f.shape.n + j;
            if (std.math.isFinite(o.exact) and @abs(@as(f64, o.oracle) - o.exact) > c.gamma(d, 0x1p-24) * o.mag + 0x1p-149) return error.OracleDrift;
            want[at] = if (bf16) c.fromBf16(c.toBf16(o.oracle)) else o.oracle;
            mag[at] = o.mag;
        };
    }

    /// Float64 magnitudes only, for inputs whose reference values come from elsewhere.
    pub fn magnitudes(f: Fp8, mag: []f64) void {
        const k = f.shape.k;
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            var m: f64 = 0;
            for (0..k) |i| m += @abs(@as(f64, f.xv(r, i)) * f.wf[j * k + i]);
            mag[r * f.shape.n + j] = m;
        };
    }

    /// On gate 4's inputs: each output as an accumulator leaves it, the closer of a flat chain and 32-term groups.
    pub fn simulate(f: Fp8, acc: c.Accum, order: rows.Order, out: []f64) !void {
        const k = f.shape.k;
        const t = try f.a.alloc(f64, k);
        defer f.a.free(t);
        const gs = try f.a.alloc(f64, k / 32);
        defer f.a.free(gs);
        const leaves = try f.a.alloc(f32, levelsOf(k / 32));
        defer f.a.free(leaves);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const want = f.one(r, j, leaves).exact;
            for (0..k) |i| t[i] = @as(f64, f.xv(r, i)) * f.wf[j * k + i];
            const whole = rows.sumIn(acc, order, t);
            for (gs, 0..) |*y, g| y.* = rows.sumIn(acc, .sequential, t[g * 32 ..][0..32]);
            const grouped = rows.sumIn(acc, order, gs);
            out[r * f.shape.n + j] = if (!std.math.isFinite(whole) or @abs(grouped - want) < @abs(whole - want)) grouped else whole;
        };
    }
};

/// A float64 .npy (version 1) payload: the values after its header.
pub fn npyF64(bytes: []const u8, out: []f64) !void {
    if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) return error.BadNpy;
    const start = 10 + std.mem.readInt(u16, bytes[8..10], .little);
    if (bytes.len != start + 8 * out.len) return error.BadNpy;
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u64, bytes[start + 8 * i ..][0..8], .little));
}

/// The release's mtp.0.attn.wkv rows 0:64, cols 0:256 (real bytes) and numpy's float64 matvec against it.
pub const fixture = struct {
    pub const shape: c.Shape = .{ .rows = 1, .n = 64, .k = 256 };
    pub const raw = @embedFile("fixtures/dsv41/fp8_wkv_raw.bin");
    pub const scale = @embedFile("fixtures/dsv41/fp8_wkv_scale.bin");
    pub const x_npy = @embedFile("fixtures/dsv41/matvec_x_bf16.npy");
    pub const ref_npy = @embedFile("fixtures/dsv41/fp8_wkv_matvec_ref.npy");

    pub fn load(f: Fp8, want: ?[]f64) !void {
        var x: [256]f64 = undefined;
        try npyF64(x_npy, &x);
        for (f.x, x) |*h, v| h.* = c.toBf16(@floatCast(v));
        @memcpy(f.codes, raw);
        @memcpy(f.scales, scale);
        f.dequant();
        if (want) |w| try npyF64(ref_npy, w);
    }
};
