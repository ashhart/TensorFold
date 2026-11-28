//! engram.metal's lookup norm and table gather against float64 at the kernels' bf16 points (metal_engram.zig).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const text = @import("metal_hy4_rows_text.zig");

const c = contract.spec;
const g = contract.gates;
const bf = h.fl.bf;
const Range = h.fl.Range;
const u = h.fl.u32_ulp;

/// The lookup's heads (engram.zig's columns: two n-gram orders of 8 heads) and each head's width (768 / 8).
const heads = h.hy.engram.columns;
const width = 96;
const cells = heads * width;
const eps = 0x1p-23;
comptime {
    std.debug.assert(heads == 16);
}

/// The served passes: DFly2's window first, one decode row, a prompt chunk.
const shapes = [_]c.Shape{ .{ .rows = h.window_rows, .n = cells, .k = width }, .{ .rows = 1, .n = cells, .k = width }, .{ .rows = h.prompt_rows, .n = cells, .k = width } };

/// A head's squares: three a lane in turn, then simd_sum over 32 lanes (31 adds at worst).
fn sqDepth(_: c.Shape) usize {
    return 2 + 31;
}

/// metal's clamp: max then min, so a NaN becomes the low end.
fn clampM(v: f64, lo: f64, hi: f64) f64 {
    return @min(@max(v, lo), hi);
}

fn rne(v: f64) f64 {
    const f = @floor(v);
    const d = v - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// E4M3FN as hy4_e4m3 rounds: ties to even at the value's own step (subnormal steps of 2^-9), saturating at 448.
fn e4m3(x: f64) f64 {
    const a = @abs(x);
    if (a == 0) return x;
    const e: i32 = @max(@as(i32, @intCast((@as(u64, @bitCast(a)) >> 52) & 0x7ff)) - 1023, -6);
    const step: f64 = @bitCast(@as(u64, @intCast(e - 3 + 1023)) << 52);
    return std.math.copysign(@min(448, rne(a / step) * step), x);
}

fn inv(ss: f64) f64 {
    return 1 / @sqrt(ss / width + eps);
}

pub const Lookup = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    fp8: bool,
    fault: ?text.Fault,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    raw: []u16 = &.{},
    gamma: []u16 = &.{},
    in: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn base(fp8: bool) []const u8 {
        return if (fp8) "metal hy4 engram tf_hy4_engram_lookup fp8" else "metal hy4 engram tf_hy4_engram_lookup bf16";
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, fp8: bool, fault: ?text.Fault) !Lookup {
        const src = try text.engram(a, fault);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try text.pipeline(device, src, "tf_hy4_engram_lookup"), .fp8 = fp8, .fault = fault };
    }
    pub fn deinit(t: *Lookup) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Lookup) void {
        t.a.free(t.raw);
        t.a.free(t.gamma);
        t.raw = &.{};
        t.gamma = &.{};
    }

    /// The served checkpoint's engram_fp8_embedding is true; a config without it takes the plain norm.
    pub fn kernel(t: *Lookup) g.Kernel {
        t.name = h.fl.named(base(t.fp8), if (t.fault) |f| f.drops else null, &t.label);
        const s: c.Spec = .{ .name = t.name, .accum = .f32, .depth = sqDepth, .out = .bf16, .served = &shapes };
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.fault != null) .shapes else null };
    }
    const vt = g.hooks(Lookup);

    pub fn prepare(t: *Lookup, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != cells or shape.k != width) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.raw = try t.a.alloc(u16, shape.rows * cells);
        t.gamma = try t.a.alloc(u16, cells);
        for (t.gamma) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        for (t.raw, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 8 - 4) * @as(f32, if (i % 37 == 5) 16 else 1));
        if (fill == .stress) for (0..shape.rows * heads) |hd| {
            // integer squares exact in fp32 (under 2^23) that overflow fp16; fp8 heads hold E4M3 integers up to 448
            const x = t.raw[hd * width ..][0..width];
            for (x) |*v| {
                const sign: f32 = if (rng.boolean()) 1 else -1;
                const m: f32 = if (t.fp8) @floatFromInt((8 + rng.uintLessThan(u32, 8)) << @intCast(1 + rng.uintLessThan(u32, 5))) else @floatFromInt(20 + rng.uintLessThan(u32, 21));
                v.* = c.toBf16(sign * m);
            }
            // 448 the largest, so the fp8 scale is 1 and the round trip keeps every value
            if (t.fp8) x[rng.uintLessThan(usize, width)] = c.toBf16(if (rng.boolean()) 448 else -448);
        };
        if (fill == .nonfinite and shape.rows >= 4) {
            const at = [_][3]usize{ .{ 0, 0, 5 }, .{ 1, 1, 40 }, .{ 2, 2, 95 }, .{ 3, 3, 0 }, .{ 3, 3, 64 } };
            const put = [_]f32{ h.fl.planted[0], h.fl.planted[1], h.fl.planted[2], h.fl.planted[1], h.fl.planted[2] };
            for (at, put) |p, v| t.raw[(p[0] * heads + p[1]) * width + p[2]] = c.toBf16(v);
        }
        t.in[0] = try t.rig.input(std.mem.sliceAsBytes(t.raw));
        t.in[1] = try t.rig.input(std.mem.sliceAsBytes(t.gamma));
        t.out = try t.rig.output(shape.rows * cells * 2);
        return true;
    }

    /// metal_engram.zig's forward: a group of 32 threads for each head and row, fp8 as a uint.
    pub fn run(t: *Lookup) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in[0].buf, t.in[0].off, 0);
        p.e.setBuffer(t.in[1].buf, t.in[1].off, 1);
        p.e.setBuffer(t.out.buf, t.out.off, 2);
        p.e.setValue(@as(u32, @intFromBool(t.fp8)), 3);
        p.e.dispatchGroups(mtl.Size.of(heads, t.shape.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Lookup) usize {
        return t.shape.rows * cells;
    }

    /// Head `hd` of row `r` as the norm reads it: the raw bf16 row, or its fp8 round trip at the kernel's points.
    fn head(t: *const Lookup, r: usize, hd: usize, xs: *[width]f64) void {
        for (xs, t.raw[(r * heads + hd) * width ..][0..width]) |*x, w| x.* = c.fromBf16(w);
        if (!t.fp8) return;
        var most: f64 = 0;
        for (xs) |x| most = @max(most, @abs(x));
        const s = bf(@max(most, bf(1e-12)) / 448);
        for (xs) |*x| {
            const q = bf(e4m3(clampM(bf(x.* / s), -448, 448)) * s);
            x.* = bf(x.* + bf(q - x.*));
        }
    }

    fn gain(t: *const Lookup, hd: usize, j: usize) f64 {
        return c.fromBf16(t.gamma[hd * width + j]);
    }

    pub fn reference(t: *Lookup, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        var xs: [width]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..heads) |hd| {
            t.head(r, hd, &xs);
            var ss: f64 = 0;
            for (xs) |x| ss += x * x;
            const iv = inv(ss);
            for (xs, 0..) |x, j| want[(r * heads + hd) * width + j] = bf(x * iv * t.gain(hd, j));
        };
    }

    /// Each output's range: the squares within gamma(depth), the mean, epsilon and rsqrt, x * inv, * gamma, bf16.
    pub fn intervals(t: *Lookup, low: []f64, high: []f64) !void {
        const gam = if (t.fill == .stress) 0 else c.gamma(@floatFromInt(sqDepth(t.shape)), u);
        var xs: [width]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..heads) |hd| {
            t.head(r, hd, &xs);
            var ss: f64 = 0;
            for (xs) |x| ss += x * x;
            const iv = Range{ .lo = inv(ss * (1 + gam)) * (1 - 12 * u), .hi = inv(ss * (1 - gam)) * (1 + 12 * u) };
            for (xs, 0..) |x, j| {
                const o = Range.point(x).mul(iv).fp32().times(t.gain(hd, j)).fp32().bf16();
                low[(r * heads + hd) * width + j] = o.lo;
                high[(r * heads + hd) * width + j] = o.hi;
            }
        };
    }

    pub fn bounds(t: *Lookup, want: []const f64, _: []const f64, allow: []f64) !void {
        try h.fl.allowFrom(t.a, t, want, allow);
    }

    pub fn simulate(t: *Lookup, acc: c.Accum, order: g.Order, out: []f64) !void {
        var xs: [width]f64 = undefined;
        var sq: [width]f64 = undefined;
        for (0..t.shape.rows) |r| for (0..heads) |hd| {
            t.head(r, hd, &xs);
            for (&sq, xs) |*q, x| q.* = x * x;
            const iv = inv(contract.rows.sumIn(acc, order, &sq));
            for (xs, 0..) |x, j| out[(r * heads + hd) * width + j] = bf(x * iv * t.gain(hd, j));
        };
    }
};

/// Engram table rows the gather reads (the release's tables hold ~165M rows a layer; the gates need distinct rows).
const table = 4096;
/// The tables' MLX groups.
const tgroup = 32;

pub const Gather = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    bits: u8,
    fault: ?text.Fault,
    label: [192]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    codes: []u8 = &.{},
    scales: []u16 = &.{},
    biases: []u16 = &.{},
    rows: []u32 = &.{},
    in: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const gshapes = [_]c.Shape{ .{ .rows = h.window_rows, .n = cells, .k = 1 }, .{ .rows = 1, .n = cells, .k = 1 }, .{ .rows = h.prompt_rows, .n = cells, .k = 1 } };

    pub fn base(bits: u8) []const u8 {
        return if (bits == 4) "metal hy4 engram tf_hy4_engram_gather q4" else "metal hy4 engram tf_hy4_engram_gather q8";
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bits: u8, fault: ?text.Fault) !Gather {
        const src = try text.engram(a, fault);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try text.pipeline(device, src, "tf_hy4_engram_gather"), .bits = bits, .fault = fault };
    }
    pub fn deinit(t: *Gather) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Gather) void {
        inline for (.{ "codes", "scales", "biases", "rows" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }

    /// Rows: tokens, n: their 16 columns' 96 cells; each cell bf16(fp32(code * scale) + bias), exact.
    pub fn kernel(t: *Gather) g.Kernel {
        t.name = h.fl.named(base(t.bits), if (t.fault) |f| f.drops else null, &t.label);
        const s: c.Spec = .{ .name = t.name, .accum = .none, .depth = h.fl.noSums, .out = .bf16, .served = &gshapes };
        return .{ .spec = s, .ctx = t, .vt = &vt, .control = if (t.fault != null) .shapes else null };
    }
    const vt = g.hooks(Gather);

    pub fn prepare(t: *Gather, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != cells or shape.k != 1) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const per = width / tgroup;
        t.codes = try t.a.alloc(u8, table * width);
        t.scales = try t.a.alloc(u16, table * per);
        t.biases = try t.a.alloc(u16, table * per);
        t.rows = try t.a.alloc(u32, shape.rows * heads);
        const top = @as(u32, 1) << @intCast(t.bits);
        for (t.codes) |*q| q.* = @intCast(rng.uintLessThan(u32, top));
        for (t.scales, t.biases) |*s, *b| {
            const sc = 0.0005 + rng.float(f32) * 0.01;
            s.* = c.toBf16(sc);
            b.* = c.toBf16(-sc * rng.float(f32) * @as(f32, @floatFromInt(top)));
        }
        for (t.rows) |*r| r.* = rng.uintLessThan(u32, table);
        // tokens 0-3 read rows 0-3 first (the planted ones), the last cell the table's last row
        for (0..@min(4, shape.rows)) |i| t.rows[i * heads] = @intCast(i);
        t.rows[t.rows.len - 1] = table - 1;
        if (fill == .nonfinite) {
            t.scales[0 * per + 0] = c.toBf16(h.fl.planted[0]);
            t.biases[1 * per + 1] = c.toBf16(h.fl.planted[1]);
            t.biases[2 * per + 2] = c.toBf16(h.fl.planted[2]);
            t.scales[3 * per + 0] = c.toBf16(h.fl.planted[1]);
            t.biases[3 * per + 2] = c.toBf16(h.fl.planted[2]);
        }
        // the table's codes as MLX's little-endian bit stream: a byte each at 8 bits, two to a byte (low first) at 4
        if (t.bits == 8) t.in[0] = try t.rig.input(t.codes) else {
            const words = try t.a.alloc(u8, t.codes.len / 2);
            defer t.a.free(words);
            for (words, 0..) |*v, i| v.* = t.codes[2 * i] | t.codes[2 * i + 1] << 4;
            t.in[0] = try t.rig.input(words);
        }
        t.in[1] = try t.rig.input(std.mem.sliceAsBytes(t.scales));
        t.in[2] = try t.rig.input(std.mem.sliceAsBytes(t.biases));
        t.in[3] = try t.rig.input(std.mem.sliceAsBytes(t.rows));
        t.out = try t.rig.output(shape.rows * cells * 2);
        return true;
    }

    /// metal_engram.zig's forward: a thread a cell in groups of 256, (bits, cells) as its uint2.
    pub fn run(t: *Gather) !void {
        const n = t.shape.rows * cells;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.in, 0..) |r, i| p.e.setBuffer(r.buf, r.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 4);
        p.e.setValue([2]u32{ t.bits, @intCast(n) }, 5);
        p.e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Gather) usize {
        return t.shape.rows * cells;
    }

    /// code * scale is exact in fp32 (at most 16 bits), so the cell is fp32's sum with the bias, rounded to bf16.
    pub fn reference(t: *Gather, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        const per = width / tgroup;
        for (want, 0..) |*w, i| {
            const row: usize = t.rows[i / width];
            const j = i % width;
            const q: f64 = @floatFromInt(t.codes[row * width + j]);
            w.* = bf(q * c.fromBf16(t.scales[row * per + j / tgroup]) + c.fromBf16(t.biases[row * per + j / tgroup]));
        }
    }

    /// Exact: every cell must give float64's bf16 bits.
    pub fn bounds(_: *Gather, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }
};
