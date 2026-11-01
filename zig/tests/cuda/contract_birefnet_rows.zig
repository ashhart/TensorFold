//! tf_br_gate, tf_br_head, tf_br_pool, tf_br_resize and tf_br_tiles under the contract against float64.
const std = @import("std");
const cm = @import("contract_birefnet_common.zig");

const c = cm.c;
const g = cm.g;
const Range = cm.Range;
const Ends = cm.Ends;
const ks = cm.ks;
const uniform = cm.uniform;
const plant = cm.plant;

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

// --- tf_br_gate: x *= sigmoid(g . w + b), the gradient attention's gate ---------------------------------------------

pub const Gate = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shape: c.Shape = undefined,
    gv: []f32 = &.{},
    w: [16]f32 = undefined,
    b: [1]f32 = undefined,
    x: []f32 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    fn depth(_: c.Shape) usize {
        return 16 + 1;
    }

    /// Rows: pixels; n: channels gated; k: the 16 gate channels. The output is x gated, x given as an input copy.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = 1 + cm.exp_ulps + cm.div_ulps + 2, .out = .f32, .tiles = .{ .{ .size = 4 }, .{ .size = 64 }, .{} }, .served = &.{ .{ .rows = 1024, .n = 1536, .k = 16 }, .{ .rows = 4096, .n = 768, .k = 16 }, .{ .rows = 999, .n = 384, .k = 16 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "16 terms into a sigmoid: no input makes fp16 sums miss by 100x and keep the gate's fp32 value" } };
    }

    pub fn deinit(t: *Gate) void {
        t.a.free(t.gv);
        t.a.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *Gate, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Gate);

    pub fn fill(t: *Gate, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 16) return false;
        t.a.free(t.gv);
        t.a.free(t.x);
        t.shape = shape;
        t.gv = try t.a.alloc(f32, shape.rows * 16);
        t.x = try t.a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.gv, 0, 2);
        uniform(rng, &t.w, -1, 1);
        uniform(rng, &t.b, -1, 1);
        uniform(rng, t.x, -3, 3);
        if (mode == .nonfinite) {
            plant(t.x, shape.n);
            plant(t.gv, 16);
        }
        return true;
    }

    pub fn prepare(t: *Gate, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.gv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(&t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(&t.b));
        // the kernel gates in place: the output starts as x
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Gate) !void {
        try cm.set(&t.rig, t.out, std.mem.sliceAsBytes(t.x));
        try ks.gate(t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, t.shape.rows, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Gate) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Gate 4's stress fill is the random one: the gate's 16 terms summed in `acc`, the rest in float64.
    pub fn simulate(t: *Gate, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [16]f64 = undefined;
        for (0..t.shape.rows) |p| {
            for (&terms, 0..) |*v, i| v.* = @as(f64, t.gv[p * 16 + i]) * t.w[i];
            const sg = sigmoid(cm.rows.sumIn(acc, order, &terms) + t.b[0]);
            for (0..t.shape.n) |ch| out[p * t.shape.n + ch] = @as(f64, t.x[p * t.shape.n + ch]) * sg;
        }
    }

    pub fn reference(t: *Gate, want: []f64, mag: []f64) !void {
        for (0..t.shape.rows) |p| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (0..16) |i| {
                const v = @as(f64, t.gv[p * 16 + i]) * t.w[i];
                s += v;
                m += @abs(v);
            }
            const sg = sigmoid(s + t.b[0]);
            for (0..t.shape.n) |ch| {
                const x: f64 = t.x[p * t.shape.n + ch];
                want[p * t.shape.n + ch] = x * sg;
                const mm = @abs(x) * 0.25 * (m + @abs(@as(f64, t.b[0])));
                mag[p * t.shape.n + ch] = if (std.math.isFinite(mm)) mm else 0;
            }
        }
    }
};

// --- tf_br_head: conv_out1 over [the last level resized (align_corners) | ipt_blk1's channels] -----------------------

pub const Head = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shape: c.Shape = undefined,
    src: []f32 = &.{},
    q: []f32 = &.{},
    w: []f32 = &.{},
    b: [1]f32 = undefined,
    ins: [4]u64 = @splat(0),
    out: u64 = 0,

    const cq = 48;
    const in_side = 7;

    fn depth(s: c.Shape) usize {
        return s.k + 4;
    }

    /// Rows: the output side (rows x rows from 7 x 7); n: 1; k: the resized channels plus 48.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = &.{ .{ .rows = 29, .n = 1, .k = 240 }, .{ .rows = 64, .n = 1, .k = 240 } }, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = "each term is a bilinear sample rounded in fp32, so no input keeps fp32's sums exact" } };
    }

    pub fn deinit(t: *Head) void {
        inline for (.{ &t.src, &t.q, &t.w }) |p| t.a.free(p.*);
        t.rig.deinit();
    }
    pub fn kernel(t: *Head, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Head);

    fn cp(t: *const Head) usize {
        return t.shape.k - cq;
    }

    pub fn fill(t: *Head, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows < 2 or shape.n != 1 or shape.k <= cq) return false;
        inline for (.{ &t.src, &t.q, &t.w }) |p| t.a.free(p.*);
        t.shape = shape;
        t.src = try t.a.alloc(f32, in_side * in_side * t.cp());
        t.q = try t.a.alloc(f32, shape.rows * shape.rows * cq);
        t.w = try t.a.alloc(f32, shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        uniform(rng, t.src, -2, 2);
        uniform(rng, t.q, -2, 2);
        uniform(rng, t.w, -0.3, 0.3);
        uniform(rng, &t.b, -1, 1);
        if (mode == .nonfinite) plant(t.q, cq);
        return true;
    }

    pub fn prepare(t: *Head, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.src));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.q));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(&t.b));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Head) !void {
        const s = t.shape.rows;
        try ks.head(t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.out, .{ .h = in_side, .w = in_side }, .{ .h = s, .w = s }, t.cp(), cq);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Head) usize {
        return t.shape.rows * t.shape.rows;
    }

    /// Output (oy, ox)'s terms and magnitude; a sample counts its four corners whole, covering its position's rounding.
    fn terms(t: *const Head, oy: usize, ox: usize, out: []f64) f64 {
        const s = t.shape.rows;
        const y = cm.tap(in_side, s, oy);
        const x = cm.tap(in_side, s, ox);
        var m: f64 = 0;
        for (0..t.cp()) |ch| {
            const v = [4]f64{ t.src[(y.lo * in_side + x.lo) * t.cp() + ch], t.src[(y.lo * in_side + x.hi) * t.cp() + ch], t.src[(y.hi * in_side + x.lo) * t.cp() + ch], t.src[(y.hi * in_side + x.hi) * t.cp() + ch] };
            const top = @as(f64, y.l0) * (@as(f64, x.l0) * v[0] + @as(f64, x.l1) * v[1]);
            const bottom = @as(f64, y.l1) * (@as(f64, x.l0) * v[2] + @as(f64, x.l1) * v[3]);
            out[ch] = (top + bottom) * t.w[ch];
            m += (@abs(v[0]) + @abs(v[1]) + @abs(v[2]) + @abs(v[3])) * @abs(@as(f64, t.w[ch]));
        }
        const at = oy * s + ox;
        for (0..cq) |ch| {
            out[t.cp() + ch] = @as(f64, t.q[at * cq + ch]) * t.w[t.cp() + ch];
            m += @abs(out[t.cp() + ch]);
        }
        return m;
    }

    pub fn reference(t: *Head, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        const s = t.shape.rows;
        for (0..s) |oy| for (0..s) |ox| {
            const m = t.terms(oy, ox, buf);
            var sum: f64 = t.b[0];
            for (buf) |v| sum += v;
            want[oy * s + ox] = sum;
            const mm = m + @abs(@as(f64, t.b[0]));
            mag[oy * s + ox] = if (std.math.isFinite(mm)) mm else 0;
        };
    }

    pub fn simulate(t: *Head, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        const s = t.shape.rows;
        for (0..s) |oy| for (0..s) |ox| {
            _ = t.terms(oy, ox, buf);
            out[oy * s + ox] = cm.rows.sumIn(acc, order, buf) + t.b[0];
        };
    }
};

// --- tf_br_pool: the ASPP's global average pool, a channel's mean over the pixels -----------------------------------

pub const Pool = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    ins: [1]u64 = @splat(0),
    out: u64 = 0,

    /// A thread's strided chain, the warp's five levels and the eight warps' chain.
    fn depth(s: c.Shape) usize {
        return (s.k + 255) / 256 + 5 + 8;
    }

    /// Rows: channels; n: 1; k: pixels (32^2 .. 256^2 at 64 channels).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .ops = cm.div_ulps, .out = .f32, .tiles = .{ .{ .size = 8 }, .{}, .{ .size = 256 } }, .served = &.{ .{ .rows = 64, .n = 1, .k = 1024 }, .{ .rows = 64, .n = 1, .k = 4096 }, .{ .rows = 64, .n = 1, .k = 65536 } } };
    }

    pub fn deinit(t: *Pool) void {
        t.a.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *Pool, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Pool);

    pub fn fill(t: *Pool, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        t.a.free(t.x);
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.k);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (mode == .stress) {
            // integers to 255: every partial sum under 2^24 is exact in fp32, past fp16's range in a few hundred
            for (t.x) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, 0, 255));
        } else {
            uniform(rng, t.x, -3, 3);
            if (mode == .nonfinite) plant(t.x, shape.rows);
        }
        return true;
    }

    pub fn prepare(t: *Pool, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Pool) !void {
        try ks.pool(t.k, t.rig.s, t.ins[0], t.out, t.shape.k, t.shape.rows);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Pool) usize {
        return t.shape.rows;
    }

    fn column(t: *const Pool, ch: usize, out: []f64) void {
        for (out, 0..) |*v, i| v.* = t.x[i * t.shape.rows + ch];
    }

    pub fn reference(t: *Pool, want: []f64, mag: []f64) !void {
        const n: f64 = @floatFromInt(t.shape.k);
        for (0..t.shape.rows) |ch| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (0..t.shape.k) |i| {
                const v: f64 = t.x[i * t.shape.rows + ch];
                s += v;
                m += @abs(v);
            }
            want[ch] = s / n;
            mag[ch] = if (std.math.isFinite(m)) m / n else 0;
        }
    }

    pub fn simulate(t: *Pool, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.shape.k);
        defer t.a.free(buf);
        for (0..t.shape.rows) |ch| {
            t.column(ch, buf);
            out[ch] = cm.rows.sumIn(acc, order, buf) / @as(f64, @floatFromInt(t.shape.k));
        }
    }
};

// --- tf_br_resize: F.interpolate(bilinear, align_corners=True) between columns of row-major grids ------------------

/// One axis's source position as the kernel's fp32 computes it: its value and possible ends (8 ulps, as on Metal).
const Pos = struct { at: f64, lo: f64, hi: f64 };

fn position(in: usize, out: usize, o: usize) Pos {
    if (out <= 1) return .{ .at = 0, .lo = 0, .hi = 0 };
    const scale: f32 = @as(f32, @floatFromInt(in - 1)) / @as(f32, @floatFromInt(out - 1));
    const at: f64 = scale * @as(f32, @floatFromInt(o));
    const e = 8 * 0x1p-24 * at;
    return .{ .at = at, .lo = at - e, .hi = at + e };
}

/// x's bilinear value of channel ch at (fy, fx), the kernel's clamps, exactly.
fn sampleAt(x: []const f32, ld: usize, off: usize, h: usize, w: usize, ch: usize, fy: f64, fx: f64) f64 {
    const y0 = @min(@as(usize, @intFromFloat(@max(fy, 0))), h - 1);
    const x0 = @min(@as(usize, @intFromFloat(@max(fx, 0))), w - 1);
    const y1 = @min(y0 + 1, h - 1);
    const x1 = @min(x0 + 1, w - 1);
    const ly = std.math.clamp(fy - @as(f64, @floatFromInt(y0)), 0, 1);
    const lx = std.math.clamp(fx - @as(f64, @floatFromInt(x0)), 0, 1);
    const v = [4]f64{ x[(y0 * w + x0) * ld + off + ch], x[(y0 * w + x1) * ld + off + ch], x[(y1 * w + x0) * ld + off + ch], x[(y1 * w + x1) * ld + off + ch] };
    const top = (1 - lx) * v[0] + lx * v[1];
    const bottom = (1 - lx) * v[2] + lx * v[3];
    return (1 - ly) * top + ly * bottom;
}

/// The positions to try across [lo, hi]: both ends and the grid line between (a piece's extremes lie at its corners).
fn tries(p: Pos) [3]f64 {
    const mid = @floor(p.hi);
    return .{ p.lo, if (mid > p.lo) mid else p.lo, p.hi };
}

pub const Resize = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    ins: [1]u64 = @splat(0),
    out: u64 = 0,
    ends: Ends,

    const xoff = 2;

    /// Rows: the output's side; n: channels; k: the input's side (the input wider than its channels, from column 2).
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .none, .depth = cm.noSums, .out = .f32, .tiles = .{ .{ .size = 16 }, .{ .size = 64 }, .{} }, .served = &.{ .{ .rows = 512, .n = 3, .k = 1024 }, .{ .rows = 64, .n = 384, .k = 32 }, .{ .rows = 32, .n = 384, .k = 256 }, .{ .rows = 64, .n = 256, .k = 1 }, .{ .rows = 33, .n = 96, .k = 7 } } };
    }

    pub fn deinit(t: *Resize) void {
        t.a.free(t.x);
        t.ends.deinit();
        t.rig.deinit();
    }
    pub fn kernel(t: *Resize, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Resize);

    fn ldx(t: *const Resize) usize {
        return t.shape.n + 4;
    }

    pub fn fill(t: *Resize, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.a.free(t.x);
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.k * shape.k * t.ldx());
        var prng = std.Random.DefaultPrng.init(seed);
        uniform(prng.random(), t.x, -3, 3);
        if (mode == .nonfinite) plant(t.x[xoff..], t.ldx());
        return true;
    }

    pub fn prepare(t: *Resize, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Resize) !void {
        const k = t.shape.k;
        const s = t.shape.rows;
        try ks.resize(t.k, t.rig.s, t.ins[0], t.out, .{ .h = k, .w = k }, .{ .h = s, .w = s }, t.shape.n, t.ldx(), xoff, t.shape.n, 0);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Resize) usize {
        return t.shape.rows * t.shape.rows * t.shape.n;
    }

    /// The value at the nominal positions; its range over fp32's possible positions, widened by the eight roundings.
    pub fn reference(t: *Resize, want: []f64, mag: []f64) !void {
        try t.ends.size(t.count());
        const k = t.shape.k;
        const s = t.shape.rows;
        const n = t.shape.n;
        for (0..s) |oy| for (0..s) |ox| {
            const py = position(k, s, oy);
            const px = position(k, s, ox);
            for (0..n) |ch| {
                const w = sampleAt(t.x, t.ldx(), xoff, k, k, ch, py.at, px.at);
                var lo = w;
                var hi = w;
                var size: f64 = 0;
                for (tries(py)) |fy| for (tries(px)) |fx| {
                    const v = sampleAt(t.x, t.ldx(), xoff, k, k, ch, fy, fx);
                    lo = @min(lo, v);
                    hi = @max(hi, v);
                };
                const y0 = @min(@as(usize, @intFromFloat(py.at)), k - 1);
                const x0 = @min(@as(usize, @intFromFloat(px.at)), k - 1);
                for ([_]usize{ y0, @min(y0 + 1, k - 1) }) |yy| for ([_]usize{ x0, @min(x0 + 1, k - 1) }) |xx| {
                    size = @max(size, @abs(@as(f64, t.x[(yy * k + xx) * t.ldx() + xoff + ch])));
                };
                const i = (oy * s + ox) * n + ch;
                cm.put(&t.ends, want, mag, i, w, (Range{ .lo = lo, .hi = hi }).widen(8 * 0x1p-24 * size + 0x1p-149));
            }
        };
    }
    pub fn bounds(t: *Resize, want: []const f64, _: []const f64, allow: []f64) !void {
        t.ends.bounds(want, allow);
    }
    pub fn intervals(t: *Resize, low: []f64, high: []f64) !void {
        t.ends.intervals(low, high);
    }
};

// --- tf_br_tiles: the patch embedding's 4 x 4 patches and image2patches' grid of tiles, copies -----------------------

pub const Tiles = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    grid: bool,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    ins: [1]u64 = @splat(0),
    out: u64 = 0,

    /// Rows: the output's side; n: its channels (cpad, zero past 3 a^2); k: the tile's side a; 3 source channels.
    pub fn spec(name: []const u8, grid: bool) c.Spec {
        return .{ .name = name, .accum = .none, .depth = cm.noSums, .out = .f32, .tiles = .{ .{ .size = 8 }, .{}, .{} }, .served = if (grid) &.{ .{ .rows = 32, .n = 3072, .k = 32 }, .{ .rows = 64, .n = 768, .k = 16 }, .{ .rows = 128, .n = 192, .k = 8 }, .{ .rows = 256, .n = 64, .k = 4 } } else &.{ .{ .rows = 256, .n = 64, .k = 4 }, .{ .rows = 128, .n = 64, .k = 4 } } };
    }

    pub fn deinit(t: *Tiles) void {
        t.a.free(t.x);
        t.rig.deinit();
    }
    pub fn kernel(t: *Tiles, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.grid), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Tiles);

    fn side(t: *const Tiles) usize {
        return t.shape.rows * t.shape.k;
    }

    pub fn fill(t: *Tiles, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n < 3 * shape.k * shape.k) return false;
        t.a.free(t.x);
        t.shape = shape;
        t.x = try t.a.alloc(f32, t.side() * t.side() * 3);
        var prng = std.Random.DefaultPrng.init(seed);
        uniform(prng.random(), t.x, -3, 3);
        if (mode == .nonfinite) plant(t.x, 3);
        return true;
    }

    pub fn prepare(t: *Tiles, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Tiles) !void {
        const s = t.shape.rows;
        try ks.tiles(t.k, t.rig.s, t.ins[0], t.out, .{ .h = s, .w = s }, t.side(), 3, t.shape.k, t.shape.k, t.grid, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Tiles) usize {
        return t.shape.rows * t.shape.rows * t.shape.n;
    }

    /// Channel ch * a^2 + i * a + j of (oy, ox): a patch's pixel, or image2patches' tile (i, j) as einops places it.
    pub fn reference(t: *Tiles, want: []f64, mag: []f64) !void {
        const s = t.shape.rows;
        const a = t.shape.k;
        for (0..s * s) |px| for (0..t.shape.n) |col| {
            const i = px * t.shape.n + col;
            mag[i] = 0;
            if (col >= 3 * a * a) {
                want[i] = 0;
                continue;
            }
            const ch = col / (a * a);
            const ti = (col % (a * a)) / a;
            const tj = col % a;
            const oy = px / s;
            const ox = px % s;
            const sy = if (t.grid) ti * s + oy else oy * a + ti;
            const sx = if (t.grid) tj * s + ox else ox * a + tj;
            want[i] = t.x[(sy * t.side() + sx) * 3 + ch];
        };
    }
};
