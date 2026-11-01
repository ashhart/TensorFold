//! tf_br_conv's forms and tf_br_conv3 under the contract: BiRefNet's fp32 sums against float64 (as the Metal tests).
const std = @import("std");
const cm = @import("contract_birefnet_common.zig");

const c = cm.c;
const g = cm.g;
const ks = cm.ks;
const uniform = cm.uniform;
const plant = cm.plant;

/// GELU's exact form in float64, as torch writes it: x / 2 (1 + erf(x / sqrt 2)).
fn gelu(x: f64) f64 {
    return 0.5 * x * (1 + erf(x * std.math.sqrt1_2));
}

extern "c" fn erf(x: f64) f64;

/// A form the network launches; rows: rows (1x1) or picture rows of 9 columns (k x k); n outputs; k channels a tap.
pub const Form = struct { label: []const u8, side: usize = 1, flags: u32 = 0, served: []const c.Shape };

const bn = ks.flag_bn;
const relu = ks.flag_relu;
pub const forms = [_]Form{
    .{ .label = "tf_br_conv 1x1", .served = &.{ .{ .rows = 1100, .n = 576, .k = 192 }, .{ .rows = 300, .n = 1152, .k = 384 }, .{ .rows = 130, .n = 2304, .k = 768 }, .{ .rows = 70, .n = 4608, .k = 1536 }, .{ .rows = 1100, .n = 192, .k = 64 }, .{ .rows = 300, .n = 147, .k = 3136 } } },
    .{ .label = "tf_br_conv 1x1 gelu", .flags = ks.flag_gelu, .served = &.{ .{ .rows = 1100, .n = 768, .k = 192 }, .{ .rows = 70, .n = 6144, .k = 1536 } } },
    .{ .label = "tf_br_conv 1x1 + residual", .flags = ks.flag_add, .served = &.{ .{ .rows = 1100, .n = 192, .k = 768 }, .{ .rows = 130, .n = 1536, .k = 1536 } } },
    .{ .label = "tf_br_conv 1x1 + residual by map", .flags = ks.flag_add | ks.flag_map, .served = &.{ .{ .rows = 1100, .n = 192, .k = 192 }, .{ .rows = 300, .n = 768, .k = 768 } } },
    .{ .label = "tf_br_conv 1x1 bn relu", .flags = bn | relu, .served = &.{ .{ .rows = 300, .n = 256, .k = 3136 }, .{ .rows = 1100, .n = 64, .k = 1280 }, .{ .rows = 1, .n = 256, .k = 64 } } },
    .{ .label = "tf_br_conv 3x3", .side = 3, .served = &.{ .{ .rows = 8, .n = 64, .k = 3072 }, .{ .rows = 8, .n = 384, .k = 64 }, .{ .rows = 8, .n = 48, .k = 64 } } },
    .{ .label = "tf_br_conv 3x3 bn relu", .side = 3, .flags = bn | relu, .served = &.{ .{ .rows = 8, .n = 64, .k = 5760 }, .{ .rows = 8, .n = 16, .k = 1536 } } },
    .{ .label = "tf_br_conv 3x3 bn", .side = 3, .flags = bn, .served = &.{ .{ .rows = 8, .n = 1536, .k = 64 }, .{ .rows = 8, .n = 3072, .k = 64 } } },
    .{ .label = "tf_br_conv 7x7", .side = 7, .served = &.{.{ .rows = 8, .n = 147, .k = 64 }} },
};

const cols9 = 9;

pub const Conv = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    form: Form,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    post: []f32 = &.{},
    r: []f32 = &.{},
    map: []i32 = &.{},
    ins: [6]u64 = @splat(0),
    out: u64 = 0,

    fn depthOf(s: c.Shape, taps: usize) usize {
        return taps * s.k + 2;
    }
    fn depth1(s: c.Shape) usize {
        return depthOf(s, 1);
    }
    fn depth3(s: c.Shape) usize {
        return depthOf(s, 9);
    }
    fn depth7(s: c.Shape) usize {
        return depthOf(s, 49);
    }

    pub fn spec(name: []const u8, f: Form) c.Spec {
        return .{
            .name = name,
            .accum = .f32,
            .depth = switch (f.side) {
                1 => depth1,
                3 => depth3,
                else => depth7,
            },
            .ops = if (f.flags & ks.flag_gelu != 0) 8 else if (f.flags & bn != 0) 3 else 1,
            .out = .f32,
            .tiles = .{ if (f.side == 1) .{ .size = 64 } else .{ .size = 8 }, .{ .size = 64 }, .{ .size = 32, .step = 32, .min = 32 } },
            .served = f.served,
        };
    }

    pub fn deinit(t: *Conv) void {
        t.free();
        t.rig.deinit();
    }
    pub fn free(t: *Conv) void {
        inline for (.{ &t.x, &t.w, &t.b, &t.post, &t.r }) |p| {
            t.a.free(p.*);
            p.* = &.{};
        }
        t.a.free(t.map);
        t.map = &.{};
    }
    pub fn kernel(t: *Conv, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name, t.form), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Conv);

    fn ldw(t: *const Conv) usize {
        return std.mem.alignForward(usize, t.shape.n, 64);
    }
    fn cols(t: *const Conv) usize {
        return if (t.form.side == 1) 1 else cols9;
    }
    /// The GEMM's rows (a map's slots), and the output rows (a map's targets: one in nine slots is padding).
    fn slots(t: *const Conv) usize {
        return t.shape.rows * t.cols();
    }
    fn outRows(t: *const Conv) usize {
        return if (t.form.flags & ks.flag_map != 0) t.slots() - (t.slots() + 4) / 9 else t.slots();
    }
    fn has(t: *const Conv, f: u32) bool {
        return t.form.flags & f != 0;
    }

    /// The inputs on the host for a shape: false when the kernel refuses it.
    pub fn fill(t: *Conv, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        t.free();
        t.shape = shape;
        const L = t.ldw();
        const taps = t.form.side * t.form.side;
        const K = taps * shape.k;
        const px = t.slots();
        t.x = try t.a.alloc(f32, px * shape.k);
        t.w = try t.a.alloc(f32, K * L);
        t.b = try t.a.alloc(f32, L);
        t.post = try t.a.alloc(f32, 3 * L);
        t.r = try t.a.alloc(f32, if (t.has(ks.flag_add)) t.outRows() * shape.n else 0);
        t.map = try t.a.alloc(i32, if (t.has(ks.flag_map)) px else 0);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(t.w, 0);
        @memset(t.b, 0);
        @memset(t.post, 0);
        if (mode == .stress) {
            // second half mirrors the first, weights 1 then -1: a half's sum is exact in fp32, past fp16, and cancels
            const half = shape.k / 2;
            const big: f32 = @floatFromInt(@min(32639, (1 << 23) / half - 2041));
            for (0..px) |p| for (0..shape.k) |ci| {
                const at = if (ci < half) ci else ci - half;
                const v: f32 = if (at % 2 == 0) big else 2041;
                t.x[p * shape.k + ci] = if (ci < half) v else v + @as(f32, if (at < 4 and p % 3 == 0) 1 else 0);
            };
            for (0..K) |k| for (0..shape.n) |o| {
                t.w[k * L + o] = if (k % shape.k < half) 1 else -1;
            };
            // GELU on small positive sums; batch norms exact in fp32
            for (t.b[0..shape.n]) |*v| v.* = if (t.has(ks.flag_gelu)) 2 else @floatFromInt(rng.intRangeAtMost(i32, -2, 2));
            for (0..shape.n) |o| {
                t.post[o] = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
                t.post[L + o] = if (o % 2 == 0) 0.5 else 2;
                t.post[2 * L + o] = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
            }
            for (t.r) |*v| v.* = @floatFromInt(rng.intRangeAtMost(i32, -3, 3));
        } else {
            uniform(rng, t.x, -2, 2);
            for (0..K) |k| uniform(rng, t.w[k * L ..][0..shape.n], -0.1, 0.1);
            uniform(rng, t.b[0..shape.n], -1, 1);
            uniform(rng, t.post[0..shape.n], -0.5, 0.5);
            uniform(rng, t.post[L..][0..shape.n], 0.2, 3);
            uniform(rng, t.post[2 * L ..][0..shape.n], -1, 1);
            uniform(rng, t.r, -3, 3);
            if (mode == .nonfinite) plant(t.x, shape.k);
        }
        if (t.map.len > 0) {
            // every ninth slot padding, the rest a shuffled permutation of the output rows
            const targets = try t.a.alloc(i32, t.outRows());
            defer t.a.free(targets);
            for (targets, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, targets);
            var next: usize = 0;
            for (t.map, 0..) |*m, s| {
                if (s % 9 == 4) {
                    m.* = -1;
                } else {
                    m.* = targets[next];
                    next += 1;
                }
            }
        }
        return true;
    }

    pub fn prepare(t: *Conv, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.post));
        t.ins[4] = if (t.r.len > 0) try t.rig.input(std.mem.sliceAsBytes(t.r)) else 0;
        t.ins[5] = if (t.map.len > 0) try t.rig.input(std.mem.sliceAsBytes(t.map)) else 0;
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Conv) !void {
        const s: ks.ConvShape = .{ .rows = t.shape.rows, .cols = t.cols(), .cin = t.shape.k, .ldx = t.shape.k, .cout = t.shape.n, .ldw = t.ldw(), .k = t.form.side, .ldy = t.shape.n, .flags = t.form.flags };
        try ks.conv(t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.ins[4], t.ins[5], t.out, s);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Conv) usize {
        return t.outRows() * t.shape.n;
    }

    /// Slot p's products for output o in the kernel's order (tap, then channel); taps past the picture are 0.
    fn terms(t: *const Conv, p: usize, o: usize, out: []f64) void {
        const K = t.shape.k;
        const side = t.form.side;
        const pad: isize = @intCast(side / 2);
        const oy: isize = @intCast(p / t.cols());
        const ox: isize = @intCast(p % t.cols());
        for (0..side * side) |tp| {
            const iy = oy + @as(isize, @intCast(tp / side)) - pad;
            const ix = ox + @as(isize, @intCast(tp % side)) - pad;
            const inside = iy >= 0 and ix >= 0 and iy < t.shape.rows and ix < t.cols();
            for (0..K) |ci| {
                out[tp * K + ci] = if (inside) blk: {
                    const at = (@as(usize, @intCast(iy)) * t.cols() + @as(usize, @intCast(ix))) * K + ci;
                    break :blk @as(f64, t.x[at]) * t.w[(tp * K + ci) * t.ldw() + o];
                } else 0;
            }
        }
    }

    /// Output row of slot p (a map's target), or null where the map pads.
    fn target(t: *const Conv, p: usize) ?usize {
        if (t.map.len == 0) return p;
        return if (t.map[p] < 0) null else @intCast(t.map[p]);
    }

    /// The epilogue on s = sum + bias: its value, and the magnitude its error scales with.
    fn epilogue(t: *const Conv, s: f64, mag: f64, o: usize, i: usize) [2]f64 {
        const L = t.ldw();
        var v = s;
        var m = mag;
        if (t.has(bn)) {
            const scale: f64 = t.post[L + o];
            v = (v - t.post[o]) * scale + t.post[2 * L + o];
            m = @abs(scale) * (m + @abs(@as(f64, t.post[o]))) + @abs(@as(f64, t.post[2 * L + o]));
        }
        if (t.has(relu) and v < 0) v = 0;
        if (t.has(ks.flag_gelu)) {
            m = 1.13 * m + @abs(v);
            v = gelu(v);
        }
        if (t.has(ks.flag_add)) {
            v = t.r[i] + v;
            m += @abs(@as(f64, t.r[i]));
        }
        return .{ v, m };
    }

    pub fn reference(t: *Conv, want: []f64, mag: []f64) !void {
        const buf = try t.a.alloc(f64, t.form.side * t.form.side * t.shape.k);
        defer t.a.free(buf);
        for (0..t.slots()) |p| {
            const row = t.target(p) orelse continue;
            for (0..t.shape.n) |o| {
                t.terms(p, o, buf);
                var sum: f64 = 0;
                var m: f64 = 0;
                for (buf) |v| {
                    sum += v;
                    m += @abs(v);
                }
                const i = row * t.shape.n + o;
                const e = t.epilogue(sum + t.b[o], m + @abs(@as(f64, t.b[o])), o, i);
                want[i] = e[0];
                mag[i] = if (std.math.isFinite(e[1])) e[1] else 0;
            }
        }
    }

    pub fn simulate(t: *Conv, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try t.a.alloc(f64, t.form.side * t.form.side * t.shape.k);
        defer t.a.free(buf);
        for (0..t.slots()) |p| {
            const row = t.target(p) orelse continue;
            for (0..t.shape.n) |o| {
                t.terms(p, o, buf);
                const i = row * t.shape.n + o;
                out[i] = t.epilogue(cm.rows.sumIn(acc, order, buf) + t.b[o], 0, o, i)[0];
            }
        }
    }
};

// --- tf_br_conv3: a 3x3 of the 3-channel picture, fp32 FMAs ---------------------------------------------------------

pub const Conv3 = struct {
    a: std.mem.Allocator,
    rig: cm.rig_.Rig,
    k: *const ks.Kernels,
    name: []const u8,
    shape: c.Shape = undefined,
    x: []f32 = &.{},
    w: []f32 = &.{},
    b: []f32 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    fn depth(_: c.Shape) usize {
        return 27 + 1;
    }

    /// Rows: the picture's rows; n: output channels; k: its columns.
    pub fn spec(name: []const u8) c.Spec {
        return .{ .name = name, .accum = .f32, .depth = depth, .out = .f32, .tiles = .{ .{ .size = 4 }, .{ .size = 64 }, .{ .size = 4 } }, .served = &.{ .{ .rows = 12, .n = 64, .k = 1024 }, .{ .rows = 1024, .n = 64, .k = 12 }, .{ .rows = 37, .n = 64, .k = 41 } } };
    }

    pub fn deinit(t: *Conv3) void {
        inline for (.{ &t.x, &t.w, &t.b }) |p| t.a.free(p.*);
        t.rig.deinit();
    }
    pub fn kernel(t: *Conv3, control: ?g.Gate) g.Kernel {
        return .{ .spec = spec(t.name), .ctx = t, .vt = &vt, .control = control };
    }
    const vt = g.hooks(Conv3);

    pub fn fill(t: *Conv3, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n == 0) return false;
        inline for (.{ &t.x, &t.w, &t.b }) |p| t.a.free(p.*);
        t.shape = shape;
        t.x = try t.a.alloc(f32, shape.rows * shape.k * 3);
        t.w = try t.a.alloc(f32, 27 * shape.n);
        t.b = try t.a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (mode == .stress) {
            // pixels 2041 and 127 by parity, weights 509, -509, 1 by channel: partial sums ~2^20, cancelling
            for (0..shape.rows * shape.k) |p| {
                const v: f32 = if (p % 2 == 0) 2041 else 127;
                t.x[p * 3] = v;
                t.x[p * 3 + 1] = v + 1;
                t.x[p * 3 + 2] = 0;
            }
            for (0..27) |rr| @memset(t.w[rr * shape.n ..][0..shape.n], switch (rr % 3) {
                0 => 509,
                1 => -509,
                else => 1,
            });
            @memset(t.b, 0);
        } else {
            uniform(rng, t.x, -2, 2);
            uniform(rng, t.w, -0.5, 0.5);
            uniform(rng, t.b, -1, 1);
            if (mode == .nonfinite) plant(t.x, shape.k * 3);
        }
        return true;
    }

    pub fn prepare(t: *Conv3, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        t.rig.reset();
        if (!try t.fill(shape, mode, seed)) return false;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.out = try t.rig.output(t.count() * 4);
        return true;
    }

    pub fn run(t: *Conv3) !void {
        try ks.conv3(t.k, t.rig.s, t.ins[0], t.ins[1], t.ins[2], t.out, .{ .h = t.shape.rows, .w = t.shape.k }, t.shape.n);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Conv3) usize {
        return t.shape.rows * t.shape.k * t.shape.n;
    }

    fn terms(t: *const Conv3, px: usize, o: usize, out: *[27]f64) void {
        const py: isize = @intCast(px / t.shape.k);
        const pxx: isize = @intCast(px % t.shape.k);
        for (0..3) |ky| for (0..3) |kx| for (0..3) |ch| {
            const iy = py + @as(isize, @intCast(ky)) - 1;
            const ix = pxx + @as(isize, @intCast(kx)) - 1;
            const at = (ky * 3 + kx) * 3 + ch;
            out[at] = if (iy < 0 or ix < 0 or iy >= t.shape.rows or ix >= t.shape.k) 0 else @as(f64, t.x[(@as(usize, @intCast(iy)) * t.shape.k + @as(usize, @intCast(ix))) * 3 + ch]) * t.w[at * t.shape.n + o];
        };
    }

    pub fn reference(t: *Conv3, want: []f64, mag: []f64) !void {
        var buf: [27]f64 = undefined;
        for (0..t.shape.rows * t.shape.k) |px| for (0..t.shape.n) |o| {
            t.terms(px, o, &buf);
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf) |v| {
                s += v;
                m += @abs(v);
            }
            const i = px * t.shape.n + o;
            want[i] = s + t.b[o];
            m += @abs(@as(f64, t.b[o]));
            mag[i] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn simulate(t: *Conv3, acc: c.Accum, order: g.Order, out: []f64) !void {
        var buf: [27]f64 = undefined;
        for (0..t.shape.rows * t.shape.k) |px| for (0..t.shape.n) |o| {
            t.terms(px, o, &buf);
            out[px * t.shape.n + o] = cm.rows.sumIn(acc, order, &buf) + t.b[o];
        };
    }
};
