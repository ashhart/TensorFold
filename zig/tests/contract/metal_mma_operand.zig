//! M5 tensor units' fp32 mma_16x32 operands: low 13 mantissa bits cleared, bit-equal to float64; controls go red.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;

/// Which operands are fp32: the left (attention's P), the right, or both (gemm_nax's f32 kernels).
const Side = enum { left, right, both };

/// How the reference takes an fp32 operand: the measured cut, or a planted control's wrong statement of it.
const Assume = enum { cut, fp32, nearest };

const Case = struct { side: Side, assume: Assume, name: []const u8 };

const cases = [_]Case{
    .{ .side = .left, .assume = .cut, .name = "metal mma_16x32 fp32 left operand cut" },
    .{ .side = .right, .assume = .cut, .name = "metal mma_16x32 fp32 right operand cut" },
    .{ .side = .both, .assume = .cut, .name = "metal mma_16x32 fp32 operands cut" },
    .{ .side = .left, .assume = .fp32, .name = "control: metal mma_16x32 left operand stated as fp32" },
    .{ .side = .right, .assume = .fp32, .name = "control: metal mma_16x32 right operand stated as fp32" },
    .{ .side = .left, .assume = .nearest, .name = "control: metal mma_16x32 left operand stated as rounded to nearest" },
    .{ .side = .right, .assume = .nearest, .name = "control: metal mma_16x32 right operand stated as rounded to nearest" },
};

const source =
    \\#include "../nax.h"
    \\using namespace tfp;
    \\[[kernel]] void tf_operand_cut(const device LEFT_T* A [[buffer(0)]], const device RIGHT_T* B [[buffer(1)]],
    \\    device float* C [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<LEFT_T> a;
    \\  frag<RIGHT_T> b0, b1;
    \\  frag_get(a, A, 16, 0, 0, home);
    \\  frag_get(b0, B, 32, 0, 0, home);
    \\  frag_get(b1, B, 32, 0, 16, home);
    \\  frag<float> c0 = frag<float>(0), c1 = frag<float>(0);
    \\  mma_16x32<false, false>(c0, c1, a, b0, b1);
    \\  frag_put(c0, C, 32, 0, 0, home);
    \\  frag_put(c1, C, 32, 0, 16, home);
    \\}
;

/// One op: C (16 x 32) = A (16 x 16) times B (16 x 32), fp32 accumulators.
const m = 16;
const n = 32;
const kd = 16;

/// The measured cut: fp32's low 13 mantissa bits cleared; NaN stays NaN.
fn cut(x: f32) f64 {
    if (std.math.isNan(x)) return x;
    return @as(f32, @bitCast(@as(u32, @bitCast(x)) & 0xffffe000));
}

/// 10 mantissa bits rounded to nearest, ties to even: the nearest control's statement.
fn nearest(x: f32) f64 {
    if (!std.math.isFinite(x)) return x;
    const b: u32 = @bitCast(x);
    const r = (b + 0xfff + ((b >> 13) & 1)) & 0xffffe000;
    return @as(f32, @bitCast(r));
}

fn noSums(_: c.Shape) usize {
    return 0;
}

/// No sums are tested: each output is one nonzero product, exact in fp32, so every output must equal float64's.
fn spec(v: Case) c.Spec {
    return .{
        .name = v.name,
        .accum = .none,
        .depth = noSums,
        .out = .f32,
        .served = &.{ .{ .rows = m, .n = n, .k = kd }, .{ .rows = 7, .n = n, .k = kd }, .{ .rows = 1, .n = n, .k = kd } },
    };
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    v: Case,
    rows: usize = 0,
    /// Both operands as values, row-major: what the device was given.
    x: [m * kd]f32 = undefined,
    y: [kd * n]f32 = undefined,
    refs: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Case) !Probe {
        const head = switch (v.side) {
            .left => "#define LEFT_T float\n#define RIGHT_T bfloat\n",
            .right => "#define LEFT_T bfloat\n#define RIGHT_T float\n",
            .both => "#define LEFT_T float\n#define RIGHT_T float\n",
        };
        const body = try tf.frags.source(device, a, source);
        defer a.free(body);
        const src = try std.mem.concat(a, u8, &.{ head, body });
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, "tf_operand_cut"), .v = v };
    }

    pub fn deinit(p: *Probe) void {
        p.pipe.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.v), .ctx = p, .vt = &vt, .control = if (p.v.assume == .cut) null else .shapes };
    }

    const vt = g.hooks(Probe);

    fn leftF32(p: *const Probe) bool {
        return p.v.side != .right;
    }

    fn rightF32(p: *const Probe) bool {
        return p.v.side != .left;
    }

    /// A probed fp32 value: random sign and all 23 mantissa bits, exponent in 2^-30 to 2^30, or a subnormal.
    fn probed(rng: std.Random, subnormals: bool) f32 {
        const sign = @as(u32, rng.int(u1)) << 31;
        const mant = rng.int(u23);
        if (subnormals and rng.uintLessThan(u32, 8) == 0) return @bitCast(sign | @max(mant, 1));
        const exp: u32 = 127 - 30 + rng.uintLessThan(u32, 61);
        return @bitCast(sign | exp << 23 | mant);
    }

    /// The other side's value: plus or minus 2^0 to 2^8, so a product is exact in fp32, subnormals included.
    fn scale(rng: std.Random) f32 {
        const s: f32 = if (rng.boolean()) -1 else 1;
        return s * std.math.ldexp(@as(f32, 1), @intCast(rng.uintLessThan(u32, 9)));
    }

    /// One product an output: left and both C[r][j] = A[r][j % 16] B[j % 16][j], right C[r][j] = A[r][r] B[r][j].
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > m or shape.n != n or shape.k != kd) return false;
        p.rig.reset();
        p.rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        @memset(&p.x, 0);
        @memset(&p.y, 0);
        const sub = p.v.side != .both; // a scale side keeps a cut subnormal's product exact
        if (p.v.side == .right) {
            for (0..shape.rows) |r| p.x[r * kd + r] = scale(rng);
            for (&p.y) |*y| y.* = probed(rng, sub);
        } else {
            for (p.x[0 .. shape.rows * kd]) |*x| x.* = probed(rng, sub);
            for (0..n) |j| p.y[(j % kd) * n + j] = if (p.v.side == .both) probed(rng, false) else scale(rng);
        }
        if (fill == .nonfinite) {
            // NaN, +Inf, -Inf a row each and both infinities in a fourth, on the fp32 side the case probes
            const at = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, 1 }, .{ 3, 9 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |q, v| if (q[0] < shape.rows) {
                if (p.v.side == .right) p.y[q[0] * n + q[1] * 2] = v else p.x[q[0] * kd + q[1]] = v;
            };
        }
        p.refs[0] = try p.upload(&p.x, p.leftF32());
        p.refs[1] = try p.upload(&p.y, p.rightF32());
        p.out = try p.rig.output(m * n * 4);
        return true;
    }

    fn upload(p: *Probe, vals: []const f32, f32_side: bool) !rig_.Ref {
        if (f32_side) return p.rig.input(std.mem.sliceAsBytes(vals));
        var h: [kd * n]u16 = undefined;
        for (vals, 0..) |v, i| h[i] = c.toBf16(v);
        return p.rig.input(std.mem.sliceAsBytes(h[0..vals.len]));
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        pass.e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
        pass.e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
        pass.e.setBuffer(p.out.buf, p.out.off, 2);
        pass.e.dispatchThreads(mtl.Size.of(32, 1, 1), mtl.Size.of(32, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Probe) usize {
        return p.rows * n;
    }

    /// An fp32 operand as the case states the tensor units take it (bf16 values are exact).
    fn take(p: *const Probe, x: f32, f32_side: bool) f64 {
        if (!f32_side) return x;
        return switch (p.v.assume) {
            .cut => cut(x),
            .fp32 => x,
            .nearest => nearest(x),
        };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        for (0..p.rows) |r| for (0..n) |j| {
            var s: f64 = 0;
            var t: f64 = 0;
            for (0..kd) |k| {
                const prod = p.take(p.x[r * kd + k], p.leftF32()) * p.take(p.y[k * n + j], p.rightF32());
                s += prod;
                t += @abs(prod);
            }
            want[r * n + j] = s;
            mag[r * n + j] = t;
        };
    }

    /// Bit-equal: one exact product an output leaves no rounding to allow for.
    pub fn bounds(_: *Probe, _: []const f64, _: []const f64, allow: []f64) !void {
        @memset(allow, 0);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    /// Only on the tensor units, where the cut was measured; before the M5 the operands' precision is unmeasured.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (cases) |v| {
            if (only) |o| if (std.mem.indexOf(u8, v.name, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, v);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
