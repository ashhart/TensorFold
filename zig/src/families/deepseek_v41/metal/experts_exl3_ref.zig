//! Serial DeepSeek V4.1 routed EXL3 experts: the bits the Metal lane must give, from the one-row EXL3 dot and main's.
const std = @import("std");
const fmt = @import("core").exl3_format;
const row = @import("core").exl3_projection_row_ref;

/// W[k, n] as stored (tiles [k/16][n/16], `width` words each), with fp16 input and output scales and no bias.
pub const Matrix = struct { width: fmt.Width, book: fmt.Codebook, trellis: []const i16, su: []const f16, sv: []const f16 };

/// The device's silu(g) = g / (1 + exp(-g)) at every bf16 gate (tf_dsv41_x3_silu): the one value not derived here.
pub const Silu = [65536]f32;

/// Metal's fp32 arithmetic flushes subnormal operands and results to a zero of the same sign.
fn ftz(v: f32) f32 {
    return if (v != 0 and @abs(v) < std.math.floatMin(f32)) std.math.copysign(@as(f32, 0), v) else v;
}

fn mul(a: f32, b: f32) f32 {
    return ftz(ftz(a) * ftz(b));
}

fn add(a: f32, b: f32) f32 {
    return ftz(ftz(a) + ftz(b));
}

fn sub(a: f32, b: f32) f32 {
    return ftz(ftz(a) - ftz(b));
}

/// The kernels' fp32 1/sqrt(128).
const inv: f32 = 0.08838834764831845;

/// tf_exl3_hadamard serially: strides 1 to 64, each pair (a, b) becoming (a + b, a - b).
pub fn hadamard(v: *[128]f32) void {
    var h: usize = 1;
    while (h < 128) : (h *= 2) {
        for (0..128) |i| if (i & h == 0) {
            const a = v[i];
            const b = v[i + h];
            v[i] = add(a, b);
            v[i + h] = sub(a, b);
        };
    }
}

pub fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate(bits >> 16);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// tf_exl3_rotate_in of a bf16 row: fp16(((x * su) rotated) * 1/sqrt(128)), a 128-block at a time.
pub fn rotIn(x: []const u16, su: []const f16, out: []f16) void {
    for (0..x.len / 128) |b| {
        var v: [128]f32 = undefined;
        for (&v, 0..) |*y, i| y.* = mul(fromBf16(x[b * 128 + i]), su[b * 128 + i]);
        hadamard(&v);
        for (v, 0..) |y, i| out[b * 128 + i] = @floatCast(mul(y, inv));
    }
}

/// tf_exl3_rotate_out of fp32 sums written as bf16: ((sums rotated) * 1/sqrt(128)) * sv.
pub fn rotOut(sums: []const f32, sv: []const f16, out: []u16) void {
    for (0..sums.len / 128) |b| {
        var v: [128]f32 = sums[b * 128 ..][0..128].*;
        hadamard(&v);
        for (v, 0..) |y, i| out[b * 128 + i] = bf16(mul(mul(y, inv), sv[b * 128 + i]));
    }
}

fn clip(x: f32, lo: f32, hi: f32) f32 {
    return if (std.math.isNan(x)) x else std.math.clamp(x, lo, hi);
}

/// experts.metal's epilogue: silu(gate clipped above) * up clipped * route weight, to bf16.
pub fn act(g: u16, u: u16, w: f32, limit: f32, silu: *const Silu) u16 {
    const gc = bf16(clip(fromBf16(g), -std.math.inf(f32), limit));
    return bf16(mul(mul(silu[gc], clip(fromBf16(u), -limit, limit)), w));
}

/// One rotated row through `m` (k = x.len, n = sums.len): exl3_projection_row_ref's serial dot over every tile column.
pub fn dot(m: Matrix, x: []const f16, sums: []f32) void {
    row.dot(m.width, m.book, m.trellis, sums.len / 16, x, 0, sums);
}

/// Scratch for one pair: rotated rows and sums of either direction.
pub const Work = struct {
    rotated: []f16,
    sums: [2][]f32,
    out: [2][]u16,

    pub fn init(a: std.mem.Allocator, hidden: usize, width: usize) !Work {
        const big = @max(hidden, width);
        const rotated = try a.alloc(f16, big);
        errdefer a.free(rotated);
        const s = try a.alloc(f32, 2 * big);
        errdefer a.free(s);
        const o = try a.alloc(u16, 2 * big);
        return .{ .rotated = rotated, .sums = .{ s[0..big], s[big..] }, .out = .{ o[0..big], o[big..] } };
    }

    pub fn deinit(w: Work, a: std.mem.Allocator) void {
        a.free(w.rotated);
        a.free(w.sums[0].ptr[0 .. 2 * w.rotated.len]);
        a.free(w.out[0].ptr[0 .. 2 * w.rotated.len]);
    }
};

/// One pair's gate and up from its token's bf16 row: the slots' sums (gate, up) into `sums`, act [width] bf16.
pub fn upPair(gate: Matrix, up: Matrix, x: []const u16, w: f32, limit: f32, silu: *const Silu, sums: [2][]f32, out: []u16, k: *Work) void {
    for ([_]Matrix{ gate, up }, 0..) |m, v| {
        rotIn(x, m.su, k.rotated[0..x.len]);
        dot(m, k.rotated[0..x.len], sums[v]);
        rotOut(sums[v], m.sv, k.out[v][0..out.len]);
    }
    for (out, k.out[0][0..out.len], k.out[1][0..out.len]) |*y, g, u| y.* = act(g, u, w, limit, silu);
}

/// One pair's down from its act row: the slots' sums into `sums`, its bf16 output [hidden] into `out`.
pub fn downPair(m: Matrix, act_row: []const u16, sums: []f32, out: []u16, k: *Work) void {
    rotIn(act_row, m.su, k.rotated[0..act_row.len]);
    dot(m, k.rotated[0..act_row.len], sums);
    rotOut(sums, m.sv, out);
}
