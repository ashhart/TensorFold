//! Ling KDA mixer: k3_kda job (conv, gates, delta, RMS). D=128, 4 taps; host formula plus GPU check.
const std = @import("std");
const config = @import("config.zig");

pub const dim = config.fixed.kda_dim;
pub const taps = config.fixed.conv;
/// Stack conv window in mixAs. Tiny is 16 heads x 128 = 2048. Wider Lings need a heap window.
pub const max_width: u32 = 2048;
pub const dump_h16_r2 = @embedFile("fixtures/kda_tiny_h16_r2.bin");
pub const dump_h16_r4 = @embedFile("fixtures/kda_tiny_h16_r4.bin");
const q_scale: f32 = 0.08838834764831845; // 1/sqrt(128)

pub const Shape = struct {
    heads: u32,
    lower_bound: f32,
    eps: f32,

    pub fn of(c: *const config.Config) Shape {
        return .{ .heads = c.kda_heads, .lower_bound = c.lower_bound, .eps = c.eps };
    }

    pub fn width(s: Shape) u32 {
        return s.heads * dim;
    }
};

fn sig(comptime T: type, x: T) T {
    return 1.0 / (1.0 + @exp(-x));
}

/// Same round-trip as Metal `float(bfloat(x))` / `k3_kda`'s conv and delta output.
fn roundBf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const h: u16 = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    return @bitCast(@as(u32, h) << 16);
}

fn maybeRound(comptime T: type, comptime to_bf16: bool, v: T) T {
    if (!to_bf16) return v;
    return @floatCast(roundBf16(@floatCast(v)));
}

fn mixAs(
    comptime T: type,
    comptime to_bf16: bool,
    y: []T,
    S: []T,
    proj: []const f32,
    fgate: []const f32,
    braw: []const f32,
    g2: []const f32,
    conv_q: []const f32,
    conv_k: []const f32,
    conv_v: []const f32,
    A_log: []const f32,
    dt_bias: []const f32,
    o_norm: []const f32,
    s: Shape,
    rows: u32,
) void {
    const W = s.width();
    const H = s.heads;
    const window_cap = 3 * max_width;
    var windows: [window_cap][3]T = @splat(.{ 0, 0, 0 });
    const scale: T = if (T == f32) q_scale else 1.0 / @sqrt(@as(T, @floatFromInt(dim)));
    const lower: T = @floatCast(s.lower_bound);
    const eps: T = @floatCast(s.eps);

    for (0..rows) |r| {
        for (0..H) |h| {
            var q: [dim]T = undefined;
            var k: [dim]T = undefined;
            var v: [dim]T = undefined;
            var a: [dim]T = undefined;
            const rate = @exp(@as(T, @floatCast(A_log[h])));
            for (0..dim) |d| {
                const ch_base = h * dim + d;
                inline for (.{ 0, 1, 2 }, .{ &q, &k, &v }, .{ conv_q, conv_k, conv_v }) |part, slot, cw| {
                    const ch = part * W + ch_base;
                    const x: T = @floatCast(proj[r * 3 * W + ch]);
                    const w = windows[ch];
                    const acc = w[0] * @as(T, @floatCast(cw[ch_base * taps])) +
                        w[1] * @as(T, @floatCast(cw[ch_base * taps + 1])) +
                        w[2] * @as(T, @floatCast(cw[ch_base * taps + 2])) +
                        x * @as(T, @floatCast(cw[ch_base * taps + 3]));
                    slot[d] = maybeRound(T, to_bf16, acc * sig(T, acc));
                    windows[ch] = .{ w[1], w[2], x };
                }
                const g = @as(T, @floatCast(fgate[r * W + ch_base])) + @as(T, @floatCast(dt_bias[ch_base]));
                a[d] = @exp(lower * sig(T, rate * g));
            }
            const beta = sig(T, @floatCast(braw[r * H + h]));
            var ss_q: T = 0;
            var ss_k: T = 0;
            for (0..dim) |d| {
                ss_q += q[d] * q[d];
                ss_k += k[d] * k[d];
            }
            const rq = @sqrt(ss_q + 1e-6);
            const rk = @sqrt(ss_k + 1e-6);
            for (0..dim) |d| {
                q[d] = q[d] / rq * scale;
                k[d] = k[d] / rk;
            }
            const Sh = S[h * dim * dim ..][0 .. dim * dim];
            var o: [dim]T = undefined;
            for (0..dim) |j| {
                var p: T = 0;
                for (0..dim) |i| p += Sh[j * dim + i] * a[i] * k[i];
                const u = (v[j] - p) * beta;
                for (0..dim) |i| Sh[j * dim + i] = Sh[j * dim + i] * a[i] + k[i] * u;
                var acc: T = 0;
                for (0..dim) |i| acc += Sh[j * dim + i] * q[i];
                o[j] = maybeRound(T, to_bf16, acc);
            }
            var var_o: T = 0;
            for (o) |v_o| var_o += v_o * v_o;
            const rstd = 1.0 / @sqrt(var_o / @as(T, @floatFromInt(dim)) + eps);
            for (0..dim) |v_i| {
                const g2v = sig(T, @floatCast(g2[r * W + h * dim + v_i]));
                y[r * W + h * dim + v_i] = maybeRound(T, to_bf16, o[v_i] * rstd * @as(T, @floatCast(o_norm[v_i])) * g2v);
            }
        }
    }
}

/// One stream of `rows` through one KDA layer, f32 with the kernel's bf16 rounds. `proj` is q|k|v at 3W.
pub fn mix(
    y: []f32,
    S: []f32,
    proj: []const f32,
    fgate: []const f32,
    braw: []const f32,
    g2: []const f32,
    conv_q: []const f32,
    conv_k: []const f32,
    conv_v: []const f32,
    A_log: []const f32,
    dt_bias: []const f32,
    o_norm: []const f32,
    s: Shape,
    rows: u32,
) error{KdaWidthTooWide}!void {
    if (s.width() > max_width) return error.KdaWidthTooWide;
    mixAs(f32, true, y, S, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);
}

/// Same mixer in float64, no extra rounding: oracle/kda_inner.py (copy of Model.kda, r = identity).
pub fn mixF64(
    y: []f64,
    S: []f64,
    proj: []const f32,
    fgate: []const f32,
    braw: []const f32,
    g2: []const f32,
    conv_q: []const f32,
    conv_k: []const f32,
    conv_v: []const f32,
    A_log: []const f32,
    dt_bias: []const f32,
    o_norm: []const f32,
    s: Shape,
    rows: u32,
) error{KdaWidthTooWide}!void {
    if (s.width() > max_width) return error.KdaWidthTooWide;
    mixAs(f64, false, y, S, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);
}

test "tiny KDA geometry is the k3_kda kernel's: 16 heads, D=128, 4 taps, lower_bound -5" {
    const tiny = @embedFile("fixtures/tiny_config.json");
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, tiny, &why);
    const s = Shape.of(&c);
    try std.testing.expectEqual(@as(u32, 16), s.heads);
    try std.testing.expectEqual(@as(u32, 128), dim);
    try std.testing.expectEqual(@as(u32, 4), taps);
    try std.testing.expectEqual(@as(u32, 18), c.countKind(.kda));
    try std.testing.expect(c.lower_bound == -5);
}

test "a zero stream stays zero: empty proj and gates give y = 0" {
    const s = Shape{ .heads = 1, .lower_bound = -5, .eps = 1e-6 };
    var y: [dim]f32 = undefined;
    var S: [dim * dim]f32 = @splat(0);
    const z3: [3 * dim]f32 = @splat(0);
    const z1: [dim]f32 = @splat(0);
    const braw = [_]f32{0};
    var conv: [dim * taps]f32 = @splat(0);
    const A = [_]f32{0};
    const on: [dim]f32 = @splat(1);
    try mix(&y, &S, &z3, &z1, &braw, &z1, &conv, &conv, &conv, &A, &z1, &on, s, 1);
    for (y) |v| try std.testing.expect(v == 0);
}

test "mix refuses width over 2048 so ReleaseFast cannot overrun the stack window" {
    const s = Shape{ .heads = 17, .lower_bound = -5, .eps = 1e-6 };
    try std.testing.expect(s.width() > max_width);
    try std.testing.expectError(error.KdaWidthTooWide, mix(&.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, s, 1));
    try std.testing.expectError(error.KdaWidthTooWide, mixF64(&.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, &.{}, s, 1));
}

fn takeU32(b: []const u8, off: *usize) u32 {
    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn takeF32(b: []const u8, off: *usize) f32 {
    const v = takeU32(b, off);
    return @bitCast(v);
}

fn takeF32s(a: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f32 {
    const x = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 4]);
    off.* += n * 4;
    return x;
}

fn checkOracle(bytes: []const u8) !void {
    const a = std.testing.allocator;
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGK1"));
    var off: usize = 8;
    const heads = takeU32(bytes, &off);
    const rows = takeU32(bytes, &off);
    const d = takeU32(bytes, &off);
    const tp = takeU32(bytes, &off);
    try std.testing.expectEqual(dim, d);
    try std.testing.expectEqual(taps, tp);
    const lower = takeF32(bytes, &off);
    const eps = takeF32(bytes, &off);
    off += 8; // seed, unused at check time
    const s = Shape{ .heads = heads, .lower_bound = lower, .eps = eps };
    const W = s.width();
    const H = s.heads;
    const proj = try takeF32s(a, bytes, &off, rows * 3 * W);
    defer a.free(proj);
    const fgate = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(fgate);
    const braw = try takeF32s(a, bytes, &off, rows * H);
    defer a.free(braw);
    const g2 = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(g2);
    const conv_q = try takeF32s(a, bytes, &off, W * taps);
    defer a.free(conv_q);
    const conv_k = try takeF32s(a, bytes, &off, W * taps);
    defer a.free(conv_k);
    const conv_v = try takeF32s(a, bytes, &off, W * taps);
    defer a.free(conv_v);
    const A_log = try takeF32s(a, bytes, &off, H);
    defer a.free(A_log);
    const dt_bias = try takeF32s(a, bytes, &off, W);
    defer a.free(dt_bias);
    const o_norm = try takeF32s(a, bytes, &off, dim);
    defer a.free(o_norm);
    const want_n = rows * W;
    const want = try a.alloc(f64, want_n);
    defer a.free(want);
    @memcpy(std.mem.sliceAsBytes(want), bytes[off .. off + want_n * 8]);
    off += want_n * 8;
    try std.testing.expectEqual(bytes.len, off);

    const got64 = try a.alloc(f64, want_n);
    defer a.free(got64);
    const S64 = try a.alloc(f64, H * dim * dim);
    defer a.free(S64);
    @memset(S64, 0);
    try mixF64(got64, S64, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);
    var worst64: f64 = 0;
    for (want, got64) |w_y, g_y| worst64 = @max(worst64, @abs(w_y - g_y));
    std.debug.print("mixF64 vs kda_inner.py worst abs {e} ({d} heads, {d} rows)\n", .{ worst64, H, rows });
    try std.testing.expect(worst64 < 1e-8);

    const got32 = try a.alloc(f32, want_n);
    defer a.free(got32);
    const S32 = try a.alloc(f32, H * dim * dim);
    defer a.free(S32);
    @memset(S32, 0);
    try mix(got32, S32, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);
    var row: u32 = 0;
    while (row < rows) : (row += 1) {
        var worst: f64 = 0;
        const lo = row * W;
        for (want[lo .. lo + W], got32[lo .. lo + W]) |w_y, g_y|
            worst = @max(worst, @abs(w_y - @as(f64, g_y)));
        std.debug.print("mix vs dump row {d} worst abs {d:.6} ({d} heads)\n", .{ row, worst, H });
    }
}

test "mixF64 matches oracle/kda_inner.py on tiny 16 heads x 2 rows" {
    try checkOracle(@embedFile("fixtures/kda_tiny_h16_r2.bin"));
}

test "mixF64 matches oracle/kda_inner.py on tiny 16 heads x 4 rows" {
    try checkOracle(@embedFile("fixtures/kda_tiny_h16_r4.bin"));
}
