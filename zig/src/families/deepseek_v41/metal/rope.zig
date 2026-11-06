//! The RoPE tables a layer reads: [position][64] fp32, 32 cosines then 32 sines, as the reference's freqs_cis.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;

/// Sliding-window layers rotate at rope_theta without YaRN; compressed layers at compress_rope_theta with YaRN.
pub const Kind = enum { window, compressed };

/// The 32 frequencies of a kind, in fp32 as precompute_freqs_cis forms them.
pub fn freqs(c: *const Config, kind: Kind) [32]f32 {
    const dim: f64 = @floatFromInt(c.rope_dim);
    const base: f64 = if (kind == .window) c.rope.theta else c.rope.compress_theta;
    const yarn = kind == .compressed and c.rope.yarn and c.rope.original_max_positions > 0;
    var f: [32]f32 = undefined;
    for (&f, 0..) |*v, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatCast(dim));
        v.* = 1.0 / std.math.pow(f32, @floatCast(base), e);
    }
    if (!yarn) return f;
    const orig: f64 = @floatFromInt(c.rope.original_max_positions);
    const corrected = struct {
        fn at(d: f64, o: f64, b: f64, rot: f64) f64 {
            return d * @log(o / (rot * 2 * std.math.pi)) / (2 * @log(b));
        }
    }.at;
    const low = @max(@floor(corrected(dim, orig, base, c.rope.beta_fast)), 0);
    const high = @min(@ceil(corrected(dim, orig, base, c.rope.beta_slow)), dim - 1);
    const factor: f32 = @floatCast(c.rope.factor);
    for (&f, 0..) |*v, i| {
        const ramp: f32 = std.math.clamp((@as(f32, @floatFromInt(i)) - @as(f32, @floatCast(low))) / @as(f32, @floatCast(@max(high - low, 1e-3))), 0, 1);
        const smooth = 1 - ramp;
        v.* = v.* / factor * (1 - smooth) + v.* * smooth;
    }
    return f;
}

/// One kind's table for positions [0, positions).
pub fn fill(out: []f32, c: *const Config, kind: Kind, positions: usize) void {
    const f = freqs(c, kind);
    for (0..positions) |p| for (0..32) |i| {
        const angle = @as(f32, @floatFromInt(p)) * f[i];
        out[p * 64 + i] = @cos(angle);
        out[p * 64 + 32 + i] = @sin(angle);
    };
}

/// Both kinds' tables on the device.
pub const Tables = struct {
    window: mtl.Buffer,
    compressed: mtl.Buffer,
    positions: usize,

    pub fn init(device: mtl.Device, c: *const Config, positions: usize) !Tables {
        const bytes = positions * 64 * 4;
        const w = try device.buffer(bytes, mtl.ResourceOptions.shared);
        errdefer w.deinit();
        const k = try device.buffer(bytes, mtl.ResourceOptions.shared);
        fill(w.slice(f32, positions * 64), c, .window, positions);
        fill(k.slice(f32, positions * 64), c, .compressed, positions);
        return .{ .window = w, .compressed = k, .positions = positions };
    }

    pub fn deinit(t: *Tables) void {
        t.window.deinit();
        t.compressed.deinit();
    }

    pub fn of(t: *const Tables, kind: Kind) mtl.Buffer {
        return if (kind == .window) t.window else t.compressed;
    }
};
