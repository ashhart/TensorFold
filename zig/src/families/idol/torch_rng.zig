//! torch's CPU generator and its float32 randn, so the denoiser's starting noise is torch.Generator("cpu")'s.
const std = @import("std");

/// torch's CPU generator: mt19937 seeded as CPUGeneratorImpl seeds it.
pub const TorchCpu = struct {
    s: [624]u32,
    i: usize = 624,
    /// normal_distribution<double>'s cached second sample
    spare: ?f64 = null,

    pub fn init(seed: u64) TorchCpu {
        var g: TorchCpu = .{ .s = undefined };
        g.s[0] = @truncate(seed);
        for (1..624) |j| g.s[j] = 1812433253 *% (g.s[j - 1] ^ (g.s[j - 1] >> 30)) +% @as(u32, @intCast(j));
        return g;
    }

    fn next(g: *TorchCpu) u32 {
        if (g.i >= 624) {
            for (0..624) |k| {
                const y = (g.s[k] & 0x80000000) | (g.s[(k + 1) % 624] & 0x7fffffff);
                g.s[k] = g.s[(k + 397) % 624] ^ (y >> 1) ^ (if (y & 1 != 0) @as(u32, 0x9908b0df) else 0);
            }
            g.i = 0;
        }
        var y = g.s[g.i];
        g.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        return y ^ (y >> 18);
    }

    fn uniform(g: *TorchCpu) f32 {
        return @as(f32, @floatFromInt(g.next() & 0xffffff)) * 0x1p-24;
    }

    /// normal_fill_16: Box-Muller on 8 pairs in place; theta is rounded to float as torch's scalar_t is.
    fn fill16(d: *[16]f32) void {
        for (0..8) |j| {
            const r = @sqrt(-2 * logf(1 - d[j]));
            const theta: f32 = @floatCast(2.0 * std.math.pi * @as(f64, d[j + 8]));
            d[j] = r * @as(f32, @floatCast(@cos(@as(f64, theta))));
            d[j + 8] = r * @as(f32, @floatCast(@sin(@as(f64, theta))));
        }
    }

    /// torch.randn(out.len, generator=g) for float32: normal_fill from 16 values on, else normal_distribution<double>.
    pub fn randn(g: *TorchCpu, out: []f32) void {
        if (out.len >= 16) {
            for (out) |*v| v.* = g.uniform();
            var i: usize = 0;
            while (i + 16 <= out.len) : (i += 16) fill16(out[i..][0..16]);
            if (out.len % 16 != 0) {
                const tail = out[out.len - 16 ..][0..16];
                for (tail) |*v| v.* = g.uniform();
                fill16(tail);
            }
            return;
        }
        for (out) |*v| v.* = @floatCast(g.normal());
    }

    fn normal(g: *TorchCpu) f64 {
        if (g.spare) |s| {
            g.spare = null;
            return s;
        }
        const u_1 = g.uniform53();
        const u_2 = g.uniform53();
        const r = @sqrt(-2 * std.math.log1p(-u_2));
        const theta = 2 * std.math.pi * u_1;
        g.spare = r * @sin(theta);
        return r * @cos(theta);
    }

    fn uniform53(g: *TorchCpu) f64 {
        const hi: u64 = g.next();
        const lo: u64 = g.next();
        return @as(f64, @floatFromInt(((hi << 32) | lo) & ((1 << 53) - 1))) * 0x1p-53;
    }
};

extern "c" fn logf(x: f32) f32;

fn bits(x: f32) u32 {
    return @bitCast(x);
}

/// Bits from torch 2.5.1+cpu, x86-64 Linux, ATEN_CPU_CAPABILITY=default (the scalar path the Mac runs).
const produced = "g = torch.Generator('cpu').manual_seed(42); torch.randn([8192, 16], generator=g), then [5]";

test "seed 42's [8192, 16] then [5] draws match torch's" {
    _ = produced;
    const a = std.testing.allocator;
    const latent = try a.alloc(f32, 8192 * 16);
    defer a.free(latent);
    var g = TorchCpu.init(42);
    g.randn(latent);
    var camera: [5]f32 = undefined;
    g.randn(&camera);
    // the camera (normal_distribution<double>, no float libm): bit-equal, on AVX2 torch too
    const want = [5]u32{ 0x3fb5a113, 0xbf66bf0d, 0xbf991081, 0x3f1039dc, 0xbf1e8dff };
    for (camera, want) |x, w| try std.testing.expectEqual(w, bits(x));
    // the latent within 3 ulps: torch's float logf, cosf and sinf are libm's (5,854 of 131,072 differ on glibc)
    const ends = [8]u32{ 0x3ff6a527, 0x3fbe5f54, 0x3f669567, 0xc006c0dd, 0xbf829062, 0x3e5e3715, 0x3f1d4a81, 0xbf31a466 };
    for (ends, 0..) |w, k| {
        const x = latent[if (k < 4) k else latent.len - 8 + k];
        try std.testing.expect(@abs(@as(i64, bits(x)) - @as(i64, w)) <= 3);
    }
    var sum: f64 = 0;
    for (latent) |x| sum += x;
    try std.testing.expectApproxEqAbs(22.993049482419792, sum, 1e-4);
}
