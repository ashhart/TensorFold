//! The pipeline's seeded starting latents: torch.randn on a CPU generator, as torch's x86 AVX2 path computes them.
const std = @import("std");

// avx_mathfun.h (zlib licence), multiply-adds fused as in torch's GCC build; ARM torch's libm path differs from it.

/// torch's mt19937 (ATen's MT19937RNGEngine): the seed's low 32 bits, outputs tempered.
pub const Twister = struct {
    state: [624]u32,
    next: usize = 0,
    left: usize = 1,

    pub fn init(seed: u64) Twister {
        var t: Twister = .{ .state = undefined };
        t.state[0] = @truncate(seed);
        for (1..624) |j| {
            const p = t.state[j - 1];
            t.state[j] = 1812433253 *% (p ^ (p >> 30)) +% @as(u32, @intCast(j));
        }
        return t;
    }

    fn twist(u: u32, v: u32) u32 {
        return (((u & 0x80000000) | (v & 0x7fffffff)) >> 1) ^ (if (v & 1 != 0) @as(u32, 0x9908b0df) else 0);
    }

    fn refill(t: *Twister) void {
        const s = &t.state;
        for (0..624) |k| s[k] = s[(k + 397) % 624] ^ twist(s[k], s[(k + 1) % 624]);
        t.left = 624;
        t.next = 0;
    }

    pub fn random(t: *Twister) u32 {
        t.left -= 1;
        if (t.left == 0) t.refill();
        var y = t.state[t.next];
        t.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        return y;
    }

    /// uniform_real_distribution<float>(0, 1): the low 24 bits over 2^24.
    pub fn uniform(t: *Twister) f32 {
        return @as(f32, @floatFromInt(t.random() & 0xffffff)) * 0x1p-24;
    }
};

const V = @Vector(8, f32);
const I = @Vector(8, i32);

fn splat(x: f32) V {
    return @splat(x);
}

fn fma(a: V, b: V, c: V) V {
    return @mulAdd(V, a, b, c);
}

/// avx_mathfun's log256_ps; its inputs here lie in [2^-24, 1].
fn log(input: V) V {
    const x0 = @max(input, splat(0x1p-126));
    const bits: I = @bitCast(x0);
    const e0: V = @floatFromInt((bits >> @splat(23)) - @as(I, @splat(0x7f)));
    var x: V = @bitCast((bits & @as(I, @splat(~@as(i32, 0x7f800000)))) | @as(I, @bitCast(splat(0.5))));
    const below = x < splat(0.707106781186547524);
    const e = e0 + splat(1) - @select(f32, below, splat(1), splat(0));
    x = (x - splat(1)) + @select(f32, below, x, splat(0));
    const z = x * x;
    var y = splat(7.0376836292e-2);
    inline for (.{ -1.1514610310e-1, 1.1676998740e-1, -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1, 2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1 }) |p| y = fma(y, x, splat(p));
    y = fma(y * x, z, e * splat(-2.12194440e-4));
    y = fma(-z, splat(0.5), y);
    return fma(e, splat(0.693359375), x + y);
}

/// avx_mathfun's sincos256_ps; its inputs here lie in [0, 2 pi).
fn sincos(input: V) struct { sin: V, cos: V } {
    const sign_mask: I = @splat(std.math.minInt(i32));
    const sign_sin0 = @as(I, @bitCast(input)) & sign_mask;
    var x = @abs(input);
    var j: I = @intFromFloat(x * splat(1.27323954473516));
    j = (j + @as(I, @splat(1))) & @as(I, @splat(~@as(i32, 1)));
    const y0: V = @floatFromInt(j);
    const sign_sin = sign_sin0 ^ ((j & @as(I, @splat(4))) << @splat(29));
    const poly = (j & @as(I, @splat(2))) == @as(I, @splat(0));
    x = fma(y0, splat(-0.78515625), x);
    x = fma(y0, splat(-2.4187564849853515625e-4), x);
    x = fma(y0, splat(-3.77489497744594108e-8), x);
    const sign_cos = (~(j - @as(I, @splat(2))) & @as(I, @splat(4))) << @splat(29);
    const z = x * x;
    var c = fma(splat(2.443315711809948e-5), z, splat(-1.388731625493765e-3));
    c = fma(c, z, splat(4.166664568298827e-2));
    c = fma(c * z, z, -(z * splat(0.5))) + splat(1);
    var s = fma(splat(-1.9515295891e-4), z, splat(8.3321608736e-3));
    s = fma(s, z, splat(-1.6666654611e-1));
    s = fma(s * z, x, x);
    // the source's masked selection, adds of +0 included
    const from_s = @select(f32, poly, s, splat(0));
    const from_c = @select(f32, poly, splat(0), c);
    const sin = from_c + from_s;
    const cos = (c - from_c) + (s - from_s);
    return .{ .sin = @bitCast(@as(I, @bitCast(sin)) ^ sign_sin), .cos = @bitCast(@as(I, @bitCast(cos)) ^ sign_cos) };
}

/// Box-Muller on 16 uniforms in place: element j and j + 8 from the pair (1 - u[j], u[j + 8]).
fn fill16(d: *[16]f32) void {
    const near = splat(1) - @as(V, d[0..8].*);
    const turn: V = d[8..16].*;
    const radius = @sqrt(splat(-2) * log(near));
    const sc = sincos(splat(@floatCast(2.0 * std.math.pi)) * turn);
    d[0..8].* = fma(radius * sc.cos, splat(1), splat(0));
    d[8..16].* = fma(radius * sc.sin, splat(1), splat(0));
}

/// torch.randn of `out.len` float32 values (16 or more) in torch's order, as manual_seed(seed) makes them.
pub fn randn(seed: u64, out: []f32) void {
    std.debug.assert(out.len >= 16);
    var t = Twister.init(seed);
    for (out) |*v| v.* = t.uniform();
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) fill16(out[i..][0..16]);
    if (out.len % 16 != 0) {
        const tail = out[out.len - 16 ..][0..16];
        for (tail) |*v| v.* = t.uniform();
        fill16(tail);
    }
}

/// The starting latents of a rows x columns grid, laid out rows x columns x 128 as tf-moss-image reads them.
pub fn latents(gpa: std.mem.Allocator, seed: u64, rows: usize, columns: usize, out: []f32) !void {
    const channels = 128;
    std.debug.assert(out.len == rows * columns * channels);
    const torch_order = try gpa.alloc(f32, out.len);
    defer gpa.free(torch_order);
    randn(seed, torch_order);
    for (0..channels) |ch| for (0..rows * columns) |at| {
        out[at * channels + ch] = torch_order[ch * rows * columns + at];
    };
}

test "the angle's 2 pi is torch's float of the double" {
    try std.testing.expectEqual(@as(u32, 0x40c90fdb), @as(u32, @bitCast(@as(f32, @floatCast(2.0 * std.math.pi)))));
}

test "torch's mt19937: seed 5489 gives the reference 10000th output" {
    var t = Twister.init(5489);
    for (0..9999) |_| _ = t.random();
    try std.testing.expectEqual(@as(u32, 4123659995), t.random());
}

test "latents equal x86 torch.randn's for every recorded seed and size (tools/zig/moss_image_case.py noise)" {
    const gpa = std.testing.allocator;
    const Case = struct { seed: u64, width: usize, height: usize, sha256: []const u8 };
    const Refs = struct { torch: []const u8, cpu: []const u8, cases: []const Case };
    const refs = try std.json.parseFromSlice(Refs, gpa, @embedFile("noise_torch.json"), .{});
    defer refs.deinit();
    try std.testing.expect(refs.value.cases.len >= 60);
    for (refs.value.cases) |c| {
        const rows = c.height / 16;
        const columns = c.width / 16;
        const out = try gpa.alloc(f32, rows * columns * 128);
        defer gpa.free(out);
        try latents(gpa, c.seed, rows, columns, out);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        if (!std.mem.eql(u8, &hex, c.sha256)) {
            std.debug.print("seed {d} at {d}x{d}: {s}, torch {s}\n", .{ c.seed, c.width, c.height, hex, c.sha256 });
            return error.TestUnexpectedResult;
        }
    }
}
