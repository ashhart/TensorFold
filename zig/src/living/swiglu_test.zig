//! A SwiGLU pocket neuron on the host: fitted, written in 4-bit groups of 32 and read back, it fires as relu^2 would.
const std = @import("std");
const pocket_fit = @import("pocket_fit.zig");
const neuron = @import("neuron.zig");
const affine = @import("affine.zig");
const families = @import("families.zig");

const f = @import("test_families.zig").swiglu;
const d = 128;

fn bf(x: f32) u16 {
    return @truncate(@as(u32, @bitCast(x)) >> 16);
}

fn dot(a: []const f32, b: []const u16) f32 {
    return pocket_fit.dotBf16(a, b);
}

/// `x` written and read back as the family stores its experts.
fn stored(x: []f32) void {
    var w: [affine.wordsOf(f.quant, d)]u32 = undefined;
    var s: [d / f.quant.group]u16 = undefined;
    var b: [d / f.quant.group]u16 = undefined;
    affine.quantize(f.quant, x, &w, &s, &b);
    affine.dequantize(f.quant, &w, &s, &b, x);
}

test "a SwiGLU neuron fitted at margin 0.5 and stored at 4 bits in groups of 32 fires on its rows and stays near zero elsewhere" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(21);
    const r = prng.random();
    var common: [d]f32 = undefined;
    var lean: [d]f32 = undefined;
    for (&common, &lean) |*c, *l| {
        c.* = 1 + r.floatNorm(f32) * 0.1;
        l.* = r.floatNorm(f32);
    }
    const n_fact = 16;
    const n_rest = 300;
    const rows = try gpa.alloc(u16, (n_fact + n_rest) * d);
    defer gpa.free(rows);
    const xs = try gpa.alloc([]const u16, n_fact + n_rest);
    defer gpa.free(xs);
    const kinds = try gpa.alloc(pocket_fit.Kind, n_fact + n_rest);
    defer gpa.free(kinds);
    for (0..n_fact + n_rest) |i| {
        for (0..d) |k| {
            var v = common[k] + r.floatNorm(f32) * 0.5;
            if (i < n_fact) v += 1.5 * lean[k];
            rows[i * d + k] = bf(v);
        }
        xs[i] = rows[i * d ..][0..d];
        kinds[i] = if (i < n_fact) .fact else .other;
    }
    var u: [d]f32 = undefined;
    const got = try pocket_fit.fit(gpa, xs, kinds, d, &u, false, pocket_fit.fit_steps, 0, f.neuron.margin);
    try std.testing.expectEqual(@as(f32, 1), got.fact_on);
    try std.testing.expectEqual(@as(usize, 0), got.fires);
    try std.testing.expect(@max(got.steady_most, got.other_most) <= -f.neuron.margin + 0.05);
    // as the forward and the shards hold it: gate row sharp u, up row u / sharp, each quantized
    var g: [d]f32 = undefined;
    var w: [d]f32 = undefined;
    neuron.gateRow(f.neuron, &u, &g);
    neuron.upRow(f.neuron, &u, &w);
    stored(&g);
    stored(&w);
    var least: f32 = std.math.inf(f32);
    var most: f32 = 0;
    for (xs, kinds) |x, k| {
        const a = neuron.silu(dot(&g, x)) * dot(&w, x);
        if (k == .fact) {
            least = @min(least, a);
            // within a hair of relu(u x)^2 on its own rows, but for the rows' quantization
            try std.testing.expectApproxEqRel(neuron.act(.{ .act = .relu2 }, dot(&u, x), 0), a, 0.15);
        } else most = @max(most, @abs(a));
    }
    try std.testing.expect(least >= 0.7);
    try std.testing.expect(most <= 1e-3 * least);
    // the same rows through act(): u x and g x as the pocket reads them
    const x = xs[0];
    try std.testing.expectApproxEqRel(neuron.silu(dot(&g, x)) * dot(&w, x), neuron.act(f.neuron, f.neuron.sharp * dot(&w, x), dot(&g, x)), 1e-5);
}
