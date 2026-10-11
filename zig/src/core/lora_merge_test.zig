//! lora_merge on synthetic dense tensors and adapters: the merge arithmetic, its report and its names.
const std = @import("std");
const host4 = @import("affine4_host.zig");
const lm = @import("lora_merge.zig");

const gpa = std.testing.allocator;

fn alloc(n: usize, group: usize) !lm.Quant {
    return .{ .words = try gpa.alloc(u32, n / 8), .scales = try gpa.alloc(u16, n / group), .biases = try gpa.alloc(u16, n / group) };
}

fn free(q: lm.Quant) void {
    gpa.free(q.words);
    gpa.free(q.scales);
    gpa.free(q.biases);
}

fn same(a: lm.Quant, b: lm.Quant) bool {
    return std.mem.eql(u32, a.words, b.words) and std.mem.eql(u16, a.scales, b.scales) and std.mem.eql(u16, a.biases, b.biases);
}

/// A bf16 dense weight (as the public checkpoints ship) of pseudo-random values, as f32.
fn dense(n: usize, k: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const w = try gpa.alloc(f32, n * k);
    for (w) |*v| v.* = host4.f32of(host4.bf16of((prng.random().float(f32) - 0.5) * 0.2));
    return w;
}

fn factors(n: usize, k: usize, r: usize, seed: u64, a: []f32, b: []f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (a[0 .. r * k]) |*x| x.* = (prng.random().float(f32) - 0.5) * 0.1;
    for (b[0 .. n * r]) |*x| x.* = (prng.random().float(f32) - 0.5) * 0.1;
}

test "a quantization compared with itself has error ratios of exactly 1 at every group size" {
    for ([_]usize{ 32, 64, 128 }) |group| {
        const w = try dense(4, 256, 3);
        defer gpa.free(w);
        const q = try alloc(w.len, group);
        defer free(q);
        try std.testing.expect(try lm.quantize(group, w, q) == null);
        var a: [256]f32 = @splat(0);
        var b: [4]f32 = @splat(0);
        const rep = try lm.measure(gpa, group, 4, 256, w, .{ .a = &a, .b = &b, .r = 1, .scale = 1 }, lm.viewOf(q), lm.viewOf(q));
        try std.testing.expect(rep.withinTolerance() and rep.rmsRatio() == 1 and rep.worstRatio() == 1);
        try std.testing.expect(rep.merged.max_err > 0);
    }
}

test "a zero adapter is the plain quantization of the dense weights, byte for byte" {
    for ([_]usize{ 32, 64, 128 }) |group| {
        const w = try dense(8, 256, 7);
        defer gpa.free(w);
        const plain = try alloc(w.len, group);
        defer free(plain);
        try std.testing.expect(try lm.quantize(group, w, plain) == null);
        var a: [4 * 256]f32 = undefined;
        var b: [8 * 4]f32 = undefined;
        factors(8, 256, 4, 1, &a, &b);
        @memset(&b, 0);
        const out = try alloc(w.len, group);
        defer free(out);
        var bad: usize = 0;
        try lm.mergeDense(gpa, group, 8, 256, w, .{ .a = &a, .b = &b, .r = 4, .scale = 2 }, out, &bad);
        try std.testing.expect(same(plain, out));
        try std.testing.expect(lm.agreement(group, lm.viewOf(plain), lm.viewOf(out)).identical());
    }
}

test "a random adapter equals the quantization of an independently summed dense + delta" {
    for ([_]usize{ 32, 64, 128 }) |group| {
        const n = 16;
        const k = 256;
        const r = 16;
        const w = try dense(n, k, 21);
        defer gpa.free(w);
        var a: [r * k]f32 = undefined;
        var b: [n * r]f32 = undefined;
        factors(n, k, r, 5, &a, &b);
        const l: lm.Lora = .{ .a = &a, .b = &b, .r = r, .scale = 2 };
        const out = try alloc(w.len, group);
        defer free(out);
        var bad: usize = 0;
        try lm.mergeDense(gpa, group, n, k, w, l, out, &bad);
        // the same sum written out plainly, in FP64
        const sum = try gpa.alloc(f32, w.len);
        defer gpa.free(sum);
        for (0..n) |o| for (0..k) |c| {
            var d: f64 = 0;
            for (0..r) |j| d += @as(f64, b[o * r + j]) * @as(f64, a[j * k + c]);
            sum[o * k + c] = @floatCast(@as(f64, w[o * k + c]) + 2 * d);
        };
        const expect = try alloc(w.len, group);
        defer free(expect);
        try std.testing.expect(try lm.quantize(group, sum, expect) == null);
        try std.testing.expect(same(expect, out));
        const shipped = try alloc(w.len, group);
        defer free(shipped);
        try std.testing.expect(try lm.quantize(group, w, shipped) == null);
        const rep = try lm.measure(gpa, group, n, k, w, l, lm.viewOf(shipped), lm.viewOf(out));
        try std.testing.expect(rep.merged.rms_err <= rep.merged.max_err and rep.merged.max_err > 0);
    }
}

fn errors(worst: f64, rms: f64) lm.Error64 {
    return .{ .worst_col = 0, .worst_col_err = worst, .worst_col_rms = rms, .rms_err = rms, .max_err = worst };
}

test "the tolerance gates the aggregate rms only: a worst column 15% above shipped passes and prints its ratio" {
    const shipped = errors(2.478e-2, 2.0e-3);
    const near: lm.Report = .{ .merged = errors(2.511e-2, 2.0e-3 * 1.0093), .shipped = shipped };
    try std.testing.expect(near.worstRatio() > 1 and near.withinTolerance());
    const wide: lm.Report = .{ .merged = errors(2.478e-2 * 1.15, 2.0e-3 * 1.0121), .shipped = shipped };
    try std.testing.expectApproxEqAbs(@as(f64, 1.15), wide.worstRatio(), 1e-12);
    try std.testing.expect(wide.withinTolerance());
    const edge: lm.Report = .{ .merged = errors(2.478e-2, 2.0e-3 * 1.05), .shipped = shipped };
    try std.testing.expect(edge.withinTolerance());
    const noisier: lm.Report = .{ .merged = errors(2.478e-2, 2.0e-3 * 1.06), .shipped = shipped };
    try std.testing.expect(!noisier.withinTolerance());
    try std.testing.expect((lm.Report{ .merged = shipped, .shipped = shipped }).withinTolerance());
}

test "recompute counts the groups of an output that differ from dense + delta" {
    const w = try dense(4, 128, 9);
    defer gpa.free(w);
    var a: [128]f32 = undefined;
    var b: [4]f32 = undefined;
    factors(4, 128, 1, 3, &a, &b);
    const l: lm.Lora = .{ .a = &a, .b = &b, .r = 1, .scale = 2 };
    const out = try alloc(w.len, 64);
    defer free(out);
    var bad: usize = 0;
    try lm.mergeDense(gpa, 64, 4, 128, w, l, out, &bad);
    try std.testing.expectEqual(@as(usize, 0), try lm.recompute(gpa, 64, 4, 128, w, l, lm.viewOf(out), &bad));
    out.words[9] ^= 1; // one code of one group
    out.scales[5] ^= 1; // one scale of another
    try std.testing.expectEqual(@as(usize, 2), try lm.recompute(gpa, 64, 4, 128, w, l, lm.viewOf(out), &bad));
}

test "PEFT names map to checkpoint modules and only served modules pass" {
    const k = lm.parseAdapterKey("base_model.model.backbone.layers.3.mixer.in_proj.lora_A.weight").?;
    try std.testing.expectEqualStrings("backbone.layers.3.mixer.in_proj", k.module);
    try std.testing.expectEqual(lm.Side.a, k.side);
    const d = lm.parseAdapterKey("base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight").?;
    try std.testing.expectEqualStrings("model.layers.0.self_attn.q_proj", d.module);
    try std.testing.expectEqual(lm.Side.b, d.side);
    try std.testing.expect(lm.parseAdapterKey("backbone.layers.3.mixer.in_proj.lora_A.weight") == null);
    try std.testing.expect(lm.parseAdapterKey("base_model.model.x.lora_magnitude_vector") == null);
    try std.testing.expect(lm.parseAdapterKey("base_model.model.x.lora_embedding_A") == null);
    for ([_][]const u8{ "backbone.layers.1.mixer.out_proj", "backbone.layers.1.mixer.shared_experts.down_proj", "model.layers.2.self_attn.o_proj" }) |m| try std.testing.expect(lm.served(m));
    for ([_][]const u8{ "backbone.layers.1.mixer.experts.0.up_proj", "model.layers.2.self_attn.qkv_proj", "model.layers.2.mlp.gate_proj", "lm_head" }) |m| try std.testing.expect(!lm.served(m));
}

test "shapes that do not fit the projection, and a group the engine does not serve, are refused" {
    const w = try dense(4, 64, 2);
    defer gpa.free(w);
    const out = try alloc(w.len, 64);
    defer free(out);
    const z64: [64]f32 = @splat(0);
    const z8: [8]f32 = @splat(0);
    const z4: [4]f32 = @splat(0);
    var bad: usize = 0;
    try std.testing.expectError(error.BadShape, lm.mergeDense(gpa, 64, 4, 64, w, .{ .a = z64[0..32], .b = &z4, .r = 1, .scale = 1 }, out, &bad));
    try std.testing.expectError(error.BadShape, lm.mergeDense(gpa, 64, 4, 64, w, .{ .a = &z64, .b = &z8, .r = 1, .scale = 1 }, out, &bad));
    try std.testing.expectError(error.BadShape, lm.mergeDense(gpa, 64, 4, 64, w[0..128], .{ .a = &z64, .b = &z4, .r = 1, .scale = 1 }, out, &bad));
    try std.testing.expectError(error.BadGroup, lm.mergeDense(gpa, 48, 4, 64, w, .{ .a = &z64, .b = &z4, .r = 1, .scale = 1 }, out, &bad));
}

test "error ratios: two exact zeros are equal, a positive error over zero fails, a non-finite metric fails" {
    const zero = errors(0, 0);
    const equal: lm.Report = .{ .merged = zero, .shipped = zero };
    try std.testing.expectEqual(@as(f64, 1), equal.rmsRatio());
    try std.testing.expectEqual(@as(f64, 1), equal.worstRatio());
    try std.testing.expect(equal.withinTolerance());
    const worse: lm.Report = .{ .merged = errors(1e-3, 1e-3), .shipped = zero };
    try std.testing.expect(!worse.withinTolerance());
    const nan: lm.Report = .{ .merged = errors(std.math.nan(f64), 1), .shipped = errors(1, 1) };
    try std.testing.expect(!nan.withinTolerance());
}

test "a B*A whose FP64 sum is exactly zero gives an exactly zero delta, scaled only after the sum" {
    const a: f32 = 3 * 1048576;
    const b: f32 = 7 * 1048576;
    // row of B is (a, b); column of A is (b, -a): a*b - b*a is exactly zero, and 0.1 * 0 is too
    var af: [128]f32 = undefined;
    for (0..64) |c| {
        af[c] = b;
        af[64 + c] = -a;
    }
    const bf = [_]f32{ a, b };
    const l: lm.Lora = .{ .a = &af, .b = &bf, .r = 2, .scale = 0.1 };
    var delta: [64]f64 = undefined;
    lm.deltaRow(l, 64, 0, &delta);
    for (delta) |d| try std.testing.expectEqual(@as(f64, 0), d);
    // and the merge is then the plain quantization of the dense weights
    const w = try dense(1, 64, 5);
    defer gpa.free(w);
    const plain = try alloc(64, 64);
    defer free(plain);
    try std.testing.expect(try lm.quantize(64, w, plain) == null);
    const out = try alloc(64, 64);
    defer free(out);
    var bad: usize = 0;
    try lm.mergeDense(gpa, 64, 1, 64, w, l, out, &bad);
    try std.testing.expect(same(plain, out));
}

test "the contract is FP64 accumulation then one scaling: a four-term sum that is not exact keeps its residual" {
    // B = (2^100, 1, -2^100, -1) against a column of ones sums to -1 in order, since 2^100 + 1 rounds back to 2^100
    const big: f32 = 1267650600228229401496703205376.0;
    const bf = [_]f32{ big, 1, -big, -1 };
    const af: [4]f32 = @splat(1);
    const l: lm.Lora = .{ .a = &af, .b = &bf, .r = 4, .scale = 0.1 };
    var delta: [1]f64 = undefined;
    lm.deltaRow(l, 1, 0, &delta);
    try std.testing.expectEqual(@as(f64, -1) * 0.1, delta[0]);
}
