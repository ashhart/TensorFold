//! mHC split and collapse FP64 ideal-reference tests (DeepSeek-V4.1-Flash inference, MIT).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const h = @import("operator_hc_fp64.zig");
const mhc = tf.deepseek_v41.mhc;
const Tensor = tf.checkpoint_metal.Tensor;
const Hc = tf.deepseek_v41_weights.Hc;
const Config = tf.deepseek_v41_config.Config;

// The comptime parameterization must keep the FP32 source contract bitwise against the old algorithm.
test "mHC fp32 reference path stays bit-identical under the comptime parameterization" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, h.released_config);
    const fixture = try std.json.parseFromSlice(h.Fixture, gpa, @embedFile(h.fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try h.decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_scale = try h.decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try h.decodeF32(gpa, fixture.value.ffn_base_f32);
    const scale = try device.buffer(12, mtl.ResourceOptions.shared);
    defer scale.deinit();
    h.upload(f32, scale, ffn_scale);
    const base = try device.buffer(h.mix_count * 4, mtl.ResourceOptions.shared);
    defer base.deinit();
    h.upload(f32, base, ffn_base);
    var lcg = h.Lcg{};
    const weight = try device.buffer(h.mix_count * h.stream_width * 4, mtl.ResourceOptions.shared);
    defer weight.deinit();
    const weight_host = try gpa.alloc(f32, h.mix_count * h.stream_width);
    for (weight_host) |*v| v.* = lcg.uniform() * 0.03;
    h.upload(f32, weight, weight_host);
    const hc = Hc{
        .projection = .{ .buffer = weight, .offset = 0, .bytes = h.mix_count * h.stream_width * 4, .dtype = .f32 },
        .scale = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 },
        .base = .{ .buffer = base, .offset = 0, .bytes = h.mix_count * 4, .dtype = .f32 },
    };
    for (0..8) |r| {
        const row = eight_rows[r * h.width ..][0..h.width];
        var stream: [h.stream_width]u16 = undefined;
        for (0..h.copies) |c| @memcpy(stream[c * h.width ..][0..h.width], row);
        const parameterized = math.mixesOf(f32, &stream, hc, cfg);
        const legacy = mixesLegacy(&stream, hc, cfg);
        try std.testing.expect(std.meta.eql(parameterized, legacy));
    }
}

fn mixesLegacy(x: []const u16, hc: Hc, c: Config) math.Mix {
    @setFloatMode(.strict);
    var projected: [24]f32 = undefined;
    math.denseOf(f32, hc.projection, x, &projected);
    var squares: f64 = 0;
    for (x) |v| squares += @as(f64, math.f(v)) * math.f(v);
    const mean = math.meanFromWide(squares, x.len);
    const r = math.idealRsqrt(mean + c.eps);
    var m: math.Mix = undefined;
    for (0..4) |i| {
        m.pre[i] = math.sigmoid(@mulAdd(f32, projected[i] * r, math.value(hc.scale, 0), math.value(hc.base, i))) + c.hc_eps;
        m.post[i] = 2 * math.sigmoid(@mulAdd(f32, projected[i + 4] * r, math.value(hc.scale, 1), math.value(hc.base, i + 4)));
    }
    for (0..4) |i| {
        var high: f32 = -std.math.inf(f32);
        for (0..4) |j| {
            const at = i * 4 + j;
            m.comb[at] = @mulAdd(f32, projected[at + 8] * r, math.value(hc.scale, 2), math.value(hc.base, at + 8));
            high = @max(high, m.comb[at]);
        }
        for (0..4) |j| {
            m.comb[i * 4 + j] = @exp(m.comb[i * 4 + j] - high);
        }
        const denominator = (m.comb[i * 4] + m.comb[i * 4 + 2]) +
            (m.comb[i * 4 + 1] + m.comb[i * 4 + 3]);
        for (0..4) |j| m.comb[i * 4 + j] = m.comb[i * 4 + j] / denominator + c.hc_eps;
    }
    for (0..c.hc_sinkhorn_iters) |iteration| {
        if (iteration != 0) for (0..4) |i| {
            const denominator = ((m.comb[i * 4] + m.comb[i * 4 + 2]) +
                (m.comb[i * 4 + 1] + m.comb[i * 4 + 3])) + c.hc_eps;
            for (0..4) |j| m.comb[i * 4 + j] /= denominator;
        };
        for (0..4) |j| {
            const denominator = ((m.comb[j] + m.comb[8 + j]) +
                (m.comb[4 + j] + m.comb[12 + j])) + c.hc_eps;
            for (0..4) |i| m.comb[i * 4 + j] /= denominator;
        }
    }
    return m;
}

// Split-stage isolation: identical projected mixes and inverse, so only unary or reduction boundaries differ.
test "mHC split unary boundaries against the FP64 ideal on retained arguments" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, h.released_config);

    const rows = 7 + 65; // retained arguments plus every integer exponent input -32..32
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const mixes_host = try gpa.alloc(f32, rows * h.mix_count);
    @memset(mixes_host, 0);
    // Rows 0..3: the 16 retained unary arguments rotated through pre, post and comb positions.
    for (0..4) |i| {
        for (0..4) |j| {
            mixes_host[i * h.mix_count + j] = @bitCast(h.retained_exp_argument_bits[(4 * i + j) % 16]);
            mixes_host[i * h.mix_count + 4 + j] = @bitCast(h.retained_exp_argument_bits[(4 * i + j) % 16]);
        }
        for (0..16) |q| mixes_host[i * h.mix_count + 8 + q] = @bitCast(h.retained_exp_argument_bits[(i + q) % 16]);
    }
    // Row 4: the documented controlling exponent input as a post argument (exp(+1.439...) path).
    mixes_host[4 * h.mix_count + 5] = -h.controlling_exponent_input;
    // Row 5: comb affine arguments spanning the exponent range.
    mixes_host[5 * h.mix_count + 8] = -32;
    mixes_host[5 * h.mix_count + 9] = -16;
    mixes_host[5 * h.mix_count + 10] = 16;
    mixes_host[5 * h.mix_count + 11] = 32;
    // Row 6: the same span through the sigmoid pre/post arguments.
    const span = [_]f32{ -32, -16, 16, 32 };
    for (span, 0..) |v, j| {
        mixes_host[6 * h.mix_count + j] = v;
        mixes_host[6 * h.mix_count + 4 + j] = v;
    }

    // Rows 7..69: one integer exponent input per row at a pre, post and comb position each.
    for (0..65) |i| {
        const r = 7 + i;
        const v: f32 = @floatFromInt(@as(i32, @intCast(i)) - 32);
        mixes_host[r * h.mix_count] = v;
        mixes_host[r * h.mix_count + 4] = v;
        mixes_host[r * h.mix_count + 8] = v;
    }

    const mixes = try b.get(f32, rows * h.mix_count);
    h.upload(f32, mixes, mixes_host);
    const inv = try b.get(f32, rows);
    @memset(inv.slice(f32, rows), 1.0);
    const scale = try b.get(f32, 3);
    @memset(scale.slice(f32, 3), 1.0);
    const base = try b.get(f32, h.mix_count);
    const pre = try b.get(f32, rows * 4);
    const post = try b.get(f32, rows * 4);
    const comb = try b.get(f32, rows * 16);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeSplit(e, &p, h.view(mixes), h.view(inv), h.view(scale), h.view(base), h.view(pre), h.view(post), h.view(comb), .{ .rows = rows, .eps = cfg.hc_eps });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    const scale_tensor: Tensor = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 };
    const base_tensor: Tensor = .{ .buffer = base, .offset = 0, .bytes = h.mix_count * 4, .dtype = .f32 };
    const hc = Hc{ .projection = base_tensor, .scale = scale_tensor, .base = base_tensor };

    var max_ulp_metal: f64 = 0;
    var max_ulp_f32: f64 = 0;
    var worst_name: []const u8 = "";
    const stages = [_]struct { name: []const u8, offset: usize, len: usize }{
        .{ .name = "pre", .offset = 0, .len = 4 },
        .{ .name = "post", .offset = 4, .len = 4 },
        .{ .name = "comb", .offset = 8, .len = 16 },
    };
    for (stages) |stage| {
        for (0..rows) |r| {
            const m: []const f32 = mixes_host[r * h.mix_count ..][0..h.mix_count];
            var m64: [h.mix_count]f64 = undefined;
            for (m, 0..) |v, i| m64[i] = v;
            const mix64 = math.splitOf(f64, &m64, 1.0, hc, cfg);
            const mix32 = math.splitOf(f32, m, 1.0, hc, cfg);
            const src64: []const f64 = if (std.mem.eql(u8, stage.name, "pre")) &mix64.pre else if (std.mem.eql(u8, stage.name, "post")) &mix64.post else &mix64.comb;
            const src32: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) &mix32.pre else if (std.mem.eql(u8, stage.name, "post")) &mix32.post else &mix32.comb;
            const metal: []const f32 = if (std.mem.eql(u8, stage.name, "pre")) pre.slice(f32, rows * 4)[r * 4 ..][0..4] else if (std.mem.eql(u8, stage.name, "post")) post.slice(f32, rows * 4)[r * 4 ..][0..4] else comb.slice(f32, rows * 16)[r * 16 ..][0..16];
            const d = h.deltaF32VsF64(metal[0..stage.len], src64[0..stage.len]);
            const d32 = h.deltaF32VsF64(src32[0..stage.len], src64[0..stage.len]);
            if (d.max_ulp > max_ulp_metal) {
                max_ulp_metal = d.max_ulp;
                worst_name = stage.name;
            }
            if (d32.max_ulp > max_ulp_f32) max_ulp_f32 = d32.max_ulp;
            std.debug.print("HC64 {{\"case\":\"split_args\",\"stage\":\"{s}\",\"row\":{d},\"nrms_metal_vs_f64\":{d},\"max_ulp_metal_vs_f64\":{d:.4},\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ stage.name, r, d.nrms, d.max_ulp, d32.max_ulp });
        }
    }
    std.debug.print("HC64 {{\"case\":\"split_args\",\"summary\":true,\"max_ulp_metal_vs_f64\":{d:.4},\"worst_stage\":\"{s}\",\"max_ulp_f32_vs_f64\":{d:.4}}}\n", .{ max_ulp_metal, worst_name, max_ulp_f32 });
    // Metal may exceed the FP32 reference by one exp ULP; cancelled Sinkhorn cells inflate ULPs, NRMS stays below 1e-7.
    try std.testing.expect(max_ulp_metal <= max_ulp_f32 + 2.0);
    try std.testing.expect(max_ulp_metal < 48.0);
}

// Retained cancellation fixtures: the kernel keeps the FP32 accumulation contract; the ideal shows the half-ULP gap.
test "mHC collapse and post keep source FP32 accumulation under cancellation" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, h.released_config);

    var b = h.Buffers{ .device = device };
    defer b.deinit();
    // Collapse fixture: stream {3,3,0,0} (width 1), previous_pre {-1, 1+2^-23, 0, 0}.
    const stream_a = try b.get(u16, 4);
    const values_a = [_]u16{ math.b(3), math.b(3), math.b(0), math.b(0) };
    h.upload(u16, stream_a, &values_a);
    const previous_a = try b.get(f32, 4);
    const coeff = 1 + 0x1p-23;
    const previous_values = [_]f32{ -1, coeff, 0, 0 };
    h.upload(f32, previous_a, &previous_values);
    const collapsed = try b.get(u16, 1);

    // Post fixture: stream {1,1,1,1}, branch {1}, post {-1,0,0,0}, comb column 0 = {1, 2^-24, 2^-24, 0}, two half-ULPs.
    const stream_b = try b.get(u16, 4);
    const values_b = [_]u16{ math.b(1), math.b(1), math.b(1), math.b(1) };
    h.upload(u16, stream_b, &values_b);
    const branch_b = try b.get(u16, 1);
    const branch_values = [_]u16{math.b(1)};
    h.upload(u16, branch_b, &branch_values);
    const post_b = try b.get(f32, 4);
    const post_values = [_]f32{ -1, 0, 0, 0 };
    h.upload(f32, post_b, &post_values);
    const comb_b = try b.get(f32, 16);
    comb_b.slice(f32, 16)[0] = 1;
    comb_b.slice(f32, 16)[4] = 0x1p-24;
    comb_b.slice(f32, 16)[8] = 0x1p-24;
    const y_b = try b.get(u16, 4);

    // Branch-product fixture: stream {3,0,0,0}, post {1+2^-23,...}, comb[0] = -1, branch {3}.
    const stream_c = try b.get(u16, 4);
    const values_c = [_]u16{ math.b(3), math.b(0), math.b(0), math.b(0) };
    h.upload(u16, stream_c, &values_c);
    const branch_c = try b.get(u16, 1);
    const branch_values_c = [_]u16{math.b(3)};
    h.upload(u16, branch_c, &branch_values_c);
    const post_c = try b.get(f32, 4);
    const post_values_c = [_]f32{ coeff, 0, 0, 0 };
    h.upload(f32, post_c, &post_values_c);
    const comb_c = try b.get(f32, 16);
    comb_c.slice(f32, 16)[0] = -1;
    const y_c = try b.get(u16, 4);

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    mhc.encodeCollapse(e, &p, h.view(stream_a), h.view(previous_a), h.view(collapsed), .{ .rows = 1, .width = 1, .stream_stride = 4, .y_stride = 1 });
    mhc.encodePost(e, &p, h.view(branch_b), h.view(stream_b), h.view(post_b), h.view(comb_b), h.view(y_b), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    mhc.encodePost(e, &p, h.view(branch_c), h.view(stream_c), h.view(post_c), h.view(comb_c), h.view(y_c), .{ .rows = 1, .width = 1, .branch_stride = 1, .residual_stride = 4, .y_stride = 4 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.MhcFp64GpuFailure;

    // Source-contract expectations (math.zig retained boundaries).
    try std.testing.expectEqual(@as(u16, 0x3500), collapsed.slice(u16, 1)[0]);
    try std.testing.expectEqual(@as(u16, 0), y_b.slice(u16, 4)[0]);
    try std.testing.expectEqual(@as(u16, 0x3500), y_c.slice(u16, 4)[0]);

    // Reference columns for the same fixtures.
    var collapsed32: [1]u16 = undefined;
    var collapsed64: [1]u16 = undefined;
    math.collapseOf(f32, &values_a, .{ -1, coeff, 0, 0 }, &collapsed32);
    math.collapseOf(f64, &values_a, .{ -1, @as(f64, coeff), 0, 0 }, &collapsed64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"collapse_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ collapsed.slice(u16, 1)[0], collapsed32[0], collapsed64[0] });
    try std.testing.expectEqual(collapsed32[0], collapsed.slice(u16, 1)[0]);

    const mix_b = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var y32: [4]u16 = undefined;
    var h32 = values_b;
    math.expandOf(f32, &h32, &branch_values, mix_b);
    @memcpy(&y32, &h32);
    const mix_b64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ -1, 0, 0, 0 }, .comb = .{ 1, 0, 0, 0, 0x1p-24, 0, 0, 0, 0x1p-24, 0, 0, 0, 0, 0, 0, 0 } };
    var h64 = values_b;
    math.expandOf(f64, &h64, &branch_values, mix_b64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"post_half_ulps\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_b.slice(u16, 4)[0], y32[0], h64[0] });
    try std.testing.expectEqual(y32[0], y_b.slice(u16, 4)[0]);

    const mix_c = math.MixOf(f32){ .pre = .{ 1, 0, 0, 0 }, .post = .{ coeff, 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc32 = values_c;
    math.expandOf(f32, &hc32, &branch_values_c, mix_c);
    const mix_c64 = math.MixOf(f64){ .pre = .{ 1, 0, 0, 0 }, .post = .{ @as(f64, coeff), 0, 0, 0 }, .comb = .{ -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
    var hc64 = values_c;
    math.expandOf(f64, &hc64, &branch_values_c, mix_c64);
    std.debug.print("HC64 {{\"case\":\"cancellation\",\"fixture\":\"branch_product\",\"metal\":\"0x{x}\",\"f32_ref\":\"0x{x}\",\"f64_ideal\":\"0x{x}\"}}\n", .{ y_c.slice(u16, 4)[0], hc32[0], hc64[0] });
    try std.testing.expectEqual(hc32[0], y_c.slice(u16, 4)[0]);
    _ = cfg;
}
