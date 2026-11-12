//! GPU check of k3_kda on Ling-tiny's KDA shape against kda.zig. No new kernel: the source is kimi/kda.metal.
const std = @import("std");
const mtl = @import("metal");
const kda = @import("kda");
const kimi_metal = @import("kimi_metal");

const kda_src = kimi_metal.common ++ "\n" ++ kimi_metal.kda[std.mem.indexOf(u8, kimi_metal.kda, "constant constexpr uint D").?..];

const KdaArgs = extern struct { heads: u32, head0: u32, proj_stride: u32, gate_stride: u32, out_stride: u32, log_rows: u32, lower_bound: f32, eps: f32 };
const KdaSeg = extern struct { first: u32, rows: u32, kept: u32, flags: u32, slot: u32 };
const KdaState = extern struct { S: u64, conv: u64, log_a: u64, log_k: u64, log_u: u64, log_x: u64 };
const KdaWeights = extern struct { conv_q: u64, conv_k: u64, conv_v: u64, A_log: u64, dt_bias: u64, o_norm: u64 };

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @truncate((bits >> 16) | 0x40);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn fill(seed: *u64, n: usize, a: std.mem.Allocator) ![]f32 {
    const x = try a.alloc(f32, n);
    for (x) |*v| {
        seed.* = seed.* *% 1103515245 +% 12345;
        v.* = @as(f32, @floatFromInt((seed.* >> 16) & 0x7fff)) / 16384.0 - 1.0;
    }
    return x;
}

fn putBf16(buf: mtl.Buffer, x: []const f32) void {
    const d = buf.slice(u16, x.len);
    for (x, d) |v, *o| o.* = toBf16(v);
}

fn putF32(buf: mtl.Buffer, x: []const f32) void {
    @memcpy(buf.slice(f32, x.len), x);
}

fn check(rows: u32) !void {
    const a = std.testing.allocator;
    const s = kda.Shape{ .heads = 16, .lower_bound = -5, .eps = 1e-6 };
    const H = s.heads;
    const W = s.width();
    const D = kda.dim;
    var seed: u64 = 0x6b63;
    const proj = try fill(&seed, rows * 3 * W, a);
    defer a.free(proj);
    const fgate = try fill(&seed, rows * W, a);
    defer a.free(fgate);
    const braw = try fill(&seed, rows * H, a);
    defer a.free(braw);
    const g2 = try fill(&seed, rows * W, a);
    defer a.free(g2);
    const conv_q = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_q);
    const conv_k = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_k);
    const conv_v = try fill(&seed, W * kda.taps, a);
    defer a.free(conv_v);
    const A_log = try fill(&seed, H, a);
    defer a.free(A_log);
    const dt_bias = try fill(&seed, W, a);
    defer a.free(dt_bias);
    const o_norm = try fill(&seed, D, a);
    defer a.free(o_norm);
    for (o_norm) |*v| v.* = 0.5 + 0.5 * v.*;
    for (proj) |*v| v.* = fromBf16(toBf16(v.*));
    for (fgate) |*v| v.* = fromBf16(toBf16(v.*));
    for (braw) |*v| v.* = fromBf16(toBf16(v.*));
    for (g2) |*v| v.* = fromBf16(toBf16(v.*));

    const want = try a.alloc(f32, rows * W);
    defer a.free(want);
    const S_host = try a.alloc(f32, H * D * D);
    defer a.free(S_host);
    @memset(S_host, 0);
    try kda.mix(want, S_host, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);

    const want64 = try a.alloc(f64, rows * W);
    defer a.free(want64);
    const S64 = try a.alloc(f64, H * D * D);
    defer a.free(S64);
    @memset(S64, 0);
    try kda.mixF64(want64, S64, proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, s, rows);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const src = kda_src;
    try std.testing.expect(std.mem.indexOf(u8, src, "kernel void k3_kda") != null);
    const lib = try mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "k3_kda", false);
    defer pipe.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const shared = mtl.ResourceOptions.shared;

    const b_proj = try device.buffer(proj.len * 2, shared);
    defer b_proj.deinit();
    const b_f = try device.buffer(fgate.len * 2, shared);
    defer b_f.deinit();
    const b_b = try device.buffer(braw.len * 2, shared);
    defer b_b.deinit();
    const b_g = try device.buffer(g2.len * 2, shared);
    defer b_g.deinit();
    const b_y = try device.buffer(want.len * 2, shared);
    defer b_y.deinit();
    putBf16(b_proj, proj);
    putBf16(b_f, fgate);
    putBf16(b_b, braw);
    putBf16(b_g, g2);
    @memset(b_y.slice(u16, want.len), 0);

    const b_seg = try device.buffer(@sizeOf(KdaSeg), shared);
    defer b_seg.deinit();
    b_seg.slice(KdaSeg, 1)[0] = .{ .first = 0, .rows = rows, .kept = 0, .flags = 1, .slot = 0 };

    const b_S = try device.buffer(H * D * D * 4, shared);
    defer b_S.deinit();
    const b_conv = try device.buffer(3 * 3 * W * 2, shared);
    defer b_conv.deinit();
    const dummy = try device.buffer(4, shared);
    defer dummy.deinit();
    @memset(b_S.slice(f32, H * D * D), 0);
    @memset(b_conv.slice(u16, 3 * 3 * W), 0);

    const b_cq = try device.buffer(conv_q.len * 4, shared);
    defer b_cq.deinit();
    const b_ck = try device.buffer(conv_k.len * 4, shared);
    defer b_ck.deinit();
    const b_cv = try device.buffer(conv_v.len * 4, shared);
    defer b_cv.deinit();
    const b_A = try device.buffer(A_log.len * 4, shared);
    defer b_A.deinit();
    const b_dt = try device.buffer(dt_bias.len * 4, shared);
    defer b_dt.deinit();
    const b_on = try device.buffer(o_norm.len * 4, shared);
    defer b_on.deinit();
    putF32(b_cq, conv_q);
    putF32(b_ck, conv_k);
    putF32(b_cv, conv_v);
    putF32(b_A, A_log);
    putF32(b_dt, dt_bias);
    putF32(b_on, o_norm);

    const st = KdaState{
        .S = b_S.gpuAddress(),
        .conv = b_conv.gpuAddress(),
        .log_a = dummy.gpuAddress(),
        .log_k = dummy.gpuAddress(),
        .log_u = dummy.gpuAddress(),
        .log_x = dummy.gpuAddress(),
    };
    const w = KdaWeights{
        .conv_q = b_cq.gpuAddress(),
        .conv_k = b_ck.gpuAddress(),
        .conv_v = b_cv.gpuAddress(),
        .A_log = b_A.gpuAddress(),
        .dt_bias = b_dt.gpuAddress(),
        .o_norm = b_on.gpuAddress(),
    };
    const args = KdaArgs{
        .heads = H,
        .head0 = 0,
        .proj_stride = 3 * W,
        .gate_stride = W,
        .out_stride = W,
        .log_rows = 1,
        .lower_bound = s.lower_bound,
        .eps = s.eps,
    };

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(b_proj, 0, 0);
    e.setBuffer(b_f, 0, 1);
    e.setBuffer(b_b, 0, 2);
    e.setBuffer(b_g, 0, 3);
    e.setBuffer(b_y, 0, 4);
    e.setBuffer(b_seg, 0, 5);
    e.setValue(st, 6);
    e.setValue(w, 7);
    e.setValue(args, 8);
    const rw = mtl.ResourceUsage.read | mtl.ResourceUsage.write;
    e.useResource(b_S, rw);
    e.useResource(b_conv, rw);
    e.useResource(dummy, mtl.ResourceUsage.read);
    e.useResource(b_cq, mtl.ResourceUsage.read);
    e.useResource(b_ck, mtl.ResourceUsage.read);
    e.useResource(b_cv, mtl.ResourceUsage.read);
    e.useResource(b_A, mtl.ResourceUsage.read);
    e.useResource(b_dt, mtl.ResourceUsage.read);
    e.useResource(b_on, mtl.ResourceUsage.read);
    e.dispatchGroups(mtl.Size.of(H, 1, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }

    const got = b_y.slice(u16, want.len);
    var worst_mix: f32 = 0;
    var worst_f64: f64 = 0;
    var mix_f64: f64 = 0;
    var host_max: f32 = 0;
    var gpu_max: f32 = 0;
    var ymax64: f64 = 0;
    for (want, want64, got) |w_y, w64, g_bits| {
        const g_y = fromBf16(g_bits);
        worst_mix = @max(worst_mix, @abs(w_y - g_y));
        worst_f64 = @max(worst_f64, @abs(w64 - @as(f64, g_y)));
        mix_f64 = @max(mix_f64, @abs(w64 - @as(f64, w_y)));
        host_max = @max(host_max, @abs(w_y));
        gpu_max = @max(gpu_max, @abs(g_y));
        ymax64 = @max(ymax64, @abs(w64));
    }
    // 2 * ymax * 2^-7 is a relative unit of the largest |y|, not a bf16 ulp.
    // Values in [1, 2) share one true bf16 step of 2^-7.
    const rel = ymax64 * std.math.pow(f64, 2, -7);
    std.debug.print("k3_kda vs mix {d:.6} vs mixF64 {d:.6} mix vs mixF64 {d:.6} host_max {d:.6} gpu_max {d:.6} ymax64 {d:.6} (16 heads, {d} rows)\n", .{ worst_mix, worst_f64, mix_f64, host_max, gpu_max, ymax64, rows });
    try std.testing.expect(worst_mix < 0.01);
    try std.testing.expect(worst_f64 <= 2 * rel);
}

test "k3_kda on Ling-tiny (16 heads, D=128, 4 taps) matches kda.mix on two rows" {
    try check(2);
}

test "k3_kda on Ling-tiny (16 heads, D=128, 4 taps) vs mix and mixF64 on four rows" {
    try check(4);
}

fn takeU32(b: []const u8, off: *usize) u32 {
    const v = std.mem.readInt(u32, b[off.*..][0..4], .little);
    off.* += 4;
    return v;
}

fn takeF32s(alloc: std.mem.Allocator, b: []const u8, off: *usize, n: usize) ![]f32 {
    const x = try alloc.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(x), b[off.* .. off.* + n * 4]);
    off.* += n * 4;
    return x;
}

fn checkDump(bytes: []const u8) !void {
    const a = std.testing.allocator;
    try std.testing.expect(std.mem.eql(u8, bytes[0..8], "TFLINGK1"));
    var off: usize = 8;
    const heads = takeU32(bytes, &off);
    const rows = takeU32(bytes, &off);
    const d = takeU32(bytes, &off);
    const tp = takeU32(bytes, &off);
    try std.testing.expectEqual(kda.dim, d);
    try std.testing.expectEqual(kda.taps, tp);
    const lower = @as(f32, @bitCast(takeU32(bytes, &off)));
    const eps = @as(f32, @bitCast(takeU32(bytes, &off)));
    off += 8;
    const s = kda.Shape{ .heads = heads, .lower_bound = lower, .eps = eps };
    const W = s.width();
    const H = s.heads;
    const D = kda.dim;
    const proj = try takeF32s(a, bytes, &off, rows * 3 * W);
    defer a.free(proj);
    const fgate = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(fgate);
    const braw = try takeF32s(a, bytes, &off, rows * H);
    defer a.free(braw);
    const g2 = try takeF32s(a, bytes, &off, rows * W);
    defer a.free(g2);
    const conv_q = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_q);
    const conv_k = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_k);
    const conv_v = try takeF32s(a, bytes, &off, W * kda.taps);
    defer a.free(conv_v);
    const A_log = try takeF32s(a, bytes, &off, H);
    defer a.free(A_log);
    const dt_bias = try takeF32s(a, bytes, &off, W);
    defer a.free(dt_bias);
    const o_norm = try takeF32s(a, bytes, &off, D);
    defer a.free(o_norm);
    const want_n = rows * W;
    const want64 = try a.alloc(f64, want_n);
    defer a.free(want64);
    @memcpy(std.mem.sliceAsBytes(want64), bytes[off .. off + want_n * 8]);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const lib = try mtl.Library.fromSource(device, kda_src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "k3_kda", false);
    defer pipe.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const shared = mtl.ResourceOptions.shared;

    const b_proj = try device.buffer(proj.len * 2, shared);
    defer b_proj.deinit();
    const b_f = try device.buffer(fgate.len * 2, shared);
    defer b_f.deinit();
    const b_b = try device.buffer(braw.len * 2, shared);
    defer b_b.deinit();
    const b_g = try device.buffer(g2.len * 2, shared);
    defer b_g.deinit();
    const b_y = try device.buffer(want_n * 2, shared);
    defer b_y.deinit();
    putBf16(b_proj, proj);
    putBf16(b_f, fgate);
    putBf16(b_b, braw);
    putBf16(b_g, g2);
    @memset(b_y.slice(u16, want_n), 0);

    const b_seg = try device.buffer(@sizeOf(KdaSeg), shared);
    defer b_seg.deinit();
    b_seg.slice(KdaSeg, 1)[0] = .{ .first = 0, .rows = rows, .kept = 0, .flags = 1, .slot = 0 };

    const b_S = try device.buffer(H * D * D * 4, shared);
    defer b_S.deinit();
    const b_conv = try device.buffer(3 * 3 * W * 2, shared);
    defer b_conv.deinit();
    const dummy = try device.buffer(4, shared);
    defer dummy.deinit();
    @memset(b_S.slice(f32, H * D * D), 0);
    @memset(b_conv.slice(u16, 3 * 3 * W), 0);

    const b_cq = try device.buffer(conv_q.len * 4, shared);
    defer b_cq.deinit();
    const b_ck = try device.buffer(conv_k.len * 4, shared);
    defer b_ck.deinit();
    const b_cv = try device.buffer(conv_v.len * 4, shared);
    defer b_cv.deinit();
    const b_A = try device.buffer(A_log.len * 4, shared);
    defer b_A.deinit();
    const b_dt = try device.buffer(dt_bias.len * 4, shared);
    defer b_dt.deinit();
    const b_on = try device.buffer(o_norm.len * 4, shared);
    defer b_on.deinit();
    putF32(b_cq, conv_q);
    putF32(b_ck, conv_k);
    putF32(b_cv, conv_v);
    putF32(b_A, A_log);
    putF32(b_dt, dt_bias);
    putF32(b_on, o_norm);

    const st = KdaState{
        .S = b_S.gpuAddress(),
        .conv = b_conv.gpuAddress(),
        .log_a = dummy.gpuAddress(),
        .log_k = dummy.gpuAddress(),
        .log_u = dummy.gpuAddress(),
        .log_x = dummy.gpuAddress(),
    };
    const w = KdaWeights{
        .conv_q = b_cq.gpuAddress(),
        .conv_k = b_ck.gpuAddress(),
        .conv_v = b_cv.gpuAddress(),
        .A_log = b_A.gpuAddress(),
        .dt_bias = b_dt.gpuAddress(),
        .o_norm = b_on.gpuAddress(),
    };
    const args = KdaArgs{
        .heads = H,
        .head0 = 0,
        .proj_stride = 3 * W,
        .gate_stride = W,
        .out_stride = W,
        .log_rows = 1,
        .lower_bound = s.lower_bound,
        .eps = s.eps,
    };

    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(b_proj, 0, 0);
    e.setBuffer(b_f, 0, 1);
    e.setBuffer(b_b, 0, 2);
    e.setBuffer(b_g, 0, 3);
    e.setBuffer(b_y, 0, 4);
    e.setBuffer(b_seg, 0, 5);
    e.setValue(st, 6);
    e.setValue(w, 7);
    e.setValue(args, 8);
    const rw = mtl.ResourceUsage.read | mtl.ResourceUsage.write;
    e.useResource(b_S, rw);
    e.useResource(b_conv, rw);
    e.useResource(dummy, mtl.ResourceUsage.read);
    e.useResource(b_cq, mtl.ResourceUsage.read);
    e.useResource(b_ck, mtl.ResourceUsage.read);
    e.useResource(b_cv, mtl.ResourceUsage.read);
    e.useResource(b_A, mtl.ResourceUsage.read);
    e.useResource(b_dt, mtl.ResourceUsage.read);
    e.useResource(b_on, mtl.ResourceUsage.read);
    e.dispatchGroups(mtl.Size.of(H, 1, 1), mtl.Size.of(256, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }

    const got = b_y.slice(u16, want_n);
    var worst: f64 = 0;
    var ymax: f64 = 0;
    var glm_miss: u32 = 0;
    for (want64) |w_y| ymax = @max(ymax, @abs(w_y));
    // Bound is 2 * ymax * 2^-7: a relative unit of the largest |y|, not a bf16 ulp.
    // Values in [1, 2) share one true bf16 step of 2^-7. Kimi k3_kda has no GPU vs
    // float64 gate in tree; this bound is proposed. CUDA GLM KDA vs float64 uses
    // |e|*2^-8 + rowmax*2^-8 (glm_ref_kda.zig); that per-element bound misses
    // values here that are not at ymax.
    const rel = ymax * std.math.pow(f64, 2, -7);
    const true_step = if (ymax > 0) std.math.pow(f64, 2, @floor(std.math.log2(ymax)) - 7.0) else 0;
    for (want64, got) |w_y, g_bits| {
        const err = @abs(w_y - @as(f64, fromBf16(g_bits)));
        worst = @max(worst, err);
        const glm = @abs(w_y) * std.math.pow(f64, 2, -8) + ymax * std.math.pow(f64, 2, -8);
        if (err > glm) glm_miss += 1;
    }
    const of_rel = if (rel > 0) worst / rel else 0;
    const true_steps = if (true_step > 0) worst / true_step else 0;
    std.debug.print("k3_kda vs dump worst abs {d:.6} ymax {d:.6} {d:.3} of (ymax x 2^-7) {d:.3} true bf16 steps glm_miss {d} (16 heads, {d} rows)\n", .{ worst, ymax, of_rel, true_steps, glm_miss, rows });
    try std.testing.expect(worst <= 2 * rel);
}

test "k3_kda vs kda_inner dump on 16 heads x 2 rows" {
    try checkDump(kda.dump_h16_r2);
}

test "k3_kda vs kda_inner dump on 16 heads x 4 rows" {
    try checkDump(kda.dump_h16_r4);
}
