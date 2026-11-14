//! The chain kernels against the unfused dispatches they replace: every output bit must agree.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const support = @import("exl3_projection_gpu_test_support.zig");
const k = support.k;
const n = support.n;
const Owned = support.Owned;
const complete = support.complete;

/// A matrix with its scales as packed signs (the checkpoint's sign-only form), owning the sign buffers.
pub const Signed = struct {
    matrix: ex.Matrix,
    su: mtl.Buffer,
    sv: mtl.Buffer,

    pub fn init(device: mtl.Device, w: Owned, su_word: u16, sv_word: u16) !Signed {
        const su = try device.buffer(k / 8, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(n / 8, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        @memset(su.slice(i16, k / 16), @bitCast(su_word));
        @memset(sv.slice(i16, n / 16), @bitCast(sv_word));
        var m = w.matrix;
        m.suh = .{ .buffer = su };
        m.svh = .{ .buffer = sv };
        m.suh_format = .packed_signs;
        m.svh_format = .packed_signs;
        return .{ .matrix = m, .su = su, .sv = sv };
    }
    pub fn deinit(s: Signed) void {
        s.su.deinit();
        s.sv.deinit();
    }
};

/// wo_a (two indexed groups) into wo_b, unfused and fused.
fn woChain(first: ex.Matrix, second: ex.Matrix, next: ex.Matrix) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (!ex.chain.rotateOutInReady(&p)) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const groups = 2;
    const descriptors = try device.buffer(groups * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, groups)[0] = try ex.descriptor(first);
    descriptors.slice(ex.ExpertDescriptor, groups)[1] = try ex.descriptor(second);
    const ids = try device.buffer(groups * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, groups), &[_]u32{ 0, 1 });
    const x = try support.input(device, groups);
    defer x.deinit();
    const grouped_ref = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped_ref.deinit();
    const grouped = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped.deinit();
    const y_ref = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    @memset(grouped_ref.slice(u16, groups * n), 0);
    @memset(grouped.slice(u16, groups * n), 0);
    const scratch = try device.buffer(try ex.scratchBytes(k, n, groups) + try ex.scratchBytes(groups * n, n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const experts: ex.Experts = .{ .descriptors = .{ .buffer = descriptors }, .count = groups, .k = k, .n = n };
    const rows: ex.Rows = .{ .count = groups, .x_stride = k + 16, .y_stride = n, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 }, .output_rows = groups };
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, first);
    ex.useMatrix(e, second);
    // The unfused chain: wo_a, then wo_b over the stored bf16 rows.
    try ex.encodeIndexed(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = grouped_ref }, .{ .buffer = scratch }, rows);
    try ex.encode(&p, e, next, .{ .buffer = grouped_ref }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n });
    // The fused chain.
    try ex.chain.encodeIndexed(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = grouped }, .{ .buffer = scratch }, rows, .{ .next = .{ .su = next.suh, .su_signs = next.suh_format == .packed_signs, .k = groups * n } });
    try ex.encode(&p, e, next, .{ .buffer = scratch }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n, .prerotated = true });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, grouped_ref.slice(u16, groups * n), grouped.slice(u16, groups * n));
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, n), y.slice(u16, n));
}

test "wo_a epilogue fused with wo_b's rotate_in keeps every bit (f16 scales, biases)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w0 = try Owned.init(device, 6, .mul1, 21);
    defer w0.deinit();
    const w1 = try Owned.init(device, 6, .mul1, 22);
    defer w1.deinit();
    const wb = try Owned.init(device, 6, .mul1, 23);
    defer wb.deinit();
    try woChain(w0.matrix, w1.matrix, wb.matrix);
}

test "wo_a epilogue fused with wo_b's rotate_in keeps every bit (packed sign scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w0 = try Owned.init(device, 4, .inst3, 31);
    defer w0.deinit();
    const w1 = try Owned.init(device, 4, .inst3, 32);
    defer w1.deinit();
    const wb = try Owned.init(device, 4, .inst3, 33);
    defer wb.deinit();
    const s0 = try Signed.init(device, w0, 0x9635, 0x4a17);
    defer s0.deinit();
    const s1 = try Signed.init(device, w1, 0x1b2d, 0xc3a5);
    defer s1.deinit();
    const sb = try Signed.init(device, wb, 0x5f01, 0x8e3c);
    defer sb.deinit();
    try woChain(s0.matrix, s1.matrix, sb.matrix);
}

/// dsv41_rms in place, then the projection's own rotate_in, against the one-dispatch norm + rotate_in.
fn qNorm(matrix: ex.Matrix, rows: u32, weight_f32: bool, scale: f32) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (ex.chain.rmsRotateInWidth(&p) < k) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_ops, opts);
    defer lib.deinit();
    const rms = try mtl.Pipeline.init(device, lib, "dsv41_rms", false);
    defer rms.deinit();
    const stride = k + 16;
    const x_ref = try device.buffer(rows * stride * 2, mtl.ResourceOptions.shared);
    defer x_ref.deinit();
    const x = try device.buffer(rows * stride * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51a7);
    for (x_ref.slice(u16, rows * stride)) |*v| v.* = support.bf16((prng.random().float(f32) - 0.5) * scale);
    @memcpy(x.slice(u16, rows * stride), x_ref.slice(u16, rows * stride));
    const weight = try device.buffer(k * 4, mtl.ResourceOptions.shared);
    defer weight.deinit();
    for (0..k) |i| {
        const value = 0.25 + prng.random().float(f32) * 2;
        if (weight_f32) weight.slice(f32, k)[i] = value else weight.slice(u16, 2 * k)[i] = support.bf16(value);
    }
    const y_ref = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const eps: f32 = 1e-6;
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    e.setPipeline(rms);
    e.setBuffer(x_ref, 0, 0);
    e.setBuffer(weight, 0, 1);
    e.setBuffer(x_ref, 0, 2);
    e.setValue(extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 }{ .rows = rows, .width = k, .x_stride = stride, .y_stride = stride, .eps = eps, .weight_f32 = @intFromBool(weight_f32) }, 3);
    e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    e.barrier();
    try ex.encode(&p, e, matrix, .{ .buffer = x_ref }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = stride, .y_stride = n });
    try ex.chain.encodeRmsRotateIn(&p, e, .{ .buffer = x }, .{ .buffer = weight }, .{ .buffer = x }, matrix.suh, .{ .buffer = scratch }, .{ .rows = rows, .width = k, .x_stride = stride, .y_stride = stride, .eps = eps, .weight_f32 = weight_f32, .su_signs = matrix.suh_format == .packed_signs });
    try ex.encode(&p, e, matrix, .{ .buffer = scratch }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = k, .y_stride = n, .prerotated = true });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, x_ref.slice(u16, rows * stride), x.slice(u16, rows * stride));
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, rows * n), y.slice(u16, rows * n));
}

test "norm fused with the next rotate_in keeps every bit (bf16 and f32 weights, f16 scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 6, .mul1, 41);
    defer w.deinit();
    for ([_]u32{ 1, 3 }) |rows| for ([_]bool{ false, true }) |f32_weight| for ([_]f32{ 0.05, 2, 300 }) |scale|
        try qNorm(w.matrix, rows, f32_weight, scale);
}

test "norm fused with the next rotate_in keeps every bit (packed sign scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 4, .inst3, 42);
    defer w.deinit();
    const s = try Signed.init(device, w, 0x6c35, 0x3a17);
    defer s.deinit();
    try qNorm(s.matrix, 1, false, 2);
    try qNorm(s.matrix, 3, true, 2);
}

/// wq_b's epilogue then dsv41_rope in place, against the one-dispatch epilogue with the rope.
fn qRope(matrix: ex.Matrix, rows: u32, rotary: u32, position: u32, inverse: bool, prerotated: bool) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (p.chain.rotate_out_rope == null) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_rope, opts);
    defer lib.deinit();
    const rope = try mtl.Pipeline.init(device, lib, "dsv41_rope", false);
    defer rope.deinit();
    // The head is the matrix's 128 columns; the factors are arbitrary pairs (the rope is exact on any).
    const capacity = 64;
    const factors = try device.buffer(capacity * rotary / 2 * 8, mtl.ResourceOptions.shared);
    defer factors.deinit();
    var prng = std.Random.DefaultPrng.init(0xf00d);
    for (factors.slice(f32, capacity * rotary)) |*v| v.* = prng.random().float(f32) * 2 - 1;
    const x = try support.input(device, rows);
    defer x.deinit();
    const y_ref = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const stride = k + 16;
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try ex.encode(&p, e, matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = stride, .y_stride = n });
    e.setPipeline(rope);
    e.setBuffer(y_ref, 0, 0);
    e.setValue(extern struct { rows: u32, heads: u32, width: u32, rotary: u32, stride: u32, position: u32, step: u32, inverse: u32, spectrum: u32 }{ .rows = rows, .heads = 1, .width = n, .rotary = rotary, .stride = n, .position = position, .step = 1, .inverse = @intFromBool(inverse), .spectrum = 0 }, 1);
    e.setBuffer(factors, 0, 2);
    e.dispatchThreads(.{ .width = rotary / 2, .height = 1, .depth = rows }, .{ .width = 128 });
    e.barrier();
    // The fused form, from a plain or a prerotated input.
    var input_view: ex.View = .{ .buffer = x, .offset = 32 };
    var call: ex.Rows = .{ .count = rows, .x_stride = stride, .y_stride = n };
    if (prerotated) {
        e.setPipeline(p.rotate_in);
        e.setBuffer(x, 32, 0);
        e.setBuffer(matrix.suh.buffer, matrix.suh.offset, 1);
        e.setBuffer(scratch, 0, 2);
        var args = try ex.rowsArgs(k, n, call, input_view, .{ .buffer = y }, .{ .buffer = scratch });
        args.su_signs = @intFromBool(matrix.suh_format == .packed_signs);
        e.setValue(args, 3);
        e.setBuffer(scratch, 0, 4);
        e.setBuffer(scratch, 0, 5);
        e.setBuffer(scratch, 0, 6);
        e.dispatchGroups(.{ .width = (k / 128 + 3) / 4, .height = rows }, .{ .width = 128 });
        e.barrier();
        input_view = .{ .buffer = scratch };
        call.x_stride = k;
        call.prerotated = true;
    }
    try ex.chain.encodeRotateOutRope(&p, e, matrix, input_view, .{ .buffer = y }, .{ .buffer = scratch }, call, .{ .buffer = factors }, .{ .width = n, .rotary = rotary, .position = position, .inverse = inverse });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, rows * n), y.slice(u16, rows * n));
}

test "wq_b epilogue fused with the query rope keeps every bit (f16 scales, biases)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 6, .mul1, 51);
    defer w.deinit();
    for ([_]u32{ 1, 3 }) |rows| for ([_]u32{ 32, 64, 96 }) |rotary| for ([_]bool{ false, true }) |inverse| for ([_]bool{ false, true }) |pre|
        try qRope(w.matrix, rows, rotary, 7 + rows, inverse, pre);
}

test "wq_b epilogue fused with the query rope keeps every bit (packed sign scales)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w = try Owned.init(device, 4, .inst3, 52);
    defer w.deinit();
    const s = try Signed.init(device, w, 0x7a31, 0x25c9);
    defer s.deinit();
    try qRope(s.matrix, 1, 32, 5, false, false);
    try qRope(s.matrix, 2, 64, 9, true, true);
}

/// The output's inverse rope then wo_a (indexed over two groups) and wo_b, against the fused chain.
fn oRope(first: ex.Matrix, second: ex.Matrix, next: ex.Matrix, rotary: u32, position: u32, chained: bool) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (!ex.chain.ropeRotateInReady(&p) or !ex.chain.rotateOutInReady(&p)) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, sources.deepseek_v41_rope, opts);
    defer lib.deinit();
    const rope = try mtl.Pipeline.init(device, lib, "dsv41_rope", false);
    defer rope.deinit();
    const groups = 2;
    const capacity = 64;
    const factors = try device.buffer(capacity * rotary / 2 * 8, mtl.ResourceOptions.shared);
    defer factors.deinit();
    var prng = std.Random.DefaultPrng.init(0xbeef);
    for (factors.slice(f32, capacity * rotary)) |*v| v.* = prng.random().float(f32) * 2 - 1;
    const descriptors = try device.buffer(groups * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, groups)[0] = try ex.descriptor(first);
    descriptors.slice(ex.ExpertDescriptor, groups)[1] = try ex.descriptor(second);
    const ids = try device.buffer(groups * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, groups), &[_]u32{ 0, 1 });
    const q_ref = try device.buffer(groups * k * 2, mtl.ResourceOptions.shared);
    defer q_ref.deinit();
    const q = try device.buffer(groups * k * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    for (q_ref.slice(u16, groups * k), 0..) |*v, i| v.* = support.bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i * 7 % 29)) - 14)) / 8);
    @memcpy(q.slice(u16, groups * k), q_ref.slice(u16, groups * k));
    const grouped_ref = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped_ref.deinit();
    const grouped = try device.buffer(groups * n * 2, mtl.ResourceOptions.shared);
    defer grouped.deinit();
    const y_ref = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, groups) + try ex.scratchBytes(groups * n, n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const experts: ex.Experts = .{ .descriptors = .{ .buffer = descriptors }, .count = groups, .k = k, .n = n };
    const rows: ex.Rows = .{ .count = groups, .x_stride = k, .y_stride = n, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 }, .output_rows = groups };
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, first);
    ex.useMatrix(e, second);
    e.setPipeline(rope);
    e.setBuffer(q_ref, 0, 0);
    e.setValue(extern struct { rows: u32, heads: u32, width: u32, rotary: u32, stride: u32, position: u32, step: u32, inverse: u32, spectrum: u32 }{ .rows = 1, .heads = groups, .width = k, .rotary = rotary, .stride = groups * k, .position = position, .step = 1, .inverse = 1, .spectrum = 0 }, 1);
    e.setBuffer(factors, 0, 2);
    e.dispatchThreads(.{ .width = rotary / 2, .height = groups, .depth = 1 }, .{ .width = 128 });
    e.barrier();
    try ex.encodeIndexed(&p, e, experts, .{ .buffer = q_ref }, .{ .buffer = grouped_ref }, .{ .buffer = scratch }, rows);
    try ex.encode(&p, e, next, .{ .buffer = grouped_ref }, .{ .buffer = y_ref }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n });
    const rope_in: ex.chain.RopeIn = .{ .factors = .{ .buffer = factors }, .width = k, .rotary = rotary, .stride = groups * k, .position = position, .inverse = true };
    if (chained) {
        try ex.chain.encodeIndexed(&p, e, experts, .{ .buffer = q }, .{ .buffer = grouped }, .{ .buffer = scratch }, rows, .{ .rope_in = rope_in, .next = .{ .su = next.suh, .su_signs = next.suh_format == .packed_signs, .k = groups * n } });
        try ex.encode(&p, e, next, .{ .buffer = scratch }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n, .prerotated = true });
    } else {
        try ex.chain.encodeIndexed(&p, e, experts, .{ .buffer = q }, .{ .buffer = grouped }, .{ .buffer = scratch }, rows, .{ .rope_in = rope_in });
        try ex.encode(&p, e, next, .{ .buffer = grouped }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = groups * n, .y_stride = n });
    }
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, q_ref.slice(u16, groups * k), q.slice(u16, groups * k));
    try std.testing.expectEqualSlices(u16, grouped_ref.slice(u16, groups * n), grouped.slice(u16, groups * n));
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, n), y.slice(u16, n));
}

test "the output rope fused into wo_a's rotate_in keeps every bit (alone and chained to wo_b)" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const w0 = try Owned.init(device, 6, .mul1, 61);
    defer w0.deinit();
    const w1 = try Owned.init(device, 6, .mul1, 62);
    defer w1.deinit();
    const wb = try Owned.init(device, 6, .mul1, 63);
    defer wb.deinit();
    const s0 = try Signed.init(device, w0, 0x2b71, 0xd405);
    defer s0.deinit();
    const s1 = try Signed.init(device, w1, 0x91c3, 0x0f5e);
    defer s1.deinit();
    const sb = try Signed.init(device, wb, 0x6e22, 0xa917);
    defer sb.deinit();
    for ([_]u32{ 32, 64, 128 }) |rotary| for ([_]bool{ false, true }) |chained| {
        try oRope(w0.matrix, w1.matrix, wb.matrix, rotary, 11, chained);
        try oRope(s0.matrix, s1.matrix, sb.matrix, rotary, 3, chained);
    };
}

/// A random matrix of any k and n (the shared `Owned` is 256 by 128).
const Wide = struct {
    matrix: ex.Matrix,
    buffers: [4]mtl.Buffer,

    fn init(device: mtl.Device, halves: u16, book: ex.format.Codebook, seed: u64, kk: usize, nn: usize) !Wide {
        const words = @as(usize, halves) * 8 * (kk / 16) * (nn / 16);
        const t = try device.buffer(words * 2 + 16, mtl.ResourceOptions.shared);
        errdefer t.deinit();
        const su = try device.buffer(kk * 2 + 16, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(nn * 2 + 16, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        const bias = try device.buffer(nn * 2 + 16, mtl.ResourceOptions.shared);
        errdefer bias.deinit();
        var prng = std.Random.DefaultPrng.init(seed);
        for (t.slice(i16, words + 8)[8..]) |*x| x.* = prng.random().int(i16);
        for (su.slice(f16, kk + 8)[8..], 0..) |*x, i| x.* = if (i % 3 == 0) -0.75 else 1.125;
        for (sv.slice(f16, nn + 8)[8..], 0..) |*x, i| x.* = if (i % 5 == 0) -0.25 else 0.875;
        for (bias.slice(f16, nn + 8)[8..], 0..) |*x, i| x.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 16);
        return .{ .matrix = .{ .trellis = .{ .buffer = t, .offset = 16 }, .suh = .{ .buffer = su, .offset = 16 }, .svh = .{ .buffer = sv, .offset = 16 }, .bias = .{ .buffer = bias, .offset = 16 }, .k = kk, .n = nn, .width = .{ .halves = halves }, .codebook = book }, .buffers = .{ t, su, sv, bias } };
    }
    fn deinit(w: Wide) void {
        for (w.buffers) |b| b.deinit();
    }
};

/// The merged query/key call, then the latent norm, wq_b's rotate_in and the key glue, against the fused epilogue.
fn qkvTail(weight_f32: bool, position: u32, scale: f32) !void {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    if (!ex.chain.qkvEpilogueReady(&p)) return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const wq = try Wide.init(device, 6, .mul1, 71, k, 512);
    defer wq.deinit();
    const wkv = try Wide.init(device, 6, .mul1, 72, k, 512);
    defer wkv.deinit();
    const wb = try Wide.init(device, 6, .mul1, 73, 512, n);
    defer wb.deinit();
    var opts = mtl.CompileOptions.mlx();
    opts.math_functions = .precise;
    const ops = try mtl.Library.fromSource(device, sources.deepseek_v41_ops, opts);
    defer ops.deinit();
    const rms = try mtl.Pipeline.init(device, ops, "dsv41_rms", false);
    defer rms.deinit();
    const glue_lib = try mtl.Library.fromSource(device, sources.deepseek_v41_kv_glue, opts);
    defer glue_lib.deinit();
    const glue = try mtl.Pipeline.init(device, glue_lib, "dsv41_kv_glue", false);
    defer glue.deinit();
    const rotary: u32 = 64;
    const capacity = 4096;
    const factors = try device.buffer(capacity * rotary / 2 * 8, mtl.ResourceOptions.shared);
    defer factors.deinit();
    var prng = std.Random.DefaultPrng.init(0xcafe);
    for (factors.slice(f32, capacity * rotary)) |*v| v.* = prng.random().float(f32) * 2 - 1;
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, 2)[0] = try ex.descriptor(wq.matrix);
    descriptors.slice(ex.ExpertDescriptor, 2)[1] = try ex.descriptor(wkv.matrix);
    const ids = try device.buffer(2 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 2), &[_]u32{ 0, 1 });
    const pair = try device.buffer(2 * 4, mtl.ResourceOptions.shared);
    defer pair.deinit();
    @memset(pair.slice(u32, 2), 0);
    const x = try support.input(device, 1);
    defer x.deinit();
    const q_weight = try device.buffer(512 * 4, mtl.ResourceOptions.shared);
    defer q_weight.deinit();
    const kv_weight = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
    defer kv_weight.deinit();
    for (0..512) |i| {
        const value = 0.25 + prng.random().float(f32) * 2;
        if (weight_f32) q_weight.slice(f32, 512)[i] = value else q_weight.slice(u16, 1024)[i] = support.bf16(value);
    }
    for (kv_weight.slice(u16, 512)) |*v| v.* = support.bf16(0.25 + prng.random().float(f32) * 2);
    const y_ref = try device.buffer(2 * 512 * 2, mtl.ResourceOptions.shared);
    defer y_ref.deinit();
    const y = try device.buffer(2 * 512 * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    @memset(y_ref.slice(u16, 2 * 512), 0);
    @memset(y.slice(u16, 2 * 512), 0);
    const out_ref = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer out_ref.deinit();
    const out = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const packed_ref = try device.buffer(528, mtl.ResourceOptions.shared);
    defer packed_ref.deinit();
    const packed_out = try device.buffer(528, mtl.ResourceOptions.shared);
    defer packed_out.deinit();
    @memset(packed_ref.slice(u8, 528), 0);
    @memset(packed_out.slice(u8, 528), 0);
    const scratch = try device.buffer(try ex.scratchBytes(k, 512, 2) + try ex.scratchBytes(512, n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const experts: ex.Experts = .{ .descriptors = .{ .buffer = descriptors }, .count = 2, .k = k, .n = 512 };
    const rows: ex.Rows = .{ .count = 2, .x_stride = k + 16, .y_stride = 512, .input_ids = .{ .buffer = pair }, .input_rows = 1, .output_ids = .{ .buffer = ids }, .output_rows = 2, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 1 } };
    // Scale the call's output through the input: larger activations probe the norm's range.
    for (x.slice(u16, k + 16 + 16)[16..][0..k]) |*v| v.* = support.bf16((prng.random().float(f32) - 0.5) * scale);
    const eps: f32 = 1e-6;
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, wq.matrix);
    ex.useMatrix(e, wkv.matrix);
    // The unfused chain.
    try ex.encodeIndexed(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = y_ref }, .{ .buffer = scratch }, rows);
    e.setPipeline(rms);
    e.setBuffer(y_ref, 0, 0);
    e.setBuffer(q_weight, 0, 1);
    e.setBuffer(y_ref, 0, 2);
    e.setValue(extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 }{ .rows = 1, .width = 512, .x_stride = 512, .y_stride = 512, .eps = eps, .weight_f32 = @intFromBool(weight_f32) }, 3);
    e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
    e.barrier();
    try ex.encode(&p, e, wb.matrix, .{ .buffer = y_ref }, .{ .buffer = out_ref }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = 512, .y_stride = n });
    e.setPipeline(glue);
    e.setBuffer(y_ref, 512 * 2, 0);
    e.setBuffer(kv_weight, 0, 1);
    e.setBuffer(factors, 0, 2);
    e.setBuffer(packed_ref, 0, 3);
    e.setBuffer(packed_ref, 0, 4);
    e.setValue(extern struct { rows: u32, width: u32, stride: u32, packed_stride: u32, block: u32, position: u32, rotary: u32, write_y: u32, eps: f32 }{ .rows = 1, .width = 512, .stride = 512, .packed_stride = 528, .block = 32, .position = position, .rotary = rotary, .write_y = 0, .eps = eps }, 5);
    e.dispatchGroups(.{ .width = 16 }, .{ .width = 256 });
    e.barrier();
    // The fused chain.
    try ex.chain.encodeQkvEpilogue(&p, e, experts, .{ .buffer = x, .offset = 32 }, .{ .buffer = y }, .{ .buffer = scratch }, rows, .{
        .q_weight = .{ .buffer = q_weight },
        .q_weight_f32 = weight_f32,
        .q_su = wb.matrix.suh,
        .q_su_signs = false,
        .kv_width = 512,
        .kv_weight = .{ .buffer = kv_weight },
        .factors = .{ .buffer = factors },
        .packed_view = .{ .buffer = packed_out },
        .packed_stride = 528,
        .position = position,
        .rotary = rotary,
        .eps = eps,
    });
    try ex.encode(&p, e, wb.matrix, .{ .buffer = scratch }, .{ .buffer = out }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = 512, .y_stride = n, .prerotated = true });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, y_ref.slice(u16, 2 * 512), y.slice(u16, 2 * 512));
    try std.testing.expectEqualSlices(u16, out_ref.slice(u16, n), out.slice(u16, n));
    try std.testing.expectEqualSlices(u8, packed_ref.slice(u8, 528), packed_out.slice(u8, 528));
}

test "the merged call's epilogue carrying the latent norm, wq_b's rotate_in and the key glue keeps every bit" {
    for ([_]bool{ false, true }) |f32_weight| for ([_]f32{ 0.05, 2, 300 }) |scale| for ([_]u32{ 0, 7, 3001 }) |position|
        try qkvTail(f32_weight, position, scale);
}
