const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const math = @import("math.zig");
const h = @import("operator_csa2_fp64.zig");
const main = @import("operator_exl3_projection_fp64.zig");
const view = main.view;

test "one wide dispatch against two narrow ones (TF_ROOFLINE_PAIR=1)" {
    if (std.c.getenv("TF_ROOFLINE_PAIR") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 5120;
    const n = 2304;
    const tile = 64;
    const prompt_rows = 3619;
    const pool_matrices = 64; // cycled in both arms: same cache behaviour, only granularity differs
    var rng = h.Lcg{ .state = 0x00fa5117 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const per_expert = (k / 16) * (n / 16) * tile_bytes;
    const trellis = try b.get(u8, per_expert * pool_matrices);
    for (trellis.slice(u8, per_expert * pool_matrices)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();
    const one = try exl3.descriptor(matrix);
    const max_experts = 768;
    const desc_bytes = max_experts * @sizeOf(exl3.ExpertDescriptor);
    const desc_gpu = try b.get(u8, desc_bytes);
    for (0..max_experts) |e| {
        var d = one;
        d.trellis = one.trellis + @as(u64, e % pool_matrices) * per_expert;
        @memcpy(desc_gpu.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    const slots = max_experts * tile;
    const token_ids = try b.get(u32, slots);
    const slot_ids = try b.get(u32, slots);
    const expert_ids = try b.get(u32, slots);
    for (0..slots) |i| {
        token_ids.slice(u32, slots)[i] = @intCast(i % prompt_rows);
        slot_ids.slice(u32, slots)[i] = @intCast(i);
        expert_ids.slice(u32, slots)[i] = @intCast(i / tile);
    }
    const x_gpu = try b.get(u16, prompt_rows * k);
    for (x_gpu.slice(u16, prompt_rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, slots * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, slots));
    var pl = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer pl.deinit();
    const Run = struct {
        fn time(pip: anytype, q: mtl.Queue, desc: mtl.Buffer, xg: mtl.Buffer, yg: mtl.Buffer, sc: mtl.Buffer, ids: [3]mtl.Buffer, experts: usize, slot_rows: usize, mat: exl3.Matrix) !f64 {
            var samples: [5]f64 = undefined;
            for (0..5) |rep| {
                const cb = q.commandBuffer();
                const enc = cb.compute(.serial);
                exl3.useMatrix(enc, mat);
                try exl3.encodeIndexed(pip, enc, .{ .descriptors = .{ .buffer = desc }, .count = experts, .k = 5120, .n = 2304 }, view(h.view(xg)), view(h.view(yg)), view(h.view(sc)), .{
                    .count = slot_rows,
                    .x_stride = 5120,
                    .y_stride = 2304,
                    .input_rows = 3619,
                    .output_rows = slot_rows,
                    .tile_rows = 64,
                    .grouped = true,
                    .input_ids = view(h.view(ids[0])),
                    .output_ids = view(h.view(ids[1])),
                    .routes = .{ .ids = view(h.view(ids[2])) },
                });
                enc.end();
                cb.commit();
                cb.wait();
                if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
                samples[rep] = cb.gpuEnd() - cb.gpuStart();
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            return samples[2];
        }
    };
    const ids = [3]mtl.Buffer{ token_ids, slot_ids, expert_ids };
    const narrow = try Run.time(&pl, queue, desc_gpu, x_gpu, y_gpu, scratch, ids, 384, 384 * tile, matrix);
    const wide = try Run.time(&pl, queue, desc_gpu, x_gpu, y_gpu, scratch, ids, 768, 768 * tile, matrix);
    std.debug.print("PAIR {{\"narrow_ms\":{d:.2},\"wide_ms\":{d:.2},\"two_narrow_ms\":{d:.2},\"fusion_gain_pct\":{d:.1}}}\n", .{ narrow * 1e3, wide * 1e3, 2.0 * narrow * 1e3, 100.0 * (2.0 * narrow - wide) / (2.0 * narrow) });
}

test "routed trellis roofline at the engine prefill shape (TF_ROOFLINE=1)" {
    if (std.c.getenv("TF_ROOFLINE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();

    const k = 5120;
    const n = 2304;
    const experts = 384;
    const tile = 64;
    const slots = experts * tile;
    const prompt_rows = 3619;
    var rng = h.Lcg{ .state = 0x00f11e55 };
    // The pack's real width matters: at 1 bit a weight the pass is compute-bound, at 2.9 it streams.
    const roof_halves: u8 = blk: {
        const raw = std.c.getenv("TF_ROOFLINE_HALVES") orelse break :blk 6;
        break :blk std.fmt.parseInt(u8, std.mem.span(raw), 10) catch 6;
    };
    const width = exl3.format.Width{ .halves = roof_halves };
    const tile_bytes = width.tileWords() * 2;
    const per_expert = (k / 16) * (n / 16) * tile_bytes;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, per_expert * experts);
    for (trellis.slice(u8, per_expert * experts)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02)));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = .inst3,
    };
    try matrix.validate();
    // 384 descriptors, one per distinct matrix: the engine's table shape.
    const desc_bytes = experts * @sizeOf(exl3.ExpertDescriptor);
    const desc_gpu = try b.get(u8, desc_bytes);
    const one = try exl3.descriptor(matrix);
    for (0..experts) |e| {
        var d = one;
        d.trellis = one.trellis + @as(u64, e) * per_expert; // each expert's own matrix
        @memcpy(desc_gpu.slice(u8, desc_bytes)[e * @sizeOf(exl3.ExpertDescriptor) ..][0..@sizeOf(exl3.ExpertDescriptor)], std.mem.asBytes(&d));
    }
    // The engine's grouped rows: each expert's 64-row tile, token ids cycling over the prompt.
    const token_ids = try b.get(u32, slots);
    const slot_ids = try b.get(u32, slots);
    const expert_ids = try b.get(u32, slots);
    for (0..slots) |i| {
        token_ids.slice(u32, slots)[i] = @intCast(i % prompt_rows);
        slot_ids.slice(u32, slots)[i] = @intCast(i);
        expert_ids.slice(u32, slots)[i] = @intCast(i / tile);
    }
    const x_gpu = try b.get(u16, prompt_rows * k);
    for (x_gpu.slice(u16, prompt_rows * k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, slots * n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, slots));
    var p = try exl3.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();

    var best: f64 = 1e9;
    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        exl3.useMatrix(enc, matrix);
        try exl3.encodeIndexed(&p, enc, .{
            .descriptors = .{ .buffer = desc_gpu },
            .count = experts,
            .k = k,
            .n = n,
        }, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{
            .count = slots,
            .x_stride = k,
            .y_stride = n,
            .input_rows = prompt_rows,
            .output_rows = slots,
            .tile_rows = tile,
            .grouped = true,
            .input_ids = view(h.view(token_ids)),
            .output_ids = view(h.view(slot_ids)),
            .routes = .{ .ids = view(h.view(expert_ids)) },
        });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Exl3ProjectionGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
        best = @min(best, samples[rep]);
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const seconds = samples[2];
    const macs: f64 = @floatFromInt(@as(u64, slots) * @as(u64, k) * @as(u64, n));
    const weight_bytes: f64 = @floatFromInt(per_expert * experts); // the bytes the pass actually streams
    std.debug.print("ROOFLINE {{\"k\":{d},\"n\":{d},\"experts\":{d},\"slot_rows\":{d},\"gpu_ms\":{d:.2},\"tmac_s\":{d:.3},\"tflops\":{d:.1},\"gbs\":{d:.1}}}\n", .{ k, n, experts, slots, seconds * 1e3, macs / seconds / 1e12, 2.0 * macs / seconds / 1e12, weight_bytes / seconds / 1e9 });
}

test "int8 weight quantization at the routed shape, against the float64 ideal" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const k = 5120;
    const n = 2304;
    var rng = h.Lcg{ .state = 0x00f8a155 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (rng.scale(1.0) + 1.0))));
    const x_gpu = try b.get(u16, k);
    const x = x_gpu.slice(u16, k);
    for (x) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const matrix = exl3.Matrix{
        .trellis = view(h.view(trellis)),
        .suh = view(h.view(suh)),
        .svh = view(h.view(svh)),
        .k = k,
        .n = n,
        .width = width,
        .codebook = .inst3,
    };
    try matrix.validate();
    var s1 = try oracle.Scratch.init(std.testing.allocator, k, n);
    defer s1.deinit();
    const ideal = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(ideal);
    const quantized = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(quantized);
    try oracle.applyF32Trellis(matrix, x, ideal, &s1);
    try oracle.applyF32TrellisInt8(matrix, x, quantized, &s1);
    var energy: f64 = 0;
    var magnitude: f64 = 0;
    var max_abs: f64 = 0;
    var over_one_percent: usize = 0;
    var worst_column: usize = 0;
    for (0..n) |i| {
        const d = @as(f64, quantized[i]) - @as(f64, ideal[i]);
        energy += d * d;
        magnitude += @as(f64, ideal[i]) * @as(f64, ideal[i]);
        if (@abs(d) > max_abs) max_abs = @abs(d);
        const rel = @abs(d) / @max(@abs(@as(f64, ideal[i])), 0x1p-126);
        if (rel > 0.01) over_one_percent += 1;
        if (rel > max_abs) worst_column = i;
    }
    const nrms = if (magnitude > 0) @sqrt(energy / magnitude) else 0;

    std.debug.print("INT8 {{\"k\":{d},\"n\":{d},\"halves\":{d},\"nrms_vs_fp64\":{e},\"max_abs\":{e},\"columns_over_1pct\":{d},\"worst\":{d},}}\n", .{ k, n, width.halves, nrms, max_abs, over_one_percent, worst_column });
    // The gate: an int8 path is only promotable if it stays at rounding level against the exact one.
    try std.testing.expect(nrms < 0.02);
}

// Item 15's next step: the int8 mma's fragment layout, checked against a CPU dot for every cell.
const int8_value_body =
    \\using namespace tfp;
    \\kernel void int8_mma_values(device const int8_t* a [[buffer(0)]], device const int8_t* b [[buffer(1)]],
    \\                          device int* out [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<int8_t> left, r0, r1;
    \\  const char4 a0 = *(const device char4*)(a + home.x), a1 = *(const device char4*)(a + 8 * 16 + home.x);
    \\  left = frag<int8_t>(a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w);
    \\  const char4 b0 = *(const device char4*)(b + home.x), b1 = *(const device char4*)(b + 8 * 16 + home.x);
    \\  r0 = frag<int8_t>(b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w);
    \\  const char4 c0 = *(const device char4*)(b + 16 * 16 + home.x), c1 = *(const device char4*)(b + 24 * 16 + home.x);
    \\  r1 = frag<int8_t>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
    \\  frag<int32_t> lo = frag<int32_t>(0), hi = frag<int32_t>(0);
    \\  mma_16x32<false, true>(lo, hi, left, r0, r1);
    \\  // store through the helper's own layout, as the tree's kernels do
    \\  frag<float> f0, f1;
    \\  for (short e = 0; e < 8; e++) {
    \\    f0[e] = float(lo[e]);
    \\    f1[e] = float(hi[e]);
    \\  }
    \\  frag_put(f0, out, 32, 0, 0, home);
    \\  frag_put(f1, out, 32, 0, 16, home);
    \\}
;

test "int8 mma layout probe against a CPU int8 dot (TF_INT8_VALUES=1)" {
    if (std.c.getenv("TF_INT8_VALUES") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const nax = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/metal/nax.h", gpa, .limited(1 << 20));
    defer gpa.free(nax);
    const text = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax, int8_value_body });
    defer gpa.free(text);
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, text, options);
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "int8_mma_values", false);
    defer pipeline.deinit();
    var a: [16 * 16]i8 = undefined;
    var bb: [32 * 16]i8 = undefined;
    var rng = h.Lcg{ .state = 0x00b1a5e5 };
    for (&a) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 9)) - 4);
    for (&bb) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 9)) - 4);
    const a_gpu = try device.buffer(a.len, mtl.ResourceOptions.shared);
    defer a_gpu.deinit();
    const b_gpu = try device.buffer(bb.len, mtl.ResourceOptions.shared);
    defer b_gpu.deinit();
    const out_gpu = try device.buffer(16 * 32 * @sizeOf(i32), mtl.ResourceOptions.shared);
    defer out_gpu.deinit();
    @memcpy(a_gpu.slice(i8, a.len), &a);
    @memcpy(b_gpu.slice(i8, bb.len), &bb);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(pipeline);
    enc.setBuffer(a_gpu, 0, 0);
    enc.setBuffer(b_gpu, 0, 1);
    enc.setBuffer(out_gpu, 0, 2);
    enc.dispatchGroups(.{ .width = 1, .height = 1 }, .{ .width = 32 });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Int8ValuesGpuFailure;
    const got = out_gpu.slice(i32, 16 * 32);
    var mismatches: usize = 0;
    var checked: usize = 0;
    for (0..16) |r| {
        for (0..32) |c| {
            var want: i32 = 0;
            for (0..16) |d| want += @as(i32, a[r * 16 + d]) * @as(i32, bb[c * 16 + d]);
            checked += 1;
            if (got[r * 32 + c] != want) mismatches += 1;
        }
    }
    std.debug.print("INT8VALUES {{\"checked\":{d},\"mismatches\":{d}}}\n", .{ checked, mismatches });
}

// Item 15's speed half: the same tile and k-loop in int8 and in bf16.
const int8_rate_body =
    \\using namespace tfp;
    \\kernel void int8_rate(device const int8_t* x [[buffer(0)]], device const int8_t* w [[buffer(1)]],
    \\                     device float* out [[buffer(2)]], constant uint& k [[buffer(3)]],
    \\                     uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    \\  const short2 home = frag_home(ushort(lane));
    \\  frag<int32_t> lo = frag<int32_t>(0), hi = frag<int32_t>(0);
    \\  for (uint kb = 0; kb < k; kb += 16) {
    \\    frag<int8_t> a, b0, b1;
    \\    const char4 v0 = *(const device char4*)(x + (long)(tg.y * 16) * k + kb + home.x);
    \\    const char4 v1 = *(const device char4*)(x + (long)(tg.y * 16 + 8) * k + kb + home.x);
    \\    a = frag<int8_t>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
    \\    const char4 c0 = *(const device char4*)(w + (long)(tg.x * 32) * k + kb + home.x);
    \\    const char4 c1 = *(const device char4*)(w + (long)(tg.x * 32 + 8) * k + kb + home.x);
    \\    b0 = frag<int8_t>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
    \\    const char4 d0 = *(const device char4*)(w + (long)(tg.x * 32 + 16) * k + kb + home.x);
    \\    const char4 d1 = *(const device char4*)(w + (long)(tg.x * 32 + 24) * k + kb + home.x);
    \\    b1 = frag<int8_t>(d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w);
    \\    mma_16x32<false, true>(lo, hi, a, b0, b1);
    \\  }
    \\  for (short e = 0; e < 8; e++) {
    \\    const uint r = tg.y * 16 + home.y + (e >> 2) * 8, c = tg.x * 32 + (e & 3) + 8 * (e >> 2);
    \\    if (r == 0) out[c] = float(lo[e]) + float(hi[e]);
    \\  }
    \\}
;

test "int8 mma rate against bf16 at the routed tile shape (TF_INT8_RATE=1)" {
    if (std.c.getenv("TF_INT8_RATE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.testing.allocator;
    const nax = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/kernels/metal/nax.h", gpa, .limited(1 << 20));
    defer gpa.free(nax);
    const text = try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax, int8_rate_body });
    defer gpa.free(text);
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const lib = try mtl.Library.fromSource(device, text, options);
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "int8_rate", false);
    defer pipeline.deinit();
    const k: u32 = 5120;
    const rows: usize = 24576; // 384 experts x 64 rows
    const n: usize = 2304;
    const x_gpu = try device.buffer(rows * k, mtl.ResourceOptions.shared);
    defer x_gpu.deinit();
    const w_gpu = try device.buffer(n * k, mtl.ResourceOptions.shared);
    defer w_gpu.deinit();
    const out_gpu = try device.buffer(n * @sizeOf(f32), mtl.ResourceOptions.shared);
    defer out_gpu.deinit();
    var rng = h.Lcg{ .state = 0x00e8a7e5 };
    for (x_gpu.slice(i8, rows * k)) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 255)) - 127);
    for (w_gpu.slice(i8, n * k)) |*v| v.* = @intCast(@as(i32, @intCast(rng.next() % 255)) - 127);
    var samples: [5]f64 = undefined;
    for (0..5) |rep| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipeline);
        enc.setBuffer(x_gpu, 0, 0);
        enc.setBuffer(w_gpu, 0, 1);
        enc.setBuffer(out_gpu, 0, 2);
        enc.setValue(k, 3);
        enc.dispatchGroups(.{ .width = n / 32, .height = rows / 16 }, .{ .width = 32 });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.Int8RateGpuFailure;
        samples[rep] = cb.gpuEnd() - cb.gpuStart();
    }
    std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
    const seconds = samples[2];
    const macs = @as(f64, @floatFromInt(rows * n * @as(usize, k)));
    std.debug.print("INT8RATE {{\"rows\":{d},\"n\":{d},\"k\":{d},\"gpu_ms\":{d:.2},\"tmac_s\":{d:.3},\"tflops\":{d:.1}}}\\n", .{ rows, n, k, seconds * 1e3, macs / seconds / 1e12, 2.0 * macs / seconds / 1e12 });
}

// TF_FUSE_PROBE=1.
