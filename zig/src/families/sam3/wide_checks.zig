const std = @import("std");
const gpu = @import("tensor_metal");
const dense_device = @import("dense_device.zig");
const dense_pair = @import("dense_pair.zig");
const normalization = @import("normalization.zig");
const gridRow = @import("spatial.zig").gridRow;
extern "c" fn erf(f64) f64;

// Single products x * w through the wide layouts: correctly rounded when normal, else within 2^-126 of the product.
pub fn checkFlushedProducts(g: *gpu.Gpu) !void {
    @setFloatMode(.strict);
    const exponents = [_]i32{ 0, -20, -36, -40, -50, -60, -70, -80, -90, -100, -110, -120, -126, -130, -140, -149 };
    const scales = [_]i32{ 0, -6, -10, -20, -30, -36, -40, -50 };
    const rows = 32;
    const cols = 64;
    const depth = 32;
    const x = try g.alloc(rows * depth, .f32);
    defer x.deinit();
    const w = try g.alloc(cols * depth, .f32);
    defer w.deinit();
    const bias = try g.alloc(cols, .f32);
    defer bias.deinit();
    @memset(x.slice(f32, x.count), 0);
    @memset(w.slice(f32, w.count), 0);
    @memset(bias.slice(f32, cols), 0);
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    for (0..rows) |r| {
        const mantissa: f32 = if (r < 16) 1 + @as(f32, @floatFromInt(random.int(u23))) * 0x1p-23 else 1;
        x.slice(f32, x.count)[r * depth] = std.math.ldexp(mantissa, exponents[r % 16]);
    }
    for (0..cols) |c| {
        const mantissa = 1 + @as(f32, @floatFromInt(random.int(u23))) * 0x1p-23;
        w.slice(f32, w.count)[c * depth] = std.math.ldexp(mantissa, scales[c / 8]);
    }
    var flushed: usize = 0;
    for ([_]dense_device.Layout{ .wide, .wide_compensated }) |layout| {
        try g.beginBatch();
        errdefer if (g.batch != null) g.endBatch() catch {};
        const out = try dense_device.run(g, x, w, bias, rows, cols, depth, layout);
        defer out.deinit();
        try g.endBatch();
        for (0..rows) |r| for (0..cols) |c| {
            const exact = @as(f64, x.get(r * depth)) * w.get(c * depth);
            const got = out.get(r * cols + c);
            const normal = @abs(exact) >= 0x1p-126;
            if (normal and got != @as(f32, @floatCast(exact))) return error.SamWideProduct;
            if (!normal and !(@abs(got - exact) < 0x1p-126)) return error.SamWideProduct;
            flushed += @intFromBool(!normal and got != @as(f32, @floatCast(exact)));
        };
    }
    std.debug.print("SAM wide products: correctly rounded down to 2^-126, {d} subnormal products flushed within 2^-126\n", .{flushed});
}

// Verifies 3-pass FP16-plane dense_pair (plain, residual, tiled, GELU chained, >32768 fallback) against FP64.
pub fn checkPairProjections(g: *gpu.Gpu) !void {
    var prng = std.Random.DefaultPrng.init(19);
    const rng = prng.random();
    const rows: usize = 65;
    const cols: usize = 129;
    const depth: usize = 64;
    const x = try g.alloc(rows * depth, .f32);
    defer x.deinit();
    const w = try g.alloc(cols * depth, .f32);
    defer w.deinit();
    const bias = try g.alloc(cols, .f32);
    defer bias.deinit();
    const res = try g.alloc(rows * cols, .f32);
    defer res.deinit();
    for (x.slice(f32, x.count)) |*v| v.* = rng.float(f32) * 4 - 2;
    for (w.slice(f32, w.count)) |*v| v.* = rng.float(f32) * 2 - 1;
    for (bias.slice(f32, cols)) |*v| v.* = rng.float(f32) - 0.5;
    for (res.slice(f32, res.count)) |*v| v.* = rng.float(f32) * 4 - 2;

    const wp = try dense_pair.prepareWeight(g, w, bias, cols, depth);
    defer wp.deinit();
    const plain = try dense_pair.run(g, x, wp, w, bias, rows, cols, depth, .none);
    defer plain.deinit();
    const with_res = try dense_pair.run(g, x, wp, w, bias, rows, cols, depth, .{ .residual = res });
    defer with_res.deinit();
    const gelu_p = try dense_pair.run(g, x, wp, w, bias, rows, cols, depth, .gelu_pair);
    defer gelu_p.deinit();
    const w2 = try g.alloc(depth * cols, .f32);
    defer w2.deinit();
    const b2 = try g.alloc(depth, .f32);
    defer b2.deinit();
    for (w2.slice(f32, w2.count)) |*v| v.* = rng.float(f32) * 2 - 1;
    for (b2.slice(f32, depth)) |*v| v.* = rng.float(f32) - 0.5;
    const wp2 = try dense_pair.prepareWeight(g, w2, b2, depth, cols);
    defer wp2.deinit();
    const chained = try dense_pair.run(g, gelu_p, wp2, w2, b2, rows, depth, cols, .none);
    defer chained.deinit();

    var worst_plain: f64 = 0;
    var worst_chain: f64 = 0;
    for (0..rows) |r| {
        var gelu_ref: [cols]f64 = undefined;
        for (0..cols) |c| {
            var sum: f64 = bias.get(c);
            var mag: f64 = @abs(sum);
            for (0..depth) |k| {
                const prod = @as(f64, x.get(r * depth + k)) * w.get(c * depth + k);
                sum += prod;
                mag += @abs(prod);
            }
            const bound = @as(f64, @floatFromInt(depth + 12)) * std.math.floatEps(f32) * mag;
            worst_plain = @max(worst_plain, @abs(plain.get(r * cols + c) - sum) / bound);
            const r_want = @as(f64, res.get(r * cols + c)) + sum;
            const r_bound = @as(f64, @floatFromInt(depth + 12)) * std.math.floatEps(f32) * (mag + @abs(res.get(r * cols + c)));
            if (@abs(with_res.get(r * cols + c) - r_want) > r_bound) return error.SamPairResidual;
            gelu_ref[c] = 0.5 * sum * (1 + erf(sum * @sqrt(@as(f64, 0.5))));
        }
        for (0..depth) |c| {
            var sum: f64 = b2.get(c);
            var mag: f64 = @abs(sum);
            for (0..cols) |k| {
                const prod = gelu_ref[k] * w2.get(c * cols + k);
                sum += prod;
                mag += @abs(prod);
            }
            const bound = @as(f64, @floatFromInt(cols + 32)) * std.math.floatEps(f32) * mag;
            worst_chain = @max(worst_chain, @abs(chained.get(r * depth + c) - sum) / bound);
        }
    }
    // Verify >32768 sentinel fallback preserves exact FP32 value.
    x.slice(f32, x.count)[0] = 32768.25;
    const fb = try dense_pair.run(g, x, wp, w, bias, rows, cols, depth, .none);
    defer fb.deinit();
    var fb_ref: f64 = bias.get(0);
    var fb_mag: f64 = @abs(fb_ref);
    for (0..depth) |k| {
        const prod = @as(f64, x.get(k)) * w.get(k);
        fb_ref += prod;
        fb_mag += @abs(prod);
    }
    if (@abs(fb.get(0) - fb_ref) > 8 * std.math.floatEps(f32) * fb_mag) return error.SamPairFallback;
    // Verify tiled norm->pair + tiled residual scatter on the 5184-row trunk grid.
    const t_rows: usize = 5184;
    const t_w: usize = 64;
    const tx = try g.alloc(t_rows * t_w, .f32);
    defer tx.deinit();
    const tw = try g.alloc(t_w * t_w, .f32);
    defer tw.deinit();
    const tb = try g.alloc(t_w, .f32);
    defer tb.deinit();
    const tres = try g.alloc(t_rows * t_w, .f32);
    defer tres.deinit();
    for (tx.slice(f32, tx.count), 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i % 257)) * 0.05);
    for (tw.slice(f32, tw.count), 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i % 251)) * 0.04);
    for (tb.slice(f32, t_w), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    for (tres.slice(f32, tres.count), 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i % 199)) * 0.03);
    const twp = try dense_pair.prepareWeight(g, tw, tb, t_w, t_w);
    defer twp.deinit();
    const t_out = try dense_pair.run(g, tx, twp, tw, tb, t_rows, t_w, t_w, .{ .tile_residual = tres });
    defer t_out.deinit();
    for ([_]usize{ 0, 23, 576, 2000, 5183 }) |tr| {
        const gr = gridRow(72, 24, tr);
        var sum: f64 = tb.get(0);
        var mag: f64 = @abs(sum);
        for (0..t_w) |k| {
            const prod = @as(f64, tx.get(tr * t_w + k)) * tw.get(k);
            sum += prod;
            mag += @abs(prod);
        }
        const want = @as(f64, tres.get(gr * t_w)) + sum;
        if (@abs(t_out.get(gr * t_w) - want) > @as(f64, @floatFromInt(t_w + 12)) * std.math.floatEps(f32) * (mag + @abs(tres.get(gr * t_w))))
            return error.SamPairTiled;
    }
    const nw = try g.alloc(t_w, .f32);
    defer nw.deinit();
    @memset(nw.slice(f32, t_w), 1);
    const np = try normalization.runPair(g, tx, nw, tb, t_w, 1e-5, true);
    defer np.deinit();
    const n_plain = try dense_pair.run(g, np, twp, tw, tb, t_rows, t_w, t_w, .none);
    defer n_plain.deinit();
    if (!std.math.isFinite(n_plain.get(0)) or worst_plain > 1 or worst_chain > 1) return error.SamPairProjection;
    std.debug.print("SAM pair projections: plain worst {d:.3} of bound, GELU-chained worst {d:.3}, tiled & >32768 fallback pass\n", .{ worst_plain, worst_chain });
}

const int8_probe_source =
    \\#include <metal_stdlib>
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\struct ProbeShape { uint m, n, k; };
    \\template<typename T, typename Acc, ushort BM, ushort BN>
    \\inline void probe_gemm(const device T* x, const device T* w, device Acc* out, constant ProbeShape& s, uint2 g) {
    \\    using Mat = tensor<device T, dextents<int, 2>, tensor_inline>;
    \\    const int row = int(g.y) * BM, col = int(g.x) * BN, depth = int(s.k);
    \\    Mat a((device T*)x, dextents<int, 2>(depth, int(s.m)), array<int, 2>({1, depth}));
    \\    Mat b((device T*)w, dextents<int, 2>(depth, int(s.n)), array<int, 2>({1, depth}));
    \\    auto left = a.slice(0, row), right = b.slice(0, col);
    \\    constexpr auto desc = matmul2d_descriptor(BM, BN, int(dynamic_extent), false, true, false);
    \\    matmul2d<desc, execution_simdgroups<4>> op;
    \\    auto dst = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, Acc>();
    \\    op.run(left, right, dst);
    \\    for (ushort e = 0; e < dst.get_capacity(); ++e) {
    \\        const auto at = dst.get_multidimensional_index(e);
    \\        const uint r = row + at[1], c = col + at[0];
    \\        if (r < s.m && c < s.n) out[ulong(r) * s.n + c] = dst[e];
    \\    }
    \\}
    \\[[kernel, max_total_threads_per_threadgroup(128)]]
    \\void probe_f32_32x64(const device float* x, const device float* w, device float* o, constant ProbeShape& s, uint2 g [[threadgroup_position_in_grid]]) {
    \\    probe_gemm<float, float, 32, 64>(x, w, o, s, g);
    \\}
    \\[[kernel, max_total_threads_per_threadgroup(128)]]
    \\void probe_f16_32x128(const device half* x, const device half* w, device float* o, constant ProbeShape& s, uint2 g [[threadgroup_position_in_grid]]) {
    \\    probe_gemm<half, float, 32, 128>(x, w, o, s, g);
    \\}
    \\[[kernel, max_total_threads_per_threadgroup(128)]]
    \\void probe_i8_32x128(const device int8_t* x, const device int8_t* w, device int32_t* o, constant ProbeShape& s, uint2 g [[threadgroup_position_in_grid]]) {
    \\    probe_gemm<int8_t, int32_t, 32, 128>(x, w, o, s, g);
    \\}
    \\[[kernel, max_total_threads_per_threadgroup(128)]]
    \\void probe_i8_64x64(const device int8_t* x, const device int8_t* w, device int32_t* o, constant ProbeShape& s, uint2 g [[threadgroup_position_in_grid]]) {
    \\    probe_gemm<int8_t, int32_t, 64, 64>(x, w, o, s, g);
    \\}
;

pub fn benchmarkInt8Nax(g: *gpu.Gpu) !void {
    const mtl = gpu.mtl;
    const lib = mtl.Library.fromSource(g.device, int8_probe_source, mtl.CompileOptions.mlx()) catch |err| {
        std.debug.print("INT8 NAX probe shader compile failed: {t}\n", .{err});
        return;
    };
    defer lib.deinit();
    const m: usize = 5184;
    const n: usize = 4736;
    const k: usize = 1024;
    const ops: f64 = 2.0 * @as(f64, m) * @as(f64, n) * @as(f64, k);
    const Case = struct { name: []const u8, bm: usize, bn: usize, elem_bytes: usize, is_i8: bool };
    const cases = [_]Case{
        .{ .name = "probe_f32_32x64", .bm = 32, .bn = 64, .elem_bytes = 4, .is_i8 = false },
        .{ .name = "probe_f16_32x128", .bm = 32, .bn = 128, .elem_bytes = 2, .is_i8 = false },
        .{ .name = "probe_i8_32x128", .bm = 32, .bn = 128, .elem_bytes = 1, .is_i8 = true },
        .{ .name = "probe_i8_64x64", .bm = 64, .bn = 64, .elem_bytes = 1, .is_i8 = true },
    };
    const shape = [3]u32{ m, n, k };
    for (cases) |case| {
        const pipe = mtl.Pipeline.init(g.device, lib, case.name, false) catch |err| {
            std.debug.print("{s}: pipeline failed ({t})\n", .{ case.name, err });
            continue;
        };
        defer pipe.deinit();
        const a = try g.device.buffer(m * k * case.elem_bytes, mtl.ResourceOptions.shared);
        defer a.deinit();
        const b = try g.device.buffer(n * k * case.elem_bytes, mtl.ResourceOptions.shared);
        defer b.deinit();
        const c = try g.device.buffer(m * n * 4, mtl.ResourceOptions.shared);
        defer c.deinit();
        if (case.is_i8) {
            const sa = a.slice(i8, m * k);
            const sb = b.slice(i8, n * k);
            for (sa, 0..) |*v, i| v.* = @intCast(@as(isize, @intCast(i % 13)) - 6);
            for (sb, 0..) |*v, i| v.* = @intCast(@as(isize, @intCast((i * 7) % 11)) - 5);
        } else if (case.elem_bytes == 2) {
            for (a.slice(f16, m * k), 0..) |*v, i| v.* = @floatFromInt(@as(isize, @intCast(i % 7)) - 3);
            for (b.slice(f16, n * k), 0..) |*v, i| v.* = @floatFromInt(@as(isize, @intCast((i * 3) % 7)) - 3);
        } else {
            for (a.slice(f32, m * k), 0..) |*v, i| v.* = @floatFromInt(@as(isize, @intCast(i % 7)) - 3);
            for (b.slice(f32, n * k), 0..) |*v, i| v.* = @floatFromInt(@as(isize, @intCast((i * 3) % 7)) - 3);
        }
        const grid = mtl.Size.of((n + case.bn - 1) / case.bn, (m + case.bm - 1) / case.bm, 1);
        const group = mtl.Size.of(128, 1, 1);
        var best: f64 = std.math.inf(f64);
        for (0..6) |iter| {
            const cb = g.queue.commandBuffer();
            const enc = cb.compute(.serial);
            enc.setPipeline(pipe);
            enc.setBuffer(a, 0, 0);
            enc.setBuffer(b, 0, 1);
            enc.setBuffer(c, 0, 2);
            enc.setValue(shape, 3);
            const reps: usize = if (iter == 0) 1 else 4;
            for (0..reps) |_| {
                enc.dispatchGroups(grid, group);
                enc.barrier();
            }
            enc.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("{s}: GPU failure {s}\n", .{ case.name, msg });
                return error.GpuFailure;
            }
            if (iter == 0 and case.is_i8) {
                const sa = a.slice(i8, m * k);
                const sb = b.slice(i8, n * k);
                const sc = c.slice(i32, m * n);
                for ([_]usize{ 0, 31, 5183 }) |row| for ([_]usize{ 0, 63, 4735 }) |col| {
                    var ref: i64 = 0;
                    for (0..k) |t| ref += @as(i64, sa[row * k + t]) * @as(i64, sb[col * k + t]);
                    if (sc[row * n + col] != @as(i32, @intCast(ref))) return error.Int8NaxMismatch;
                };
            }
            if (iter > 0) best = @min(best, cb.gpuSeconds() / @as(f64, @floatFromInt(reps)));
        }
        std.debug.print("{s} ({d}x{d}x{d}): {d:.3} ms, {d:.2} TOPS/TFLOPS\n", .{ case.name, m, n, k, best * 1e3, ops / best / 1e12 });
    }
}
