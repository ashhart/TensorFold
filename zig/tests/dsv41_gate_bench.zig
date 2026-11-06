//! The router gate's numerics, model-free: `tf-dense-col` (the shipped fp32 column path) against the tensor-unit
//! gate GEMM's f16 and bf16 operands (fp32 destination), each against an fp64 reference of its own operands.
//! The band receipt TF_DSV41_ROUTE_GEMM needs: NRMS, max absolute error, and the 384-way argmax of every path.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// The prompt's row count; TF_GATE_BENCH_ROWS narrows it so the real operands (a capture's rows) fit.
var rows: u32 = 3619;
const k: u32 = 5120;
const n: u32 = 384;

const GemmF16 = "custom_kernel_tf_gemm_nax_f16_f32out_n_t_n_t_n_64_128_256_2_4_float16_t_float16_t_int32_t_float";
const GemmBf16 = "custom_kernel_tf_gemm_nax_bf16_f32out_n_t_n_t_n_64_128_256_2_4_bfloat16_t_bfloat16_t_int32_t_float";

const ColDense = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };

fn bf(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
fn f16v(x: u16) f32 {
    return @floatCast(@as(f16, @bitCast(x)));
}
fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
fn toF16(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

fn dotF64(x: []const u16, w: []const u16, f16x: bool, f16w: bool) f64 {
    var s: f64 = 0;
    for (x, w) |a, b| {
        const xa = if (f16x) f16v(a) else bf(a);
        const wb = if (f16w) f16v(b) else bf(b);
        s += @as(f64, xa) * @as(f64, wb);
    }
    return s;
}

fn runGemm(queue: mtl.Queue, pipe: mtl.Pipeline, a: mtl.Buffer, b: mtl.Buffer, d: mtl.Buffer) !void {
    const tn = (n + 127) / 128;
    const tm = (rows + 63) / 64;
    var p: [16]i32 = @splat(0);
    const vals = [_]u32{ rows, n, k, k, k, n, tn, tm, 2, k / 256, 0, 0, 0, 0, 0 };
    for (vals, 0..) |v, i| p[i] = @intCast(v);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(a, 0, 0);
    e.setBuffer(b, 0, 1);
    e.setBytes(std.mem.asBytes(&p), 2);
    e.setBuffer(d, 0, 3);
    e.dispatchThreads(.{ .width = @as(usize, tn << 2) * 32, .height = (tm + 3) / 4 * 4, .depth = 2 }, .{ .width = 32, .height = 4, .depth = 2 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("gemm failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
}

fn runCol(queue: mtl.Queue, pipe: mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, bf16_w: bool) !void {
    const d = ColDense{ .rows = rows, .k = k, .n = n, .x_stride = k, .bf16_w = @intFromBool(bf16_w) };
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(w, 0, 1);
    e.setBuffer(y, 0, 2);
    e.setBytes(std.mem.asBytes(&d), 3);
    e.dispatchGroups(.{ .width = n, .height = rows, .depth = 1 }, .{ .width = 512 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("dense_col failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
}

var xbuf: []u16 = undefined;
var wbuf: []u16 = undefined;
// the f16 twins / bf16 twin of the same values
var xf16: []u16 = undefined;
var wf16: []u16 = undefined;
var wbf16: []u16 = undefined;
var ycol: []f32 = undefined;
var yg16: []f32 = undefined;
var ygb: []f32 = undefined;

pub fn main(init: std.process.Init) !void {
    std.debug.print("start\n", .{});
    const gpa = init.gpa;
    if (std.c.getenv("TF_GATE_BENCH_ROWS")) |v| rows = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3619;
    xbuf = try gpa.alloc(u16, rows * k);
    wbuf = try gpa.alloc(u16, n * k);
    xf16 = try gpa.alloc(u16, rows * k);
    wf16 = try gpa.alloc(u16, n * k);
    wbf16 = try gpa.alloc(u16, n * k);
    ycol = try gpa.alloc(f32, rows * n);
    yg16 = try gpa.alloc(f32, rows * n);
    ygb = try gpa.alloc(f32, rows * n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    std.debug.print("a\n", .{});
    const device = try mtl.Device.init();
    std.debug.print("b\n", .{});
    const queue = try device.queue();
    std.debug.print("c\n", .{});

    if (std.c.getenv("TF_GATE_BENCH_X")) |xpath| {
        const blob = try std.Io.Dir.cwd().readFileAlloc(init.io, std.mem.span(xpath), gpa, .limited(1 << 30));
        const vals = std.mem.bytesAsSlice(f32, blob[0 .. blob.len - blob.len % 4]);
        for (vals, 0..) |v, i| {
            if (i >= xbuf.len) break;
            xbuf[i] = toBf16(v);
        }
        std.debug.print("real x: {d} f32 values from {s}\n", .{ vals.len, xpath });
    }
    if (std.c.getenv("TF_GATE_BENCH_W")) |wpath| {
        const blob = try std.Io.Dir.cwd().readFileAlloc(init.io, std.mem.span(wpath), gpa, .limited(1 << 30));
        const vals = std.mem.bytesAsSlice(f32, blob[0 .. blob.len - blob.len % 4]);
        for (vals, 0..) |v, i| {
            if (i >= wbuf.len) break;
            wbuf[i] = toF16(v);
        }
        std.debug.print("real w: {d} f32 values from {s}\n", .{ vals.len, wpath });
    }
    var rng: u64 = 0x243F6A8885A308D3;
    if (std.c.getenv("TF_GATE_BENCH_X") == null) {
        for (xbuf) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            const f = @as(f32, @floatFromInt(@as(i32, @intCast((rng >> 40) & 0xffff)) - 32768)) / 32768.0;
            const g = f + 0.000137 * @as(f32, @floatFromInt(@as(u32, @intCast((rng >> 20) & 0xff))));
            v.* = toBf16(g);
        }
    }
    if (std.c.getenv("TF_GATE_BENCH_W") == null) {
        for (wbuf) |*v| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            const f = @as(f32, @floatFromInt(@as(i32, @intCast((rng >> 40) & 0xffff)) - 32768)) / 65536.0;
            const g = f + 0.0000137 * @as(f32, @floatFromInt(@as(u32, @intCast((rng >> 20) & 0xff))));
            v.* = toF16(g);
        }
    }
    for (xbuf, 0..) |v, i| xf16[i] = toF16(bf(v));
    for (wbuf, 0..) |v, i| {
        wf16[i] = v;
        wbf16[i] = toBf16(f16v(v));
    }

    // Two libraries: gemm_nax (nax.h inlined) and the family's mhc kernels (self-contained).
    var gemm_src: ?[]const u8 = null;
    inline for (ks.prefill) |f| {
        if (comptime std.mem.eql(u8, f.name, "gemm_nax")) gemm_src = f.text;
    }
    const nax_text = try std.mem.replaceOwned(u8, gpa, gemm_src orelse return error.NoGemmSource, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_text);
    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    const gemm_lib = try mtl.Library.fromSource(device, nax_text, options);
    std.debug.print("d\n", .{});
    const col_lib = try mtl.Library.fromSource(device, ks.deepseek_v41_mhc, options);
    std.debug.print("e\n", .{});
    const pipe_f16 = try mtl.Pipeline.init(device, gemm_lib, GemmF16, false);
    std.debug.print("f\n", .{});
    const pipe_bf16 = try mtl.Pipeline.init(device, gemm_lib, GemmBf16, false);
    const pipe_col = try mtl.Pipeline.init(device, col_lib, "dsv41_dense_col", false);
    std.debug.print("g\n", .{});

    const x = try device.buffer(@as(usize, rows) * k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const x16 = try device.buffer(@as(usize, rows) * k * 2, mtl.ResourceOptions.shared);
    defer x16.deinit();
    const w16 = try device.buffer(@as(usize, n) * k * 2, mtl.ResourceOptions.shared);
    defer w16.deinit();
    const wb = try device.buffer(@as(usize, n) * k * 2, mtl.ResourceOptions.shared);
    defer wb.deinit();
    const y1 = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
    defer y1.deinit();
    const y2 = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
    defer y2.deinit();
    const y3 = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
    defer y3.deinit();

    @memcpy(x.slice(u16, rows * k), xbuf);
    @memcpy(x16.slice(u16, rows * k), xf16);
    @memcpy(w16.slice(u16, n * k), wf16);
    @memcpy(wb.slice(u16, n * k), wbf16);

    std.debug.print("buffers up, running dense_col\n", .{});
    try runCol(queue, pipe_col, x, w16, y1, false); // shipped: bf16 x, f16 w
    std.debug.print("dense_col done\n", .{});
    try runGemm(queue, pipe_f16, x16, w16, y2); // f16 operands (the activation converted)
    try runGemm(queue, pipe_bf16, x, wb, y3); // bf16 operands (the gate rounded)
    const p1: [*]const f32 = @ptrCast(@alignCast(y1.contents()));
    const p2: [*]const f32 = @ptrCast(@alignCast(y2.contents()));
    const p3: [*]const f32 = @ptrCast(@alignCast(y3.contents()));
    @memcpy(ycol, p1[0 .. rows * n]);
    @memcpy(yg16, p2[0 .. rows * n]);
    @memcpy(ygb, p3[0 .. rows * n]);

    {
        var nan: usize = 0;
        var zero: usize = 0;
        var mn: f32 = 1e30;
        var mx: f32 = -1e30;
        for (yg16) |v| {
            if (std.math.isNan(v)) {
                nan += 1;
            } else if (v == 0) {
                zero += 1;
            } else {
                mn = @min(mn, v);
                mx = @max(mx, v);
            }
        }
        std.debug.print("f16 out: {d} nan, {d} zero of {d}; finite range {e:.3}..{e:.3}", .{ nan, zero, yg16.len, mn, mx });
        std.debug.print("{c}", .{@as(u8, 10)});
        std.debug.print("  first 4: {e:.4} {e:.4} {e:.4} {e:.4}", .{ yg16[0], yg16[1], yg16[2], yg16[3] });
        std.debug.print("{c}", .{@as(u8, 10)});
        std.debug.print("  x16[0..3] {x} {x} {x}  w16[0..3] {x} {x} {x}", .{ xf16[0], xf16[1], xf16[2], wf16[0], wf16[1], wf16[2] });
        std.debug.print("{c}", .{@as(u8, 10)});
    }
    const probe = [_]u32{ 0, 1, rows / 3, rows / 2, (rows * 3) / 4, rows - 1 };
    var ss_col: f64 = 0;
    var ss_g16: f64 = 0;
    var ss_gb: f64 = 0;
    var ss_ref: f64 = 0;
    var max_col: f64 = 0;
    var max_g16: f64 = 0;
    var max_gb: f64 = 0;
    var max_cross: f64 = 0;
    var agree_col: u32 = 0;
    var agree_g16: u32 = 0;
    var agree_gb: u32 = 0;
    var total: u32 = 0;
    for (probe) |r| {
        const xr = xbuf[@as(usize, r) * k ..][0..k];
        const xr16 = xf16[@as(usize, r) * k ..][0..k];
        var best_ref: f32 = -1e30;
        var best_col: f32 = -1e30;
        var best_g16: f32 = -1e30;
        var best_gb: f32 = -1e30;
        var id_ref: u32 = 0;
        var id_col: u32 = 0;
        var id_g16: u32 = 0;
        var id_gb: u32 = 0;
        var rmse_col: f64 = 0;
        var rmse_g16: f64 = 0;
        var rmse_gb: f64 = 0;
        var norm: f64 = 0;
        for (0..n) |j| {
            const wr = wbuf[j * k ..][0..k];
            const ref = dotF64(xr, wr, false, true); // the shipped path's operands: bf16 x, f16 w
            const ref16 = dotF64(xr16, wf16[j * k ..][0..k], true, true);
            const refb = dotF64(xr, wbf16[j * k ..][0..k], false, false);
            const c = ycol[@as(usize, r) * n + j];
            const g16 = yg16[@as(usize, r) * n + j];
            const gb = ygb[@as(usize, r) * n + j];
            rmse_col += (c - ref) * (c - ref);
            rmse_g16 += (g16 - ref16) * (g16 - ref16);
            rmse_gb += (gb - refb) * (gb - refb);
            norm += ref * ref;
            max_col = @max(max_col, @abs(c - ref));
            max_g16 = @max(max_g16, @abs(g16 - ref16));
            max_gb = @max(max_gb, @abs(gb - refb));
            max_cross = @max(max_cross, @abs(g16 - c));
            if (c > best_col) {
                best_col = c;
                id_col = @intCast(j);
            }
            if (g16 > best_g16) {
                best_g16 = g16;
                id_g16 = @intCast(j);
            }
            if (gb > best_gb) {
                best_gb = gb;
                id_gb = @intCast(j);
            }
            if (ref > best_ref) {
                best_ref = @floatCast(ref);
                id_ref = @intCast(j);
            }
        }
        ss_col += rmse_col;
        ss_g16 += rmse_g16;
        ss_gb += rmse_gb;
        ss_ref += norm;
        total += 1;
        if (id_col == id_ref) agree_col += 1;
        if (id_g16 == id_ref) agree_g16 += 1;
        if (id_gb == id_ref) agree_gb += 1;
        std.debug.print("row {d:>5}: argmax fp64 {d:>3} col {d:>3} f16 {d:>3} bf16 {d:>3}  nrmse col {e:.3} f16 {e:.3} bf16 {e:.3}\n", .{
            r, id_ref, id_col, id_g16, id_gb,
            @sqrt(rmse_col / norm), @sqrt(rmse_g16 / norm), @sqrt(rmse_gb / norm),
        });
    }
    std.debug.print("\nrows probed {d}, {d} cols each:\n", .{ total, n });
    std.debug.print("  nrmse vs own fp64:  dense_col {e:.3}  gemm-f16 {e:.3}  gemm-bf16 {e:.3}\n", .{
        @sqrt(ss_col / ss_ref), @sqrt(ss_g16 / ss_ref), @sqrt(ss_gb / ss_ref),
    });
    std.debug.print("  max abs vs fp64:    dense_col {e:.3}  gemm-f16 {e:.3}  gemm-bf16 {e:.3}\n", .{ max_col, max_g16, max_gb });
    std.debug.print("  max abs gemm-f16 - dense_col: {e:.3}\n", .{max_cross});
    std.debug.print("  argmax agreement:   dense_col {d}/{d}  gemm-f16 {d}/{d}  gemm-bf16 {d}/{d}\n", .{ agree_col, total, agree_g16, total, agree_gb, total });

    try gateTreeSection(device, queue, x, w16);
}

const GateTree = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };

fn timeCol(queue: mtl.Queue, pipe: mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, trials: usize) !f64 {
    var samples: [8]f64 = undefined;
    const nt = @min(trials, samples.len);
    const d = ColDense{ .rows = rows, .k = k, .n = n, .x_stride = k, .bf16_w = 0 };
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(y, 0, 2);
        e.setBytes(std.mem.asBytes(&d), 3);
        e.dispatchGroups(.{ .width = n, .height = rows, .depth = 1 }, .{ .width = 512 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("col timing failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    std.sort.heap(f64, samples[0..nt], {}, std.sort.asc(f64));
    return samples[nt / 2];
}

fn timeTree(queue: mtl.Queue, pipe: mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, d: GateTree, trials: usize) !f64 {
    var samples: [8]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(y, 0, 2);
        e.setBytes(std.mem.asBytes(&d), 3);
        e.dispatchGroups(.{ .width = (n + 3) / 4, .height = (rows + 3) / 4, .depth = 1 }, .{ .width = 128 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("tree timing failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    std.sort.heap(f64, samples[0..nt], {}, std.sort.asc(f64));
    return samples[nt / 2];
}

fn gateTreeSection(device: mtl.Device, queue: mtl.Queue, x: mtl.Buffer, w: mtl.Buffer) !void {
    const lib = try mtl.Library.fromSource(device, ks.deepseek_v41_mhc, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const col = try mtl.Pipeline.init(device, lib, "dsv41_dense_col", false);
    defer col.deinit();
    const tree = try mtl.Pipeline.init(device, lib, "dsv41_gate_tree", false);
    defer tree.deinit();
    const yc = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
    defer yc.deinit();
    const yt = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
    defer yt.deinit();
    const d = GateTree{ .rows = rows, .k = k, .n = n, .x_stride = k, .bf16_w = 0 };
    try runCol(queue, col, x, w, yc, false);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(tree);
    e.setBuffer(x, 0, 0);
    e.setBuffer(w, 0, 1);
    e.setBuffer(yt, 0, 2);
    e.setBytes(std.mem.asBytes(&d), 3);
    e.dispatchGroups(.{ .width = (n + 3) / 4, .height = (rows + 3) / 4, .depth = 1 }, .{ .width = 128 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("gate_tree dispatch failed: {s}\n", .{msg});
        return error.DispatchFailed;
    }
    const pc: [*]const f32 = @ptrCast(@alignCast(yc.contents()));
    const pt: [*]const f32 = @ptrCast(@alignCast(yt.contents()));
    var diff: usize = 0;
    var worst: f32 = 0;
    for (0..rows * n) |i| {
        if (@as(u32, @bitCast(pc[i])) != @as(u32, @bitCast(pt[i]))) {
            diff += 1;
            worst = @max(worst, @abs(pc[i] - pt[i]));
        }
    }
    std.debug.print("\nbit-exact gate_tree vs dense_col (engine options, bf16 x / f16 w): {d} of {d} cells differ, worst {e:.3}\n", .{ diff, rows * n, worst });
    const tc = try timeCol(queue, col, x, w, yc, 5);
    const tt = try timeTree(queue, tree, x, w, yt, d, 5);
    std.debug.print("gate projection {d}x{d}x{d}: dense_col {d:.3} ms, gate_tree {d:.3} ms ({d:.2}x)\n", .{ rows, k, n, tc, tt, tc / tt });

    const tree4 = mtl.Pipeline.init(device, lib, "dsv41_gate_tree4", false) catch null;
    if (tree4) |t4| {
        defer t4.deinit();
        const d4 = GateTree4{ .rows = rows, .k = k, .n = n, .x_stride = k, .bf16_w = 0 };
        const y4 = try device.buffer(@as(usize, rows) * n * 4, mtl.ResourceOptions.shared);
        defer y4.deinit();
        const cb4 = queue.commandBuffer();
        const e4 = cb4.compute(.serial);
        e4.setPipeline(t4);
        e4.setBuffer(x, 0, 0);
        e4.setBuffer(w, 0, 1);
        e4.setBuffer(y4, 0, 2);
        e4.setBytes(std.mem.asBytes(&d4), 3);
        e4.dispatchGroups(.{ .width = (n + 1) / 2, .height = (rows + 1) / 2, .depth = 1 }, .{ .width = 32 });
        e4.end();
        cb4.commit();
        cb4.wait();
        if (cb4.failure()) |msg| {
            std.debug.print("gate_tree4 dispatch failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        const p4: [*]const f32 = @ptrCast(@alignCast(y4.contents()));
        var diff4: usize = 0;
        var worst4: f32 = 0;
        for (0..rows * n) |i| {
            if (@as(u32, @bitCast(pc[i])) != @as(u32, @bitCast(p4[i]))) {
                diff4 += 1;
                worst4 = @max(worst4, @abs(pc[i] - p4[i]));
            }
        }
        std.debug.print("bit-exact gate_tree4 vs dense_col: {d} of {d} cells differ, worst {e:.3}\n", .{ diff4, rows * n, worst4 });
        const t4ms = try timeTree4(queue, t4, x, w, y4, d4, 5);
        std.debug.print("gate_tree4 {d:.3} ms ({d:.2}x dense_col)\n", .{ t4ms, tc / t4ms });
    }
}

const GateTree4 = extern struct { rows: u32, k: u32, n: u32, x_stride: u32, bf16_w: u32 };

fn timeTree4(queue: mtl.Queue, pipe: mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, y: mtl.Buffer, d: GateTree4, trials: usize) !f64 {
    var samples: [8]f64 = undefined;
    const nt = @min(trials, samples.len);
    for (0..nt) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(y, 0, 2);
        e.setBytes(std.mem.asBytes(&d), 3);
        e.dispatchGroups(.{ .width = (n + 1) / 2, .height = (rows + 1) / 2, .depth = 1 }, .{ .width = 32 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("tree4 timing failed: {s}\n", .{msg});
            return error.DispatchFailed;
        }
        samples[t] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    std.sort.heap(f64, samples[0..nt], {}, std.sort.asc(f64));
    return samples[nt / 2];
}
