//! Prompt matmul kernels compiled from .metal files at run time: chained speed at one shape, bits against the first, error against fp64.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

const usage =
    \\tf-mm-bench M K N FILE[:BM:BN:THREADS[:ENTRY]] ...
    \\Each FILE takes tf_affine_mm's buffers (x, w, scales, biases, row sums, args, y, strides) and is compiled with
    \\TF_BITS 4, TF_GROUP 64 and nax.h inlined; each file's tf_affine_row_sums fills its row sums (timed too).
    \\MM_REPS (dispatches a timed buffer, 16), MM_TRIALS (5), MM_COPIES (weight copies cycled so reads come from DRAM, 4),
    \\MM_CHECK_ROWS (rows checked against fp64, 16), MM_F32=1 (fp32 outputs).
;

const Args = extern struct { rows: i32, n: i32, k: i32, experts: i32 };
const Strides = extern struct { x_row: i32 = 0, y_row: i32 = 0, sums_row: i32 = 1, x_batch: i32 = 0, y_batch: i32 = 0, sums_batch: i32 = 0, w_batch: i32 = 0, pad: i32 = 0 };

const Kernel = struct { path: []const u8, bm: u32 = 64, bn: u32 = 64, threads: u32 = 128, entry: []const u8 = "tf_affine_mm", lib: mtl.Library = undefined, pipe: mtl.Pipeline = undefined, sums_pipe: mtl.Pipeline = undefined, sums: mtl.Buffer = undefined, y: mtl.Buffer = undefined, ms: [64]f64 = undefined, sums_ms: f64 = 0 };

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn bf16(v: f32) u16 { // round to nearest even
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn parseKernel(spec: []const u8) !Kernel {
    var it = std.mem.splitScalar(u8, spec, ':');
    var k: Kernel = .{ .path = it.next().? };
    if (it.next()) |v| k.bm = try std.fmt.parseInt(u32, v, 10);
    if (it.next()) |v| k.bn = try std.fmt.parseInt(u32, v, 10);
    if (it.next()) |v| k.threads = try std.fmt.parseInt(u32, v, 10);
    if (it.next()) |v| k.entry = v;
    return k;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 5) {
        std.debug.print("usage: {s}\n", .{usage});
        std.process.exit(2);
    }
    const M = try std.fmt.parseInt(u32, args[1], 10);
    const K = try std.fmt.parseInt(u32, args[2], 10);
    const N = try std.fmt.parseInt(u32, args[3], 10);
    const reps = envInt("MM_REPS", 16);
    const trials = @min(envInt("MM_TRIALS", 5), 64);
    const copies = envInt("MM_COPIES", 4);
    const check_rows = @min(envInt("MM_CHECK_ROWS", 16), M);
    const out_f32 = std.c.getenv("MM_F32") != null;
    const ob: usize = if (out_f32) 4 else 2;
    const kernels = try arena.alloc(Kernel, args.len - 4);
    for (kernels, args[4..]) |*k, s| k.* = try parseKernel(s);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const opts = mtl.ResourceOptions.shared;
    const G = K / 64;
    const x = try device.buffer(@as(usize, M) * K * 2, opts);
    const w = try device.buffer(copies * @as(usize, N) * K / 2, opts);
    const sc = try device.buffer(copies * @as(usize, N) * G * 2, opts);
    const bi = try device.buffer(copies * @as(usize, N) * G * 2, opts);
    defer for ([_]mtl.Buffer{ x, w, sc, bi }) |b| b.deinit();

    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (x.slice(u16, @as(usize, M) * K)) |*v| v.* = bf16(@floatCast(rnd.floatNorm(f64)));
    rnd.bytes(w.slice(u8, copies * @as(usize, N) * K / 2));
    for (sc.slice(u16, copies * @as(usize, N) * G), bi.slice(u16, copies * @as(usize, N) * G)) |*s, *b| {
        const scale = (rnd.float(f32) + 0.5) / 256.0;
        s.* = bf16(scale);
        b.* = bf16(-7.5 * scale * (0.9 + 0.2 * rnd.float(f32)));
    }

    const defines = try std.fmt.allocPrint(arena, "#define TF_BITS 4\n#define TF_GROUP 64\n#define TF_OUT_T {s}\n", .{if (out_f32) "float" else "bfloat"});
    for (kernels) |*k| {
        const f = try mtl.MappedFile.open(try arena.dupeSentinel(u8, k.path, 0));
        defer f.deinit();
        const body = try std.mem.replaceOwned(u8, arena, f.bytes[0..f.size], "#include \"../nax.h\"", ks.nax);
        const src = try std.mem.concat(arena, u8, &.{ defines, body });
        k.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        k.pipe = try mtl.Pipeline.init(device, k.lib, k.entry, false);
        k.y = try device.buffer(@as(usize, M) * N * ob, opts);
        k.sums_pipe = try mtl.Pipeline.init(device, k.lib, "tf_affine_row_sums", false);
        k.sums = try device.buffer(@as(usize, M) * G * 4, opts);
    }
    defer for (kernels) |*k| {
        k.sums.deinit();
        k.sums_pipe.deinit();
        k.y.deinit();
        k.pipe.deinit();
        k.lib.deinit();
    };
    for (kernels) |*k| { // each kernel's own row sums, timed over `reps` dispatches
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            e.setPipeline(k.sums_pipe);
            e.setBuffer(x, 0, 0);
            e.setValue(Args{ .rows = @intCast(M), .n = 0, .k = @intCast(K), .experts = 0 }, 5);
            e.setBuffer(k.sums, 0, 6);
            e.dispatchThreads(mtl.Size.of(G, M, 1), mtl.Size.of(@min(G, 64), 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("row sums failed: {s}\n", .{msg});
        k.sums_ms = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }

    const encode = struct {
        fn one(e: mtl.ComputeEncoder, k: *const Kernel, xb: mtl.Buffer, wb: mtl.Buffer, sb: mtl.Buffer, bb: mtl.Buffer, copy: usize, m: u32, kk: u32, n: u32) void {
            const g = kk / 64;
            e.setPipeline(k.pipe);
            e.setBuffer(xb, 0, 0);
            e.setBuffer(wb, copy * @as(usize, n) * kk / 2, 1);
            e.setBuffer(sb, copy * @as(usize, n) * g * 2, 2);
            e.setBuffer(bb, copy * @as(usize, n) * g * 2, 3);
            e.setBuffer(k.sums, 0, 4);
            e.setValue(Args{ .rows = @intCast(m), .n = @intCast(n), .k = @intCast(kk), .experts = 0 }, 5);
            e.setBuffer(k.y, 0, 6);
            e.setValue(Strides{ .x_row = @intCast(kk), .y_row = @intCast(n) }, 7);
            e.dispatchGroups(mtl.Size.of(n / k.bn, (m + k.bm - 1) / k.bm, 1), mtl.Size.of(k.threads, 1, 1));
        }
    };

    // timing: trials interleaved across kernels, each a buffer of `reps` dependent dispatches cycling the weight copies
    for (0..trials + 1) |t| for (kernels) |*k| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |r| encode.one(e, k, x, w, sc, bi, r % copies, M, K, N);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("{s}: failed: {s}\n", .{ k.path, msg });
            std.process.exit(1);
        }
        if (t > 0) k.ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    };
    // one run on copy 0 for the bits
    for (kernels) |*k| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        encode.one(e, k, x, w, sc, bi, 0, M, K, N);
        e.end();
        cb.commit();
        cb.wait();
    }

    // fp64 reference on the first check_rows rows: sum over k of x (s q + b), the affine weights' exact values
    const ref = try gpa.alloc(f64, check_rows * @as(usize, N));
    defer gpa.free(ref);
    const xs = x.slice(u16, @as(usize, M) * K);
    const wb = w.slice(u8, @as(usize, N) * K / 2);
    const sv = sc.slice(u16, @as(usize, N) * G);
    const bv = bi.slice(u16, @as(usize, N) * G);
    var ref_rms: f64 = 0;
    for (0..check_rows) |r| for (0..N) |n| {
        var sum: f64 = 0;
        for (0..G) |g| {
            var dot: f64 = 0;
            var xsum: f64 = 0;
            for (0..64) |j| {
                const kk = g * 64 + j;
                const q: f64 = @floatFromInt((wb[(n * K + kk) / 2] >> @intCast(4 * (kk % 2))) & 15);
                const xv: f64 = f32of(xs[r * K + kk]);
                dot += q * xv;
                xsum += xv;
            }
            sum += @as(f64, f32of(sv[n * G + g])) * dot + @as(f64, f32of(bv[n * G + g])) * xsum;
        }
        ref[r * N + n] = sum;
        ref_rms += sum * sum;
    };
    ref_rms = @sqrt(ref_rms / @as(f64, @floatFromInt(check_rows * @as(usize, N))));

    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    std.debug.print("M {d} K {d} N {d}: {d} dispatches a buffer, {d} weight copies, {s} out\n", .{ M, K, N, reps, copies, if (out_f32) "fp32" else "bf16" });
    const y0 = kernels[0].y.contents();
    for (kernels) |*k| {
        const ms = k.ms[0..trials];
        std.mem.sort(f64, ms, {}, std.sort.asc(f64));
        const med = ms[trials / 2];
        const y = k.y.contents();
        var differ: usize = 0;
        const total = @as(usize, M) * N;
        if (out_f32) {
            const a: [*]const u32 = @ptrCast(@alignCast(y));
            const b: [*]const u32 = @ptrCast(@alignCast(y0));
            for (0..total) |i| differ += @intFromBool(a[i] != b[i]);
        } else {
            const a: [*]const u16 = @ptrCast(@alignCast(y));
            const b: [*]const u16 = @ptrCast(@alignCast(y0));
            for (0..total) |i| differ += @intFromBool(a[i] != b[i]);
        }
        var err2: f64 = 0;
        var worst: f64 = 0;
        for (0..check_rows * @as(usize, N)) |i| {
            const got: f64 = if (out_f32) @as([*]const f32, @ptrCast(@alignCast(y)))[i] else f32of(@as([*]const u16, @ptrCast(@alignCast(y)))[i]);
            const d = @abs(got - ref[i]);
            err2 += d * d;
            worst = @max(worst, d);
        }
        const rms = @sqrt(err2 / @as(f64, @floatFromInt(check_rows * @as(usize, N))));
        std.debug.print("{s:<40} {d:8.3} ms (min {d:.3} max {d:.3}) {d:6.1} TFLOPS, row sums {d:.3} ms; differ from first {d}/{d}; err rms {e:.3} max {e:.3} (of rms {e:.3})\n", .{ k.path, med, ms[0], ms[trials - 1], flops / (med * 1e-3) / 1e12, k.sums_ms, differ, total, rms / ref_rms, worst / ref_rms, ref_rms });
    }
}
