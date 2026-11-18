//! Row-exact 4-bit prompt tiles (simd_qmm.metal) at one shape: chained speed, each output's bits against the first.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");

const usage =
    \\tf-sq-bench M K N ENTRY:ROWS:COLS:THREADS ...
    \\Each ENTRY is a kernel of core/simd_qmm.metal (or of SQ_FILE) taking tf_sq_mma's buffers, launched over tiles of ROWS
    \\rows and COLS columns at THREADS a threadgroup; the first is the reference every other output is compared with.
    \\SQ_REPS (dispatches a timed buffer, 16), SQ_TRIALS (5), SQ_COPIES (weight copies cycled so reads come from DRAM, 4).
;

const Args = extern struct { k: i32, n: i32, r: i32, one: f32 = 1.0 };
const Kernel = struct { entry: [:0]const u8, rows: u32, cols: u32, threads: u32, pipe: mtl.Pipeline = undefined, y: mtl.Buffer = undefined, ms: [64]f64 = undefined };

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn bf16(v: f32) u16 { // round to nearest even
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn parseKernel(a: std.mem.Allocator, spec: []const u8) !Kernel {
    var it = std.mem.splitScalar(u8, spec, ':');
    const entry = try a.dupeSentinel(u8, it.next().?, 0);
    const rows = try std.fmt.parseInt(u32, it.next() orelse return error.BadSpec, 10);
    const cols = try std.fmt.parseInt(u32, it.next() orelse return error.BadSpec, 10);
    const threads = try std.fmt.parseInt(u32, it.next() orelse return error.BadSpec, 10);
    return .{ .entry = entry, .rows = rows, .cols = cols, .threads = threads };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 5) {
        std.debug.print("usage: {s}\n", .{usage});
        std.process.exit(2);
    }
    const M = try std.fmt.parseInt(u32, args[1], 10);
    const K = try std.fmt.parseInt(u32, args[2], 10);
    const N = try std.fmt.parseInt(u32, args[3], 10);
    const reps = envInt("SQ_REPS", 16);
    const trials = @min(envInt("SQ_TRIALS", 5), 64);
    const copies = envInt("SQ_COPIES", 4);
    const kernels = try arena.alloc(Kernel, args.len - 4);
    for (kernels, args[4..]) |*k, s| k.* = try parseKernel(arena, s);

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

    var prng = std.Random.DefaultPrng.init(918);
    const rnd = prng.random();
    for (x.slice(u16, @as(usize, M) * K)) |*v| v.* = bf16(@floatCast(rnd.floatNorm(f64)));
    rnd.bytes(w.slice(u8, copies * @as(usize, N) * K / 2));
    for (sc.slice(u16, copies * @as(usize, N) * G), bi.slice(u16, copies * @as(usize, N) * G)) |*s, *b| {
        const scale = (rnd.float(f32) + 0.5) / 256.0;
        s.* = bf16(scale);
        b.* = bf16(-7.5 * scale * (0.9 + 0.2 * rnd.float(f32)));
    }

    const source: []const u8 = if (platform.c.getenv("SQ_FILE")) |path| blk: {
        const f = try mtl.MappedFile.open(std.mem.span(path));
        defer f.deinit();
        break :blk try arena.dupe(u8, f.bytes[0..f.size]);
    } else ks.core_simd_qmm;
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (kernels) |*k| {
        k.pipe = try mtl.Pipeline.init(device, lib, k.entry, false);
        if (k.pipe.maxThreads() < k.threads) {
            std.debug.print("{s}: at most {d} threads a threadgroup\n", .{ k.entry, k.pipe.maxThreads() });
            std.process.exit(1);
        }
        k.y = try device.buffer(@as(usize, M) * N * 2, opts);
    }
    defer for (kernels) |*k| {
        k.y.deinit();
        k.pipe.deinit();
    };

    const encode = struct {
        fn one(e: mtl.ComputeEncoder, k: *const Kernel, xb: mtl.Buffer, wb: mtl.Buffer, sb: mtl.Buffer, bb: mtl.Buffer, copy: usize, m: u32, kk: u32, n: u32) void {
            const g = kk / 64;
            e.setPipeline(k.pipe);
            e.setBuffer(xb, 0, 0);
            e.setBuffer(wb, copy * @as(usize, n) * kk / 2, 1);
            e.setBuffer(sb, copy * @as(usize, n) * g * 2, 2);
            e.setBuffer(bb, copy * @as(usize, n) * g * 2, 3);
            e.setValue(Args{ .k = @intCast(kk), .n = @intCast(n), .r = @intCast(m) }, 4);
            e.setBuffer(k.y, 0, 5);
            e.dispatchGroups(mtl.Size.of((n + k.cols - 1) / k.cols, (m + k.rows - 1) / k.rows, 1), mtl.Size.of(k.threads, 1, 1));
        }
    };

    // timing: trials interleaved across kernels, each a buffer of `reps` dispatches cycling the weight copies
    for (0..trials + 1) |t| for (kernels) |*k| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |r| encode.one(e, k, x, w, sc, bi, r % copies, M, K, N);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("{s}: failed: {s}\n", .{ k.entry, msg });
            std.process.exit(1);
        }
        if (t > 0) k.ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    };
    // one run on copy 0 for the bits
    for (kernels) |*k| {
        @memset(k.y.slice(u8, @as(usize, M) * N * 2), 0xff);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        encode.one(e, k, x, w, sc, bi, 0, M, K, N);
        e.end();
        cb.commit();
        cb.wait();
    }

    const flops = 2.0 * @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    std.debug.print("M {d} K {d} N {d}: {d} dispatches a buffer, {d} weight copies\n", .{ M, K, N, reps, copies });
    const total = @as(usize, M) * N;
    const y0 = kernels[0].y.slice(u16, total);
    for (kernels) |*k| {
        const ms = k.ms[0..trials];
        std.mem.sort(f64, ms, {}, std.sort.asc(f64));
        const med = ms[trials / 2];
        var differ: usize = 0;
        for (k.y.slice(u16, total), y0) |a, b| differ += @intFromBool(a != b);
        std.debug.print("{s:<24} {d:>3}x{d:<3} {d:>4} threads  {d:8.3} ms  {d:6.2} TFLOPS  bits differ {d}/{d}\n", .{ k.entry, k.rows, k.cols, k.threads, med, flops / (med * 1e-3) / 1e12, differ, total });
    }
}
