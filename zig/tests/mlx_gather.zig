//! The reference's gathered quantised matmul (ml-explore/mlx v0.32.4, MIT) on our shapes: the kernel the sibling
//! engine dispatches for its expert bodies, timed here so the graft's prize is a measurement rather than a claim.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

const Args = extern struct { M: i32, N: i32, K: i32, num_groups: i32 };

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const M: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 3619 * 6;
    const K: u32 = 5120;
    const N: u32 = 2304;
    const gs: u32 = 64;
    const reps: u32 = 8;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const opts = mtl.ResourceOptions.shared;

    const path = if (args.len > 2) args[2] else "zig/kernels/metal/core/mlx/gather.metal";
    const f = try mtl.MappedFile.open(try arena.dupeSentinel(u8, path, 0));
    defer f.deinit();
    const src = try std.mem.replaceOwned(u8, arena, f.bytes[0..f.size], "#include \"mlx_steel.h\"", ks.core_mlx_steel);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const entry: []const u8 = if (std.c.getenv("MLX_ENTRY")) |v| try arena.dupe(u8, std.mem.span(v)) else "tf_gather_qmm_f32_gs64_b4";
    const threads: u32 = if (std.c.getenv("MLX_THREADS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 128 else 128;
    const bmv: u32 = if (std.c.getenv("MLX_BM")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64 else 64;
    const bnv: u32 = if (std.c.getenv("MLX_BN")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64 else 64;
    const pipe = try mtl.Pipeline.initSpecialized(device, lib, entry);
    defer pipe.deinit();

    const G = K / gs;
    // MLX_BANKS=n: n distinct expert banks and a cumulative offsets table over them -- the shape the sibling
    // engine dispatches, where each row tile reads a different bank and the weight stream, not the MMA, dominates.
    const banks: u32 = if (std.c.getenv("MLX_BANKS")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1 else 1;
    const x = try device.buffer(@as(usize, M) * K * 4, opts);
    const w = try device.buffer(@as(usize, banks) * N * (K / 8) * 4, opts);
    const sc = try device.buffer(@as(usize, banks) * N * G * 4, opts);
    const bi = try device.buffer(@as(usize, banks) * N * G * 4, opts);
    const off = try device.buffer(@as(usize, banks + 1) * 4, opts);
    const y = try device.buffer(@as(usize, M) * N * 4, opts);
    defer for ([_]mtl.Buffer{ x, w, sc, bi, off, y }) |b| b.deinit();

    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (x.slice(f32, @as(usize, M) * K)) |*v| v.* = @floatCast(rnd.floatNorm(f64));
    rnd.bytes(w.slice(u8, @as(usize, banks) * N * (K / 8) * 4));
    for (sc.slice(f32, @as(usize, banks) * N * G), bi.slice(f32, @as(usize, banks) * N * G)) |*s, *b| {
        const scale = (rnd.float(f32) + 0.5) / 256.0;
        s.* = scale;
        b.* = 0;
    }
    {
        const ov = off.slice(i32, @as(usize, banks) + 1);
        for (ov, 0..) |*v, i| v.* = @intCast(@as(u64, M) * i / banks);
        ov[banks] = @intCast(M);
    }

    // MLX_TRUTH=1: load the buffers MLX's own gather_qmm consumed (mlxtruth_*.bin) and compare our kernel's
    // output with MLX's, entry by entry. Same kernel family, two runtimes: that is the harness's gate.
    if (std.c.getenv("MLX_TRUTH") != null) {
        const files = [_]struct { name: []const u8, buf: mtl.Buffer, bytes: usize }{
            .{ .name = "/tmp/mlxtruth_x.bin", .buf = x, .bytes = @as(usize, M) * K * 4 },
            .{ .name = "/tmp/mlxtruth_w.bin", .buf = w, .bytes = @as(usize, N) * (K / 8) * 4 },
            .{ .name = "/tmp/mlxtruth_s.bin", .buf = sc, .bytes = @as(usize, N) * G * 4 },
            .{ .name = "/tmp/mlxtruth_b.bin", .buf = bi, .bytes = @as(usize, N) * G * 4 },
        };
        for (files) |tf| {
            const mf = try mtl.MappedFile.open(try arena.dupeSentinel(u8, tf.name, 0));
            defer mf.deinit();
            if (mf.size != tf.bytes) return error.BadTruthFile;
            @memcpy(tf.buf.contents()[0..tf.bytes], mf.bytes[0..tf.bytes]);
        }
        const truth = try mtl.MappedFile.open("/tmp/mlxtruth_y.bin");
        defer truth.deinit();
        const want: [*]const f32 = @ptrCast(@alignCast(truth.bytes));
        {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            e.setPipeline(pipe);
            e.setBuffer(x, 0, 0);
            e.setBuffer(w, 0, 1);
            e.setBuffer(sc, 0, 2);
            e.setBuffer(bi, 0, 3);
            e.setBuffer(off, 0, 4);
            e.setBuffer(y, 0, 5);
            // The kernel takes four separate `constant int&` buffers, not one struct: each is bound on its own.
            e.setValue(@as(i32, @intCast(M)), 6);
            e.setValue(@as(i32, @intCast(N)), 7);
            e.setValue(@as(i32, @intCast(K)), 8);
            e.setValue(@as(i32, @intCast(banks)), 9);
            e.dispatchGroups(mtl.Size.of(N / bnv, (M + bmv - 1) / bmv, 1), mtl.Size.of(threads, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
        }
        const got = y.slice(f32, @as(usize, M) * N);
        var worst: f64 = 0;
        var bad: usize = 0;
        for (0..@as(usize, M) * N) |i| {
            const d = @abs(@as(f64, got[i]) - @as(f64, want[i])) / @max(1.0, @abs(@as(f64, want[i])));
            if (d > 1e-3) bad += 1;
            if (d > worst) worst = d;
        }
        std.debug.print("mlx truth: {d} of {d} entries beyond 1e-3, worst {e:.3}\n", .{ bad, @as(usize, M) * N, worst });
        std.debug.print("first: ours {d:.4} {d:.4} {d:.4} | mlx {d:.4} {d:.4} {d:.4}\n", .{ got[0], got[1], got[2], want[0], want[1], want[2] });
        std.debug.print("last:  ours {d:.4} | mlx {d:.4}\n", .{ got[@as(usize, M) * N - 1], want[@as(usize, M) * N - 1] });
        if (bad > 0) return error.Mismatch;
    }
    var best: f64 = 1e9;
    for (0..reps) |_| {
        const t0 = std.Io.Timestamp.now(init.io, .awake);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        e.setBuffer(x, 0, 0);
        e.setBuffer(w, 0, 1);
        e.setBuffer(sc, 0, 2);
        e.setBuffer(bi, 0, 3);
        e.setBuffer(off, 0, 4);
        e.setBuffer(y, 0, 5);
        // The kernel takes four separate `constant int&` buffers, not one struct: each is bound on its own.
        e.setValue(@as(i32, @intCast(M)), 6);
        e.setValue(@as(i32, @intCast(N)), 7);
        e.setValue(@as(i32, @intCast(K)), 8);
        e.setValue(@as(i32, @intCast(banks)), 9);
        e.dispatchGroups(mtl.Size.of(N / bnv, (M + bmv - 1) / bmv, 1), mtl.Size.of(threads, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        const t1 = std.Io.Timestamp.now(init.io, .awake);
        const ms = @as(f64, @floatFromInt(t0.durationTo(t1).toNanoseconds())) / 1e6;
        if (ms < best) best = ms;
    }
    const bytes = @as(f64, @floatFromInt(@as(usize, N) * (K / 8) * 4)) + @as(f64, @floatFromInt(@as(usize, M) * K * 4));
    const mac = @as(f64, @floatFromInt(M)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(K));
    std.debug.print("mlx gather_qmm: rows {d} K {d} N {d} gs {d} -> {d:.3} ms, {d:.1} GB/s, {d:.2} TMAC/s ({d:.1} G weights/s at 4 bits)\n", .{
        M, K, N, gs, best, bytes / best / 1e6, mac / best / 1e9, @as(f64, @floatFromInt(N * K)) * 8.0 / 4.0 / best / 1e9 * 1000.0,
    });
}
