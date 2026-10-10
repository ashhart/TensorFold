//! Routed-expert prompt kernels from .metal files compiled at run time: a chunk's skewed top-k routing, chained speed, bits against the first.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

const usage =
    \\tf-moe-bench TOKENS FILE[:BM[:ENTRY]] ...
    \\GLU mode (default): each FILE's tf_affine_gather_glu_BM over a chunk's token rows read through the pair order,
    \\K 4096 into act [pairs, 2048]. MOE_MODE=down: tf_affine_gather_BM over act [pairs, 2048] into [pairs, 4096].
    \\MOE_EXPERTS (288), MOE_TOPK (8), MOE_SKEW (Zipf exponent of expert popularity, 0.6), MOE_EXACT (equal rows an expert),
    \\MOE_REPS (8), MOE_TRIALS (3).
;

const Args = extern struct { rows: i32, n: i32, k: i32, experts: i32 };

const Kernel = struct { path: []const u8, bm: u32 = 64, entry: []const u8 = "", lib: mtl.Library = undefined, pipe: mtl.Pipeline = undefined, y: mtl.Buffer = undefined, ms: [64]f64 = undefined };

fn envInt(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn envFloat(name: [:0]const u8, default: f64) f64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch default;
}

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print("usage: {s}\n", .{usage});
        std.process.exit(2);
    }
    const down = if (std.c.getenv("MOE_MODE")) |v| std.mem.eql(u8, std.mem.span(v), "down") else false;
    const T = try std.fmt.parseInt(u32, args[1], 10);
    const E: u32 = @intCast(envInt("MOE_EXPERTS", 288));
    const topk: u32 = @intCast(envInt("MOE_TOPK", 8));
    const skew = envFloat("MOE_SKEW", 0.6);
    const reps = envInt("MOE_REPS", 8);
    const trials = @min(envInt("MOE_TRIALS", 3), 64);
    const n = T * topk; // pairs
    const K: u32 = if (down) 2048 else 4096;
    const N: u32 = if (down) 4096 else 2048; // outputs a pair (GLU: act columns, gate and up each N rows)
    const kernels = try arena.alloc(Kernel, args.len - 2);
    for (kernels, args[2..]) |*k, spec| {
        var it = std.mem.splitScalar(u8, spec, ':');
        k.* = .{ .path = it.next().? };
        if (it.next()) |v| k.bm = try std.fmt.parseInt(u32, v, 10);
        k.entry = if (it.next()) |v| v else try std.fmt.allocPrint(arena, "{s}_{d}", .{ if (down) "tf_affine_gather" else "tf_affine_gather_glu", k.bm });
    }

    // routing: each token's topk distinct experts drawn by Zipf popularity, pairs sorted by expert (counting sort)
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const weight = try arena.alloc(f64, E);
    var total: f64 = 0;
    for (weight, 0..) |*w, e| {
        w.* = 1.0 / std.math.pow(f64, @as(f64, @floatFromInt(e + 1)), skew);
        total += w.*;
    }
    const pick = try arena.alloc(u32, n);
    const exact = std.c.getenv("MOE_EXACT") != null; // pairs dealt round-robin: every expert the same row count
    if (exact) for (pick, 0..) |*q, i| {
        q.* = @intCast((i % topk + (i / topk) * topk) % E);
    };
    if (!exact) for (0..T) |t| {
        var chosen: usize = 0;
        while (chosen < topk) {
            var r = rnd.float(f64) * total;
            var e: u32 = 0;
            while (e + 1 < E and r >= weight[e]) : (e += 1) r -= weight[e];
            var dup = false;
            for (pick[t * topk ..][0..chosen]) |q| dup = dup or q == e;
            if (!dup) {
                pick[t * topk + chosen] = e;
                chosen += 1;
            }
        }
    };
    const counts = try arena.alloc(u32, E);
    @memset(counts, 0);
    for (pick) |e| counts[e] += 1;
    const opts = mtl.ResourceOptions.shared;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const offsets = try device.buffer(E * 4, opts);
    const order = try device.buffer(n * 4, opts);
    defer offsets.deinit();
    defer order.deinit();
    {
        const o = offsets.slice(i32, E);
        var at: i32 = 0;
        for (o, counts) |*x, c| {
            x.* = at;
            at += @intCast(c);
        }
        const fill = try arena.alloc(u32, E);
        for (fill, o) |*f, x| f.* = @intCast(x);
        const ord = order.slice(i32, n);
        for (pick, 0..) |e, i| {
            ord[fill[e]] = @intCast(i);
            fill[e] += 1;
        }
    }
    var tiles64: usize = 0;
    var full: usize = 0;
    for (counts) |c| {
        tiles64 += (c + 63) / 64;
        full += c;
    }
    const G = K / 64;
    const rows_in: usize = if (down) n else T; // the input's rows: GLU reads token rows through the order
    const x = try device.buffer(rows_in * K * 2, opts);
    const xs = try device.buffer(rows_in * G * 4, opts);
    const mats: usize = if (down) 1 else 2;
    const w = try device.buffer(mats * @as(usize, E) * N * K / 2, opts);
    const sc = try device.buffer(mats * @as(usize, E) * N * G * 2, opts);
    const bi = try device.buffer(mats * @as(usize, E) * N * G * 2, opts);
    defer for ([_]mtl.Buffer{ x, xs, w, sc, bi }) |b| b.deinit();
    for (x.slice(u16, rows_in * K)) |*v| v.* = bf16(@floatCast(rnd.floatNorm(f64)));
    rnd.bytes(w.slice(u8, mats * @as(usize, E) * N * K / 2));
    for (sc.slice(u16, mats * @as(usize, E) * N * G), bi.slice(u16, mats * @as(usize, E) * N * G)) |*s, *b| {
        const scale = (rnd.float(f32) + 0.5) / 256.0;
        s.* = bf16(scale);
        b.* = bf16(-7.5 * scale * (0.9 + 0.2 * rnd.float(f32)));
    }
    {
        const xv = x.slice(u16, rows_in * K);
        const sv = xs.slice(f32, rows_in * G);
        for (0..rows_in) |r| for (0..G) |g| {
            var s: f32 = 0;
            for (0..64) |j| s += @as(f32, @bitCast(@as(u32, xv[r * K + g * 64 + j]) << 16));
            sv[r * G + g] = s;
        };
    }
    const defines = "#define TF_BITS 4\n#define TF_GROUP 64\n#define TF_OUT_T bfloat\n";
    for (kernels) |*k| {
        const f = try mtl.MappedFile.open(try arena.dupeSentinel(u8, k.path, 0));
        defer f.deinit();
        const body = try std.mem.replaceOwned(u8, arena, f.bytes[0..f.size], "#include \"../nax.h\"", ks.nax);
        k.lib = try mtl.Library.fromSource(device, try std.mem.concat(arena, u8, &.{ defines, body }), mtl.CompileOptions.mlx());
        k.pipe = try mtl.Pipeline.init(device, k.lib, try arena.dupeSentinel(u8, k.entry, 0), false);
        k.y = try device.buffer(@as(usize, n) * N * 2, opts);
    }
    defer for (kernels) |*k| {
        k.y.deinit();
        k.pipe.deinit();
        k.lib.deinit();
    };
    const enc = struct {
        fn one(e: mtl.ComputeEncoder, k: *const Kernel, b: [7]mtl.Buffer, rows: u32, nn: u32, kk: u32, experts: u32, top: u32, is_down: bool) void {
            const g = kk / 64;
            const per = @as(usize, experts) * nn; // rows of one matrix
            e.setPipeline(k.pipe);
            e.setBuffer(b[0], 0, 0);
            e.setBuffer(b[2], 0, 1);
            e.setBuffer(b[3], 0, 2);
            e.setBuffer(b[4], 0, 3);
            e.setBuffer(b[1], 0, 4);
            e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(nn), .k = @intCast(kk), .experts = @intCast(experts) }, 5);
            e.setBuffer(k.y, 0, 6);
            e.setBuffer(b[5], 0, 7);
            if (!is_down) {
                e.setBuffer(b[6], 0, 8);
                e.setBuffer(b[2], per * kk / 2, 9);
                e.setBuffer(b[3], per * g * 2, 10);
                e.setBuffer(b[4], per * g * 2, 11);
                e.setValue([2]f32{ @floatFromInt(top), 10.0 }, 12);
            }
            const tiles = @min(rows, (rows + k.bm - 1) / k.bm + experts - 1);
            const cols: u32 = if (is_down) 64 else 32;
            e.dispatchGroups(mtl.Size.of(nn / cols, tiles, 1), mtl.Size.of(32, 2, 2));
        }
    };
    const bufs = [7]mtl.Buffer{ x, xs, w, sc, bi, offsets, order };
    for (0..trials + 1) |t| for (kernels) |*k| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| enc.one(e, k, bufs, n, N, K, E, topk, down);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("{s}: failed: {s}\n", .{ k.path, msg });
            std.process.exit(1);
        }
        if (t > 0) k.ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    };
    const flops = 2.0 * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(K)) * @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(mats));
    var biggest: u32 = 0;
    for (counts) |c| biggest = @max(biggest, c);
    std.debug.print("{s}: {d} tokens, {d} pairs over {d} experts (largest {d}, 64-row tiles {d}, rows they hold {d}%), K {d} N {d}\n", .{ if (down) "down" else "gate+up", T, n, E, biggest, tiles64, full * 100 / (tiles64 * 64), K, N });
    const y0 = kernels[0].y.slice(u16, @as(usize, n) * N);
    for (kernels) |*k| {
        const ms = k.ms[0..trials];
        std.mem.sort(f64, ms, {}, std.sort.asc(f64));
        var differ: usize = 0;
        for (k.y.slice(u16, @as(usize, n) * N), y0) |a, b| differ += @intFromBool(a != b);
        std.debug.print("{s:<44} {s:<26} {d:8.3} ms {d:6.1} TFLOPS  differ from first {d}\n", .{ k.path, k.entry, ms[trials / 2], flops / (ms[trials / 2] * 1e-3) / 1e12, differ });
    }
}
