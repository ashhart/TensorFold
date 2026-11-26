//! The 2-16 row EXL3 bodies and the gate/up SwiGLU on a Metal GPU: each row's bits are the one-row kernel's.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const ref = @import("exl3_projection_row_ref.zig");
const one = @import("exl3_projection_row_test.zig");
const fmt = ex.format;
const shared = mtl.ResourceOptions.shared;

/// Rows of x: random bf16 in [-1, 1); row 0 holds a NaN, row 2 both infinities.
fn fillRows(x: mtl.Buffer, k: usize, rows: usize, rng: std.Random) void {
    const v = x.slice(u16, rows * k);
    for (v) |*e| e.* = one.bf16(rng.float(f32) * 2 - 1);
    v[3] = 0x7fc0;
    v[2 * k + 5] = 0x7f80;
    v[2 * k + 9] = 0xff80;
}

fn sameBits(name: []const u8, what: []const u8, count: usize, row: usize, want: []const u32, got: []const u32) !void {
    if (std.mem.indexOfDiff(u32, want, got)) |at| {
        std.debug.print("{s}, {d} rows, row {d}: {s} column {d}: 0x{x:0>8} where the one-row kernel gives 0x{x:0>8}\n", .{ name, count, row, what, at, got[at], want[at] });
        return error.Exl3RowsMismatch;
    }
}

test "every row of 2, 3, 4, 8 and 16 equals the one-row kernel on it alone and the serial reference, bit for bit" {
    var g = try one.Gpu.init();
    defer g.deinit();
    // every count through the rows kernels, whatever the machine
    g.p.gpu_cores = std.math.maxInt(u32);
    const main = g.main();
    var prng = std.Random.DefaultPrng.init(0x726f7773);
    for (one.v41) |s| {
        const parent_n: usize = if (s.view) 8192 else s.n;
        const w = try one.Weight.init(g.device, s.k, parent_n, s.halves, s.book, prng.random(), false);
        defer w.deinit();
        var m = w.matrix;
        const tn0: usize = if (s.view) 3 * 1024 / 16 else 0;
        if (s.view) {
            m.n = s.n;
            m.tile_k_stride = (parent_n / 16) * m.tileBytes();
            m.trellis.offset = tn0 * m.tileBytes();
            m.svh.offset = tn0 * 16 * 2;
        }
        const x = try g.device.buffer(16 * s.k * 2, shared);
        defer x.deinit();
        fillRows(x, s.k, 16, prng.random());
        const s1 = try ex.scratchBytes(s.k, s.n, 1);
        const s16 = try ex.scratchBytes(s.k, s.n, 16);
        // the rows launch, then each row alone, then tf_exl3_matmul_hoisted over the rows
        const scratch = try g.device.buffer(2 * s16 + 16 * s1, shared);
        defer scratch.deinit();
        const y = try g.device.buffer(3 * 16 * s.n * 4, shared);
        defer y.deinit();
        for ([_]usize{ 2, 3, 4, 8, 16 }) |count| {
            const rows: ex.Rows = .{ .count = count, .x_stride = s.k, .y_stride = s.n, .output = .f32 };
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try ex.encode(&g.p, e, m, .{ .buffer = x }, .{ .buffer = y }, .{ .buffer = scratch }, rows);
            for (0..count) |r| {
                var alone = rows;
                alone.count = 1;
                try ex.encode(&g.p, e, m, .{ .buffer = x, .offset = r * s.k * 2 }, .{ .buffer = y, .offset = (16 + r) * s.n * 4 }, .{ .buffer = scratch, .offset = s16 + r * s1 }, alone);
            }
            try ex.encode(&main, e, m, .{ .buffer = x }, .{ .buffer = y, .offset = 32 * s.n * 4 }, .{ .buffer = scratch, .offset = s16 + 16 * s1 }, rows);
            try complete(cb, e);
            const bytes = scratch.contents();
            const sb = try ex.scratchBytes(s.k, s.n, count);
            const rotated: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, bytes[0 .. count * s.k * 2]));
            const sums: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, bytes[sb - count * s.n * 4 ..][0 .. count * s.n * 4]));
            const hoisted: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, bytes[s16 + 16 * s1 + sb - count * s.n * 4 ..][0 .. count * s.n * 4]));
            const ys = y.slice(u32, 3 * 16 * s.n);
            try sameBits(s.name, "tf_exl3_matmul_hoisted's sums", count, 0, hoisted, sums);
            try sameBits(s.name, "tf_exl3_matmul_hoisted's outputs", count, 0, ys[32 * s.n ..][0 .. count * s.n], ys[0 .. count * s.n]);
            for (0..count) |r| {
                const own = bytes[s16 + r * s1 ..][0..s1];
                const alone_sums: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, own[s1 - s.n * 4 ..]));
                try sameBits(s.name, "sum", count, r, alone_sums, sums[r * s.n ..][0..s.n]);
                try sameBits(s.name, "output", count, r, ys[(16 + r) * s.n ..][0..s.n], ys[r * s.n ..][0..s.n]);
            }
            // the last finite row against the serial reference, first, middle and last 128-column blocks
            const rr = if (count == 3) 1 else count - 1;
            const blocks = s.n / 128;
            var want: [128]f32 = undefined;
            for ([_]usize{ 0, blocks / 2, blocks - 1 }) |b| {
                ref.dot(m.width, s.book, w.words(), parent_n / 16, rotated[rr * s.k ..][0..s.k], tn0 + b * 8, &want);
                const bits: [128]u32 = @bitCast(want);
                try sameBits(s.name, "the serial reference", count, rr, &bits, sums[rr * s.n + b * 128 ..][0..128]);
            }
        }
    }
}

fn complete(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
    return one.complete(cb, e);
}

/// One expert's w1 and w3 at DeepSeek V4.1's shared shape, each its own width, codebook and (optional) bias.
const Pair = struct { gate: struct { u16, fmt.Codebook }, up: struct { u16, fmt.Codebook }, bias: bool };

test "the gate/up projections and their SwiGLU equal both projections then tf_dsv41_f_swiglu bit for bit at 1-16 rows" {
    var g = try one.Gpu.init();
    defer g.deinit();
    const main = g.main();
    var q = try ex.gate_up.Pipelines.load(g.device);
    defer q.deinit();
    const lib = try mtl.Library.fromSource(g.device, sources.dsv41.fused, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const swiglu = try mtl.Pipeline.init(g.device, lib, "tf_dsv41_f_swiglu", false);
    defer swiglu.deinit();
    const k = 5120;
    const n = 2304;
    const limit: f32 = 10.0;
    var prng = std.Random.DefaultPrng.init(0x73776967);
    const pairs = [_]Pair{
        .{ .gate = .{ 5, .mul1 }, .up = .{ 4, .mcg }, .bias = false },
        .{ .gate = .{ 12, .inst3 }, .up = .{ 3, .mul1 }, .bias = true },
    };
    for (pairs) |pc| {
        const gw = try one.Weight.init(g.device, k, n, pc.gate[0], pc.gate[1], prng.random(), false);
        defer gw.deinit();
        const uw = try one.Weight.init(g.device, k, n, pc.up[0], pc.up[1], prng.random(), false);
        defer uw.deinit();
        const bias = try g.device.buffer(2 * n * 2, shared);
        defer bias.deinit();
        for (bias.slice(f16, 2 * n)) |*v| v.* = @floatCast(prng.random().float(f32) * 8 - 4);
        var gm = gw.matrix;
        var um = uw.matrix;
        if (pc.bias) {
            gm.bias = .{ .buffer = bias };
            um.bias = .{ .buffer = bias, .offset = n * 2 };
        }
        const x = try g.device.buffer(16 * k * 2, shared);
        defer x.deinit();
        fillRows(x, k, 16, prng.random());
        const gs = try ex.gate_up.scratchBytes(k, n, 16);
        const scratch = try g.device.buffer(2 * gs, shared);
        defer scratch.deinit();
        // fused y and yq, then main's G, U, y and yq
        const out = try g.device.buffer(6 * 16 * n * 2, shared);
        defer out.deinit();
        var clamped = [2]usize{ 0, 0 };
        // with no cores 3 to 16 rows take tf_exl3_matmul_hoisted, with all of them the rows kernels
        for ([_]u32{ 0, std.math.maxInt(u32) }) |cores| for ([_]usize{ 1, 2, 3, 4, 8, 16 }) |count| {
            g.p.gpu_cores = cores;
            @memset(out.contents()[0..out.length()], 0xa5);
            const rows: ex.Rows = .{ .count = count, .x_stride = k, .y_stride = n };
            const at = struct {
                fn of(b: mtl.Buffer, i: usize) ex.View {
                    return .{ .buffer = b, .offset = i * 16 * n * 2 };
                }
            };
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try ex.gate_up.encode(&g.p, &q, e, gm, um, .{ .buffer = x }, at.of(out, 0), at.of(out, 1), .{ .buffer = scratch }, rows, limit);
            try ex.encode(&main, e, gm, .{ .buffer = x }, at.of(out, 2), .{ .buffer = scratch, .offset = gs }, rows);
            try ex.encode(&main, e, um, .{ .buffer = x }, at.of(out, 3), .{ .buffer = scratch, .offset = gs + gs / 2 }, rows);
            e.barrier();
            e.setPipeline(swiglu);
            for (2..6, 0..) |i, b| e.setBuffer(out, i * 16 * n * 2, b);
            e.setValue(extern struct { n: u32, limit: f32 }{ .n = @intCast(count * n), .limit = limit }, 4);
            e.dispatchThreads(mtl.Size.of(count * n, 1, 1), mtl.Size.of(256, 1, 1));
            try complete(cb, e);
            const v = out.slice(u16, 6 * 16 * n);
            const len = count * n;
            for ([_]usize{ 0, 1 }, [_][]const u8{ "y", "yq" }) |i, what| {
                if (std.mem.indexOfDiff(u16, v[(4 + i) * 16 * n ..][0..len], v[i * 16 * n ..][0..len])) |d| {
                    std.debug.print("gate/up, {d} rows, {d} cores: {s}[{d}] 0x{x:0>4} where main's chain gives 0x{x:0>4}\n", .{ count, cores, what, d, v[i * 16 * n + d], v[(4 + i) * 16 * n + d] });
                    return error.Exl3GateUpMismatch;
                }
            }
            for ([_]usize{ 2, 3 }, &clamped) |i, *c| {
                for (v[i * 16 * n ..][0..len]) |b| c.* += @intFromBool(@abs(@as(f32, @bitCast(@as(u32, b) << 16))) > limit);
            }
        };
        // the clamps took part: both streams passed the limit somewhere
        try std.testing.expect(clamped[0] > 0 and clamped[1] > 0);
    }
}

test "gate/up refuses y on the up's scratch and yq on the gate's, which the epilogue reads while it writes both" {
    var g = try one.Gpu.init();
    defer g.deinit();
    var q = try ex.gate_up.Pipelines.load(g.device);
    defer q.deinit();
    const k = 256;
    const n = 128;
    var prng = std.Random.DefaultPrng.init(0x616c6961);
    const gw = try one.Weight.init(g.device, k, n, 5, .mul1, prng.random(), false);
    defer gw.deinit();
    const uw = try one.Weight.init(g.device, k, n, 4, .mcg, prng.random(), false);
    defer uw.deinit();
    const x = try g.device.buffer(k * 2, shared);
    defer x.deinit();
    fillRows(x, k, 1, prng.random());
    const half = try ex.scratchBytes(k, n, 1);
    const scratch = try g.device.buffer(2 * half, shared);
    defer scratch.deinit();
    const out = try g.device.buffer(2 * n * 2, shared);
    defer out.deinit();
    const rows: ex.Rows = .{ .count = 1, .x_stride = k, .y_stride = n };
    const xv: ex.View = .{ .buffer = x };
    const sv: ex.View = .{ .buffer = scratch };
    const y: ex.View = .{ .buffer = out };
    const yq: ex.View = .{ .buffer = out, .offset = n * 2 };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.concurrent);
    const enc = ex.gate_up.encode;
    try std.testing.expectError(error.AliasedExl3View, enc(&g.p, &q, e, gw.matrix, uw.matrix, xv, .{ .buffer = scratch, .offset = half }, yq, sv, rows, 10));
    try std.testing.expectError(error.AliasedExl3View, enc(&g.p, &q, e, gw.matrix, uw.matrix, xv, y, .{ .buffer = scratch }, sv, rows, 10));
    try enc(&g.p, &q, e, gw.matrix, uw.matrix, xv, y, yq, sv, rows, 10);
    try complete(cb, e);
}
