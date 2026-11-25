//! Inkling 2 DFlash 2 accelerated BF16 NAX / MMA projections and 32-row Q4 draft head pipeline.
const std = @import("std");
const bf = @import("bf16_round");
const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const sources = @import("inkling2_metal");
const q4 = @import("draft_head_q4.zig");
const fwd = @import("forward.zig");
const Model = @import("model.zig").Model;

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const Args = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

pub const Kind = enum {
    bf16_nax_16,
    bf16_nax_32,

    pub fn symbol(k: Kind) [:0]const u8 {
        return switch (k) {
            .bf16_nax_16 => "tf_ink2_df2_bf16_nax_16",
            .bf16_nax_32 => "tf_ink2_df2_bf16_nax_32",
        };
    }

    pub fn requiredThreads(k: Kind) u32 {
        return switch (k) {
            .bf16_nax_16 => 32,
            .bf16_nax_32 => 128,
        };
    }

    pub fn label(k: Kind) []const u8 {
        return switch (k) {
            inline else => |v| std.fmt.comptimePrint("metal inkling2 dflash2_fast {s} {s}", .{ @tagName(v), comptime v.symbol() }),
        };
    }
};

pub const Pipelines = struct {
    nax16: mtl.Pipeline,
    nax32: mtl.Pipeline,
    head_q4: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const text = if (device.tensorUnits()) sources.inkling2_dflash2_fast else sources.inkling2_dflash2_fast_pre_m5;
        return loadSource(device, text);
    }

    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const opts = mtl.CompileOptions{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
        const lib = try mtl.Library.fromSource(device, text, opts);
        defer lib.deinit();
        const p16 = try mtl.Pipeline.init(device, lib, Kind.bf16_nax_16.symbol(), false);
        errdefer p16.deinit();
        if (p16.maxThreads() < Kind.bf16_nax_16.requiredThreads()) return error.MetalThreadgroupTooSmall;
        const p32 = try mtl.Pipeline.init(device, lib, Kind.bf16_nax_32.symbol(), false);
        errdefer p32.deinit();
        if (p32.maxThreads() < Kind.bf16_nax_32.requiredThreads()) return error.MetalThreadgroupTooSmall;
        const pq4 = try mtl.Pipeline.init(device, lib, q4.batch_symbol, false);
        errdefer pq4.deinit();
        if (pq4.maxThreads() < q4.batch_threads) return error.MetalThreadgroupTooSmall;
        return .{ .nax16 = p16, .nax32 = p32, .head_q4 = pq4 };
    }

    pub fn deinit(p: *Pipelines) void {
        p.head_q4.deinit();
        p.nax32.deinit();
        p.nax16.deinit();
        p.* = undefined;
    }

    pub fn dispatch(p: *const Pipelines, target: *const Model, e: mtl.ComputeEncoder, weight: ck.Tensor, input: Ref, output: Ref, rows: usize, prefer_16: bool) void {
        const k: usize = weight.shape[weight.rank - 1];
        const n: usize = weight.shape[weight.rank - 2];
        if (target.device.tensorUnits() and rows > 0 and n > 0 and k > 0 and k % 32 == 0) {
            const use_32 = rows > 16 and !prefer_16 and n < 65536;
            const pipe = if (use_32) p.nax32 else p.nax16;
            const tile_n: usize = if (use_32) 64 else 32;
            const tile_m: usize = if (use_32) 32 else 16;
            const th: usize = if (use_32) 128 else 32;
            encodeNax(pipe, e, input, .{ .buf = weight.buffer, .off = weight.offset }, output, .{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(k), .ldy = @intCast(n) }, tile_n, tile_m, th);
            return;
        }
        fwd.project(target, e, weight, input, output, rows);
    }

    pub fn encodeNax(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, x: Ref, w_ref: Ref, out: Ref, a: Args, tile_n: usize, tile_m: usize, threads: usize) void {
        e.setPipeline(pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w_ref.buf, w_ref.off, 1);
        e.setValue(a, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of((@as(usize, a.n) + tile_n - 1) / tile_n, (@as(usize, a.rows) + tile_m - 1) / tile_m, 1), mtl.Size.of(threads, 1, 1));
    }
};

test "DFlash2 fast BF16 NAX symbols, ABI and Metal multi-row parity" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(Args));
    try std.testing.expectEqualStrings("metal inkling2 dflash2_fast bf16_nax_16 tf_ink2_df2_bf16_nax_16", Kind.bf16_nax_16.label());
    try std.testing.expectEqualStrings("metal inkling2 dflash2_fast bf16_nax_32 tf_ink2_df2_bf16_nax_32", Kind.bf16_nax_32.label());
    if (comptime @import("builtin").os.tag != .macos) return error.SkipZigTest;
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    defer dev.deinit();
    const queue = try dev.queue();
    defer queue.deinit();
    var pipes = try Pipelines.load(dev);
    defer pipes.deinit();
    var pre = try Pipelines.loadSource(dev, sources.inkling2_dflash2_fast_pre_m5);
    defer pre.deinit();
    const m: usize = 32;
    const n: usize = 67;
    const k: usize = 96;
    const xb = try dev.buffer(m * k * 2, mtl.ResourceOptions.shared);
    defer xb.deinit();
    const wb = try dev.buffer(n * k * 2, mtl.ResourceOptions.shared);
    defer wb.deinit();
    const y16 = try dev.buffer(m * n * 2, mtl.ResourceOptions.shared);
    defer y16.deinit();
    const y32 = try dev.buffer(m * n * 2, mtl.ResourceOptions.shared);
    defer y32.deinit();
    const ypre = try dev.buffer(m * n * 2, mtl.ResourceOptions.shared);
    defer ypre.deinit();
    var prng = std.Random.DefaultPrng.init(2026);
    for (xb.slice(u16, m * k)) |*slot| slot.* = bf.bits((prng.random().float(f32) * 2.0 - 1.0) * 0.75);
    for (wb.slice(u16, n * k)) |*slot| slot.* = bf.bits((prng.random().float(f32) * 2.0 - 1.0) * 0.75);
    const run = struct {
        fn go(qu: mtl.Queue, p: mtl.Pipeline, x: mtl.Buffer, wt: mtl.Buffer, out: mtl.Buffer, rows: u32, cols: u32, width: u32, tn: usize, tm: usize, th: usize) !void {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = qu.commandBuffer();
            const e = cb.compute(.serial);
            Pipelines.encodeNax(p, e, .{ .buf = x }, .{ .buf = wt }, .{ .buf = out }, .{ .rows = rows, .n = cols, .k = width, .ldy = cols }, tn, tm, th);
            e.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
        }
    }.go;
    try run(queue, pipes.nax16, xb, wb, y16, 8, n, k, 32, 16, 32);
    try run(queue, pipes.nax32, xb, wb, y32, 32, n, k, 64, 32, 128);
    try run(queue, pre.nax32, xb, wb, ypre, 32, n, k, 64, 32, 128);
    try std.testing.expectEqualSlices(u16, y16.slice(u16, 8 * n), y32.slice(u16, 8 * n));
    const got_pre = ypre.slice(u16, 32 * n);
    const xs = xb.slice(u16, m * k);
    const ws = wb.slice(u16, n * k);
    for (0..32) |r| {
        for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |d| acc += q4.asFloat(xs[r * k + d]) * q4.asFloat(ws[c * k + d]);
            const want = bf.bits(@floatCast(acc));
            const tol = @max(@abs(q4.asFloat(want)) * 0.02, 0.05);
            try std.testing.expect(@abs(q4.asFloat(want) - q4.asFloat(got_pre[r * n + c])) <= tol);
        }
    }
}
