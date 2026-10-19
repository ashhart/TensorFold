//! Borrowed output shards must retain the full projection's bits, including padded checkpoint layouts.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("exl3_projection.zig");

test "EXL3 output shards concatenate exactly for decode and prefill" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try ex.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = 256;
    const n = 512;
    const max_rows = 33;
    const su = try device.buffer(k * 2 + 16, mtl.ResourceOptions.shared);
    defer su.deinit();
    const sv = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
    defer sv.deinit();
    const bias = try device.buffer(n * 2 + 16, mtl.ResourceOptions.shared);
    defer bias.deinit();
    const x = try device.buffer(max_rows * k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const full = try device.buffer(max_rows * n * 4, mtl.ResourceOptions.shared);
    defer full.deinit();
    const split = try device.buffer(max_rows * n * 4, mtl.ResourceOptions.shared);
    defer split.deinit();
    const indexed = try device.buffer(max_rows * n * 4, mtl.ResourceOptions.shared);
    defer indexed.deinit();
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    const routes = try device.buffer(max_rows * 4, mtl.ResourceOptions.shared);
    defer routes.deinit();
    @memset(routes.slice(u32, max_rows), 0);
    const scratch = try device.buffer(try ex.scratchBytes(k, n, max_rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    for (x.slice(u16, max_rows * k), 0..) |*v, i| {
        const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 32;
        v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    var rng = std.Random.DefaultPrng.init(0x513a);
    const Case = struct { halves: u16, book: ex.format.Codebook };
    for ([_]Case{ .{ .halves = 3, .book = .mul1 }, .{ .halves = 6, .book = .mul1 }, .{ .halves = 8, .book = .mul1 }, .{ .halves = 6, .book = .mcg }, .{ .halves = 8, .book = .inst3 } }) |case| {
        const halves = case.halves;
        const width = ex.format.Width{ .halves = halves };
        const ns = width.tileWords() * 2 + 2;
        const ks = n / 16 * ns + 16;
        const trellis = try device.buffer(k / 16 * ks + 16, mtl.ResourceOptions.shared);
        defer trellis.deinit();
        rng.random().bytes(trellis.contents()[0..trellis.length()]);
        for ([_]ex.ScaleFormat{ .f16, .packed_signs }) |scale_format| {
            if (scale_format == .f16) {
                for (su.slice(f16, k + 8)[8..], 0..) |*v, i| v.* = if (i % 3 == 0) -0.75 else 1.125;
                for (sv.slice(f16, n + 8)[8..], 0..) |*v, i| v.* = if (i % 5 == 0) -0.25 else 0.875;
            } else {
                rng.random().bytes(su.contents()[16..][0 .. k / 8]);
                rng.random().bytes(sv.contents()[16..][0 .. n / 8]);
            }
            for (bias.slice(f16, n + 8)[8..], 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 16);
            const matrix = ex.Matrix{
                .trellis = .{ .buffer = trellis, .offset = 16 },
                .suh = .{ .buffer = su, .offset = 16 },
                .svh = .{ .buffer = sv, .offset = 16 },
                .bias = .{ .buffer = bias, .offset = 16 },
                .k = k,
                .n = n,
                .width = width,
                .codebook = case.book,
                .suh_format = scale_format,
                .svh_format = scale_format,
                .tile_k_stride = ks,
                .tile_n_stride = ns,
            };
            const descriptor = try ex.descriptor(matrix);
            for (descriptors.slice(ex.ExpertDescriptor, 2), 0..) |*d, rank|
                d.* = try descriptor.columns(n, rank * n / 2, n / 2);
            for ([_]usize{ 1, 9, 33 }) |rows| for ([_]ex.Output{ .bf16, .f32 }) |output| {
                const elem: usize = if (output == .f32) 4 else 2;
                const cb = queue.commandBuffer();
                const encoder = cb.compute(.concurrent);
                const layout = ex.Rows{ .count = rows, .x_stride = k, .y_stride = n, .output = output };
                try ex.encode(&pipelines, encoder, matrix, .{ .buffer = x }, .{ .buffer = full }, .{ .buffer = scratch }, layout);
                for (0..2) |rank| {
                    const shard = try matrix.columns(rank * n / 2, n / 2);
                    try ex.encode(&pipelines, encoder, shard, .{ .buffer = x }, .{ .buffer = split, .offset = rank * n / 2 * elem }, .{ .buffer = scratch }, layout);
                    ex.useMatrix(encoder, matrix);
                    var routed = layout;
                    routed.routes = .{ .ids = .{ .buffer = routes } };
                    try ex.encodeIndexed(&pipelines, encoder, .{ .descriptors = .{ .buffer = descriptors, .offset = rank * @sizeOf(ex.ExpertDescriptor) }, .count = 1, .k = k, .n = n / 2 }, .{ .buffer = x }, .{ .buffer = indexed, .offset = rank * n / 2 * elem }, .{ .buffer = scratch }, routed);
                }
                encoder.end();
                cb.commit();
                cb.wait();
                if (cb.failure() != null) return error.Exl3GpuFailure;
                try std.testing.expectEqualSlices(u8, full.contents()[0 .. rows * n * elem], split.contents()[0 .. rows * n * elem]);
                try std.testing.expectEqualSlices(u8, full.contents()[0 .. rows * n * elem], indexed.contents()[0 .. rows * n * elem]);
            };
        }
    }
}
