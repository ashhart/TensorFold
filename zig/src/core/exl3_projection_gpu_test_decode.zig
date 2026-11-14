const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("exl3_projection.zig");
const decoder = @import("exl3_projection_decode.zig");
const support = @import("exl3_projection_gpu_test_support.zig");
const k = support.k;
const n = support.n;
const rows = support.rows;
const Owned = support.Owned;
const bf16 = support.bf16;
const input = support.input;
const complete = support.complete;
const reference = support.reference;

test "EXL3 long cancellation retains small terms in BF16 projection output" {
    const long_k = 4096;
    const blocks = long_k / 128;
    const device = try mtl.Device.init();
    defer device.deinit();
    var pipelines = try ex.Pipelines.load(device);
    defer pipelines.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const base = try Owned.init(device, 6, .mul1, 0x71717);
    defer base.deinit();
    const block_words = base.matrix.width.tileWords() * (128 / 16) * (n / 16);
    const block_bytes = base.buffers[0].contents()[16..][0 .. block_words * 2];
    const trellis = try device.buffer(blocks * block_bytes.len, mtl.ResourceOptions.shared);
    defer trellis.deinit();
    for (0..blocks) |i| @memcpy(trellis.contents()[i * block_bytes.len ..][0..block_bytes.len], block_bytes);
    const scales = try device.buffer(long_k * 2, mtl.ResourceOptions.shared);
    defer scales.deinit();
    @memset(scales.slice(f16, long_k), 1);
    var matrix = base.matrix;
    matrix.k = long_k;
    matrix.trellis = .{ .buffer = trellis };
    matrix.suh = .{ .buffer = scales };
    matrix.bias = null;
    const x = try device.buffer(long_k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var rotated: [long_k]f16 = undefined;
    for (0..blocks) |i| {
        const v: f32 = if (i == 0) 4096 else if (i == blocks - 1) -4096 else 0x1p-13;
        @memset(x.slice(u16, long_k)[i * 128 ..][0..128], bf16(v));
        var block: [128]f32 = @splat(v);
        decoder.hadamard(&block);
        for (block, 0..) |value, j| rotated[i * 128 + j] = @floatCast(value);
    }
    // Equal large endpoint blocks cancel; intervening terms must survive the long reduction.
    var decoded: [128 * n]f16 = undefined;
    ex.format.unpack(base.matrix.width, base.matrix.codebook, base.buffers[0].slice(i16, block_words + 8)[8..], 128 / 16, n / 16, &decoded);
    var expected: [n]f32 = undefined;
    for (&expected, 0..) |*value, col| {
        var sum: f64 = 0;
        for (rotated, 0..) |v, j| sum += @as(f64, v) * @as(f64, decoded[(j % 128) * n + col]);
        value.* = @floatCast(sum);
    }
    decoder.hadamard(&expected);
    const sv = base.buffers[2].slice(f16, n + 8)[8..];
    var expected_bf16: [n]u16 = undefined;
    for (expected, sv, &expected_bf16) |v, scale, *out| out.* = bf16(v * @as(f32, scale));
    const y = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(long_k, n, 1), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const cb = queue.commandBuffer();
    const encoder = cb.compute(.concurrent);
    try ex.encode(&pipelines, encoder, matrix, .{ .buffer = x }, .{ .buffer = y }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = long_k, .y_stride = n });
    try complete(cb, encoder);
    try std.testing.expectEqualSlices(u16, &expected_bf16, y.slice(u16, n));
}

test "Metal EXL3 codebooks and half widths match the rotated scalar projection" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const x = try input(device, rows);
    defer x.deinit();
    const y = try device.buffer(rows * (n + 8) * 4 + 32, mtl.ResourceOptions.shared);
    defer y.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    for ([_]ex.format.Codebook{ .inst3, .mcg, .mul1 }) |book| {
        for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |halves| {
            if (halves % 2 == 1 and book != .mul1) continue;
            const w = try Owned.init(device, halves, book, 0xdec0de + @as(u64, halves));
            defer w.deinit();
            const cb = queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = y, .offset = 32 }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = k + 16, .y_stride = n + 8, .output = .f32 });
            try complete(cb, e);
            for (0..rows) |r| {
                var want: [n]f32 = undefined;
                reference(w, x.slice(u16, rows * (k + 16) + 16)[16 + r * (k + 16) ..][0..k], &want);
                const got = y.slice(f32, rows * (n + 8) + 8)[8 + r * (n + 8) ..][0..n];
                var mag: f32 = 1;
                for (want) |v| mag = @max(mag, @abs(v));
                for (got, want) |a, b| try std.testing.expectApproxEqAbs(b, a, 0.00005 * mag);
            }
        }
    }
}

test "Metal EXL3 rows retain exact BF16 bits across full and independently resumed windows" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 6, .mul1, 0x71717);
    defer w.deinit();
    const x = try input(device, rows);
    defer x.deinit();
    const all = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer all.deinit();
    const separate = try device.buffer(rows * n * 2, mtl.ResourceOptions.shared);
    defer separate.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, rows), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = all }, .{ .buffer = scratch }, .{ .count = rows, .x_stride = k + 16, .y_stride = n });
    for (0..rows) |r| try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 + r * (k + 16) * 2 }, .{ .buffer = separate, .offset = r * n * 2 }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = k + 16, .y_stride = n });
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, all.slice(u16, rows * n), separate.slice(u16, rows * n));
}

test "GPU router descriptors select mixed-width experts with gather and scatter without host routing" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w0 = try Owned.init(device, 3, .mul1, 11);
    defer w0.deinit();
    const w1 = try Owned.init(device, 16, .mcg, 12);
    defer w1.deinit();
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, 2)[0] = try ex.descriptor(w0.matrix);
    descriptors.slice(ex.ExpertDescriptor, 2)[1] = try ex.descriptor(w1.matrix);
    const ids = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 6), &[_]u32{ 0, 1, 1, 0, 1, 0 });
    const gather = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer gather.deinit();
    @memcpy(gather.slice(u32, 6), &[_]u32{ 2, 0, 1, 2, 1, 0 });
    const scatter = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer scatter.deinit();
    @memcpy(scatter.slice(u32, 6), &[_]u32{ 5, 4, 3, 2, 1, 0 });
    const x = try input(device, 3);
    defer x.deinit();
    const indexed = try device.buffer(6 * n * 4, mtl.ResourceOptions.shared);
    defer indexed.deinit();
    const plain = try device.buffer(6 * n * 4, mtl.ResourceOptions.shared);
    defer plain.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, 6), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, w0.matrix);
    ex.useMatrix(e, w1.matrix);
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = descriptors }, .count = 2, .k = k, .n = n }, .{ .buffer = x, .offset = 32 }, .{ .buffer = indexed }, .{ .buffer = scratch }, .{ .count = 6, .x_stride = k + 16, .y_stride = n, .output = .f32, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 2 }, .input_rows = 3, .output_rows = 6, .input_ids = .{ .buffer = gather }, .output_ids = .{ .buffer = scatter } });
    for (0..6) |slot| {
        const w = if (ids.slice(u32, 6)[slot] == 0) w0 else w1;
        const source = gather.slice(u32, 6)[slot];
        const dest = scatter.slice(u32, 6)[slot];
        try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 + source * (k + 16) * 2 }, .{ .buffer = plain, .offset = dest * n * 4 }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = k + 16, .y_stride = n, .output = .f32 });
    }
    try complete(cb, e);
    try std.testing.expectEqualSlices(u32, plain.slice(u32, 6 * n), indexed.slice(u32, 6 * n));
}

test "packed signs and GPU expert filtering preserve untouched output slots" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const w = try Owned.init(device, 4, .inst3, 0xfeed);
    defer w.deinit();
    const su = try device.buffer(k / 8, mtl.ResourceOptions.shared);
    defer su.deinit();
    const sv = try device.buffer(n / 8, mtl.ResourceOptions.shared);
    defer sv.deinit();
    @memset(su.slice(i16, k / 16), @bitCast(@as(u16, 0x9635)));
    @memset(sv.slice(i16, n / 16), @bitCast(@as(u16, 0x4a17)));
    ex.format.unpackSigns(su.slice(i16, k / 16), w.buffers[1].slice(f16, k + 8)[8..]);
    ex.format.unpackSigns(sv.slice(i16, n / 16), w.buffers[2].slice(f16, n + 8)[8..]);
    var signed = w.matrix;
    signed.suh = .{ .buffer = su };
    signed.svh = .{ .buffer = sv };
    signed.suh_format = .packed_signs;
    signed.svh_format = .packed_signs;
    const ids = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 6), &[_]u32{ 2, 1, 1, 2, 2, 1 });
    const x = try input(device, 3);
    defer x.deinit();
    const filtered = try device.buffer(6 * n * 2, mtl.ResourceOptions.shared);
    defer filtered.deinit();
    const plain = try device.buffer(6 * n * 2, mtl.ResourceOptions.shared);
    defer plain.deinit();
    @memset(filtered.slice(u16, 6 * n), 0x7f7f);
    @memset(plain.slice(u16, 6 * n), 0x7f7f);
    const scratch = try device.buffer(try ex.scratchBytes(k, n, 6), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try ex.encode(&p, e, signed, .{ .buffer = x, .offset = 32 }, .{ .buffer = filtered }, .{ .buffer = scratch }, .{ .count = 6, .x_stride = k + 16, .y_stride = n, .routes = .{ .ids = .{ .buffer = ids }, .expert_id = 1, .repeat = 2 } });
    for (0..6) |slot| {
        if (ids.slice(u32, 6)[slot] != 1) continue;
        try ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 32 + (slot / 2) * (k + 16) * 2 }, .{ .buffer = plain, .offset = slot * n * 2 }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = k + 16, .y_stride = n });
    }
    try complete(cb, e);
    try std.testing.expectEqualSlices(u16, plain.slice(u16, 6 * n), filtered.slice(u16, 6 * n));
}

test "rejected views leave an encoder reusable and successful output retains its guards" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const w = try Owned.init(device, 6, .mul1, 17);
    defer w.deinit();
    const x = try input(device, 1);
    defer x.deinit();
    const y = try device.buffer((n + 16) * 2, mtl.ResourceOptions.shared);
    defer y.deinit();
    @memset(y.slice(u16, n + 16), 0x7f7f);
    const scratch_bytes = try ex.scratchBytes(k, n, 1);
    const scratch = try device.buffer(scratch_bytes + 32, mtl.ResourceOptions.shared);
    defer scratch.deinit();
    @memset(scratch.slice(u8, scratch.length()), 0xa5);
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    const xv = ex.View{ .buffer = x, .offset = 32 };
    const yv = ex.View{ .buffer = y, .offset = 16 };
    const sv = ex.View{ .buffer = scratch, .offset = 16 };
    const r = ex.Rows{ .count = 1, .x_stride = k, .y_stride = n };
    try std.testing.expectError(error.InvalidExl3View, ex.encode(&p, e, w.matrix, .{ .buffer = x, .offset = 33 }, yv, sv, r));
    try std.testing.expectError(error.InvalidExl3View, ex.encode(&p, e, w.matrix, xv, yv, .{ .buffer = scratch, .offset = 48 }, r));
    try std.testing.expectError(error.AliasedExl3View, ex.encode(&p, e, w.matrix, xv, xv, sv, r));
    var bad_rows = r;
    bad_rows.x_stride = std.math.maxInt(usize);
    bad_rows.count = 2;
    try std.testing.expectError(error.Overflow, ex.encode(&p, e, w.matrix, xv, yv, sv, bad_rows));
    bad_rows = r;
    bad_rows.input_ids = .{ .buffer = x };
    try std.testing.expectError(error.InvalidExl3Capacity, ex.encode(&p, e, w.matrix, xv, yv, sv, bad_rows));
    bad_rows.input_rows = 2;
    try std.testing.expectError(error.InvalidExl3View, ex.encode(&p, e, w.matrix, xv, yv, sv, bad_rows));
    var bad_matrix = w.matrix;
    bad_matrix.trellis.offset = bad_matrix.trellis.buffer.length();
    try std.testing.expectError(error.InvalidExl3View, ex.encode(&p, e, bad_matrix, xv, yv, sv, r));
    try ex.encode(&p, e, w.matrix, xv, yv, sv, r);
    try complete(cb, e);
    for (y.slice(u16, n + 16)[0..8]) |guard| try std.testing.expectEqual(@as(u16, 0x7f7f), guard);
    for (y.slice(u16, n + 16)[n + 8 ..]) |guard| try std.testing.expectEqual(@as(u16, 0x7f7f), guard);
    for (scratch.slice(u8, scratch.length())[0..16]) |guard| try std.testing.expectEqual(@as(u8, 0xa5), guard);
    for (scratch.slice(u8, scratch.length())[scratch_bytes + 16 ..]) |guard|
        try std.testing.expectEqual(@as(u8, 0xa5), guard);
    var expected: [n]f32 = undefined;
    reference(w, x.slice(u16, k + 32)[16..][0..k], &expected);
    for (expected, y.slice(u16, n + 16)[8..][0..n]) |want, got|
        try std.testing.expectApproxEqAbs(want, @as(f32, @bitCast(@as(u32, got) << 16)), 0.008 * @max(1, @abs(want)));
}

test "caller-prepared grouped expert tiles preserve scatter slots and ignore padding" {
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try ex.Pipelines.load(device);
    defer p.deinit();
    const w0 = try Owned.init(device, 3, .mul1, 81);
    defer w0.deinit();
    const w1 = try Owned.init(device, 16, .mcg, 82);
    defer w1.deinit();
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    descriptors.slice(ex.ExpertDescriptor, 2)[0] = try ex.descriptor(w0.matrix);
    descriptors.slice(ex.ExpertDescriptor, 2)[1] = try ex.descriptor(w1.matrix);
    const ids = try device.buffer(16 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    const gather = try device.buffer(16 * 4, mtl.ResourceOptions.shared);
    defer gather.deinit();
    const scatter = try device.buffer(16 * 4, mtl.ResourceOptions.shared);
    defer scatter.deinit();
    for (0..16) |slot| {
        const valid = slot < 5 or (slot >= 8 and slot < 14);
        ids.slice(u32, 16)[slot] = if (valid) @intCast(slot / 8) else std.math.maxInt(u32);
        gather.slice(u32, 16)[slot] = if (valid) @intCast(slot % 3) else std.math.maxInt(u32);
        scatter.slice(u32, 16)[slot] = if (valid) @intCast(15 - slot) else std.math.maxInt(u32);
    }
    const x = try input(device, 3);
    defer x.deinit();
    const grouped = try device.buffer(16 * n * 4, mtl.ResourceOptions.shared);
    defer grouped.deinit();
    const plain = try device.buffer(16 * n * 4, mtl.ResourceOptions.shared);
    defer plain.deinit();
    @memset(grouped.slice(u32, 16 * n), 0x7fbebebe);
    @memset(plain.slice(u32, 16 * n), 0x7fbebebe);
    const scratch = try device.buffer(try ex.scratchBytes(k, n, 16), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    ex.useMatrix(e, w0.matrix);
    ex.useMatrix(e, w1.matrix);
    try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = descriptors }, .count = 2, .k = k, .n = n }, .{ .buffer = x, .offset = 32 }, .{ .buffer = grouped }, .{ .buffer = scratch }, .{ .count = 16, .x_stride = k + 16, .y_stride = n, .output = .f32, .grouped = true, .input_ids = .{ .buffer = gather }, .output_ids = .{ .buffer = scatter }, .input_rows = 3, .output_rows = 16, .routes = .{ .ids = .{ .buffer = ids } } });
    for (0..16) |slot| {
        if (ids.slice(u32, 16)[slot] == std.math.maxInt(u32)) continue;
        const matrix = if (slot < 8) w0.matrix else w1.matrix;
        try ex.encode(&p, e, matrix, .{ .buffer = x, .offset = 32 + (slot % 3) * (k + 16) * 2 }, .{ .buffer = plain, .offset = (15 - slot) * n * 4 }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = k + 16, .y_stride = n, .output = .f32 });
    }
    try complete(cb, e);
    try std.testing.expectEqualSlices(u32, plain.slice(u32, 16 * n), grouped.slice(u32, 16 * n));
}

test "the exact half-FMA decode bodies keep every bit of the table bodies for any tile count" {
    const device = try mtl.Device.init();
    defer device.deinit();
    var p = try ex.Pipelines.loadWith(device, std.testing.allocator);
    defer p.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const saved = ex.knobs.fma_tiles;
    const saved_groups = ex.knobs.fma_tile_groups;
    defer ex.knobs.fma_tiles = saved;
    defer ex.knobs.fma_tile_groups = saved_groups;
    ex.knobs.fma_tile_groups = 0;
    const x = try input(device, 3);
    defer x.deinit();
    const scratch = try device.buffer(try ex.scratchBytes(k, n, 6), mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const ref = try device.buffer(7 * n * 4, mtl.ResourceOptions.shared);
    defer ref.deinit();
    const got = try device.buffer(7 * n * 4, mtl.ResourceOptions.shared);
    defer got.deinit();
    const descriptors = try device.buffer(2 * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    const ids = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    @memcpy(ids.slice(u32, 6), &[_]u32{ 0, 1, 1, 0, 1, 0 });
    const gather = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer gather.deinit();
    @memcpy(gather.slice(u32, 6), &[_]u32{ 2, 0, 1, 2, 1, 0 });
    for ([_]u16{ 2, 4, 6, 8, 10, 12 }) |halves| {
        const w0 = try Owned.init(device, halves, .mul1, 0xa0 + @as(u64, halves));
        defer w0.deinit();
        const w1 = try Owned.init(device, halves, .mcg, 0xb0 + @as(u64, halves));
        defer w1.deinit();
        descriptors.slice(ex.ExpertDescriptor, 2)[0] = try ex.descriptor(w0.matrix);
        descriptors.slice(ex.ExpertDescriptor, 2)[1] = try ex.descriptor(w1.matrix);
        // Zero tiles is the table body; every other count writes the same bits, dense and routed.
        for ([_]u32{ 0, 1, 2, 4 }) |tiles| {
            ex.knobs.fma_tiles = tiles;
            const out = if (tiles == 0) ref else got;
            const cb = queue.commandBuffer();
            const e = cb.compute(.concurrent);
            ex.useMatrix(e, w0.matrix);
            ex.useMatrix(e, w1.matrix);
            try ex.encode(&p, e, w0.matrix, .{ .buffer = x, .offset = 32 }, .{ .buffer = out }, .{ .buffer = scratch }, .{ .count = 1, .x_stride = k + 16, .y_stride = n, .output = .f32 });
            try ex.encodeIndexed(&p, e, .{ .descriptors = .{ .buffer = descriptors }, .count = 2, .k = k, .n = n }, .{ .buffer = x, .offset = 32 }, .{ .buffer = out, .offset = n * 4 }, .{ .buffer = scratch }, .{ .count = 6, .x_stride = k + 16, .y_stride = n, .output = .f32, .routes = .{ .ids = .{ .buffer = ids }, .repeat = 2 }, .input_rows = 3, .output_rows = 6, .input_ids = .{ .buffer = gather } });
            try complete(cb, e);
            if (tiles != 0) try std.testing.expectEqualSlices(u32, ref.slice(u32, 7 * n), got.slice(u32, 7 * n));
        }
    }
}
