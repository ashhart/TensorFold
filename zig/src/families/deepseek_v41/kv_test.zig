const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv.zig");
const cache = @import("metal_kv_state.zig");
const View = @import("view.zig").View;
fn bf(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
fn view(b: mtl.Buffer) View {
    return .{ .buffer = b };
}
fn submit(cb: mtl.CommandBuffer, enc: mtl.ComputeEncoder) !void {
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
}

test "cache owners and one million token storage match the released layout" {
    try std.testing.expectEqual(@as(usize, 0), try cache.ownerFor(7, 2));
    try std.testing.expectEqual(@as(usize, 1), try cache.ownerFor(13, 2));
    try std.testing.expectEqual(@as(usize, 2), try cache.ownerFor(19, 2));
    try std.testing.expectEqual(@as(usize, 3), try cache.ownerFor(39, 1));
    try std.testing.expectError(error.NoCompressedCache, cache.ownerFor(0, 0));
    const plan = cache.Plan{ .capacity = 1048576, .layers = 40 };
    try std.testing.expectEqual(@as(usize, 1048576 * 890), plan.packedBytes());
    try std.testing.expectEqual(@as(usize, 2863104), plan.mutableBytes()); // The capture region
    try plan.validate();
    try std.testing.expectError(error.InvalidCacheCapacity, (cache.Plan{ .capacity = 1, .layers = 41 }).validate());
    try std.testing.expectError(error.InvalidCacheLayer, cache.ownerFor(40, 1));
    try std.testing.expectError(error.InvalidCacheOwner, cache.ownerFor(39, 2));
}

test "FP4 E2M1 ties, signs, zero scales and BF16 in-place values" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    const x = try device.buffer(64, mtl.ResourceOptions.shared);
    defer x.deinit();
    const storage = try device.buffer(36, mtl.ResourceOptions.shared);
    defer storage.deinit();
    const values = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, 0.25, 0.75, 1.25, 1.75 };
    for (values, 0..) |v, i| x.slice(u16, 32)[i] = bf(v);
    @memset(x.slice(u16, 32)[16..32], 0);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try pipes.encodePack(enc, view(x), view(storage), .{ .rows = 2, .width = 16, .stride = 16, .packed_stride = 9, .block = 16, .kind = .main_fp4, .inplace = 1 });
    try submit(cb, enc);
    const expected = [_]u8{ 0x10, 0x32, 0x54, 0x76, 0xa9, 0xcb, 0x20, 0x42, 0x38 };
    try std.testing.expectEqualSlices(u8, &expected, storage.contents()[0..9]);
    try std.testing.expectEqual(@as(u8, 1), storage.contents()[17]);
    const rounded = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, 0, 1, 1, 2 };
    for (rounded, 0..) |v, i| try std.testing.expectEqual(bf(v), x.slice(u16, 32)[i]);
}

test "compressor carries a partial group across chunks without changing the pooled latent" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    const values = try device.buffer(3 * 512 * 4, mtl.ResourceOptions.shared);
    defer values.deinit();
    const gates = try device.buffer(3 * 512 * 4, mtl.ResourceOptions.shared);
    defer gates.deinit();
    const partial = try device.buffer(4 * 512 * 4, mtl.ResourceOptions.shared);
    defer partial.deinit();
    const one_shot = try device.buffer(4 * 512 * 4, mtl.ResourceOptions.shared);
    defer one_shot.deinit();
    const weight = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
    defer weight.deinit();
    const out = try device.buffer(2 * 512 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const capture = try device.buffer(8 * 2 * 512 * 4, mtl.ResourceOptions.shared);
    defer capture.deinit();
    @memset(partial.contents()[0 .. 4 * 512 * 4], 0);
    @memset(one_shot.contents()[0 .. 4 * 512 * 4], 0);
    @memset(weight.slice(u16, 512), bf(1));
    for (0..3 * 512) |i| {
        values.slice(f32, 3 * 512)[i] = @as(f32, @floatFromInt((i % 512) % 11)) - 5 + @as(f32, @floatFromInt(i / 512));
        gates.slice(f32, 3 * 512)[i] = @as(f32, @floatFromInt(i % 7)) * 0.25;
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try pipes.encodePool(enc, view(values), view(gates), view(partial), view(weight), view(out), view(capture), .{ .rows = 1, .ratio = 2, .position = 0, .stride = 512, .input_f32 = 1, .capture_rows = 8 });
    try pipes.encodePool(enc, view(values).shifted(512 * 4), view(gates).shifted(512 * 4), view(partial), view(weight), view(out), view(capture), .{ .rows = 2, .ratio = 2, .position = 1, .stride = 512, .input_f32 = 1, .capture_rows = 8 });
    try pipes.encodePool(enc, view(values), view(gates), view(one_shot), view(weight), view(out).shifted(512 * 2), view(capture), .{ .rows = 3, .ratio = 2, .position = 0, .stride = 512, .input_f32 = 1, .capture_rows = 8 });
    try submit(cb, enc);
    try std.testing.expectEqualSlices(u16, out.slice(u16, 1024)[0..512], out.slice(u16, 1024)[512..1024]);
    try std.testing.expectEqualSlices(f32, partial.slice(f32, 2048)[0..512], one_shot.slice(f32, 2048)[0..512]);
    try std.testing.expectEqualSlices(f32, partial.slice(f32, 2048)[1024..1536], one_shot.slice(f32, 2048)[1024..1536]);
}

test "compressor rounds the pooled FP32 latent to BF16 before RMS normalization" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    const values = try device.buffer(4 * 4, mtl.ResourceOptions.shared);
    defer values.deinit();
    const gates = try device.buffer(4 * 4, mtl.ResourceOptions.shared);
    defer gates.deinit();
    const partial = try device.buffer(4 * 2 * 4, mtl.ResourceOptions.shared);
    defer partial.deinit();
    const weight = try device.buffer(2 * 2, mtl.ResourceOptions.shared);
    defer weight.deinit();
    const out = try device.buffer(2 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memcpy(values.slice(f32, 4), &[_]f32{ 1.00390625, 0.5, 1.00390625, 0.5 });
    @memset(gates.slice(f32, 4), 0);
    @memset(partial.contents()[0..32], 0);
    @memset(weight.slice(u16, 2), bf(1));
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    // view(partial) rides in the capture slot: unused here, the args set neither capture_rows nor completed.
    try pipes.encodePool(enc, view(values), view(gates), view(partial), view(weight), view(out), view(partial), .{ .rows = 2, .width = 2, .ratio = 2, .position = 0, .stride = 2, .input_f32 = 1 });
    try submit(cb, enc);
    // 1.00390625 ties to 1 in BF16; normalizing the unrounded value changes the second result.
    try std.testing.expectEqual(bf(1.265625), out.slice(u16, 2)[0]);
    try std.testing.expectEqual(bf(0.6328125), out.slice(u16, 2)[1]);
}

test "ring publish handles chunks beyond the window and transaction rollback restores every mutable byte" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    var state = try cache.State.init(device, .{ .capacity = 1024, .layers = 40 });
    defer state.deinit();
    const chunk = try device.buffer(131 * 528, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    for (0..131) |r| @memset(chunk.contents()[r * 528 .. (r + 1) * 528], @intCast(r));
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try pipes.publishWindow(enc, view(chunk), try state.ring(0), 131, 0);
    try state.append(131);
    try pipes.snapshot(enc, &state);
    try pipes.publishWindow(enc, view(chunk), try state.ring(0), 131, 131);
    try state.append(131);
    try pipes.rollback(enc, &state);
    try submit(cb, enc);
    try std.testing.expectEqual(@as(u32, 131), state.position);
    try std.testing.expectEqualSlices(u8, state.backup.contents()[0..state.plan.mutableBytes()], state.live.contents()[0..state.plan.mutableBytes()]);
    for (3..131) |r| {
        const slot = r % 128;
        for (state.live.contents()[slot * 528 .. (slot + 1) * 528]) |byte|
            try std.testing.expectEqual(@as(u8, @intCast(r)), byte);
    }
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    try pipes.snapshot(enc2, &state);
    try pipes.publishWindow(enc2, view(chunk), try state.ring(0), 131, 131);
    try state.append(131);
    try state.commit();
    try submit(cb2, enc2);
    try std.testing.expectEqual(@as(u32, 262), state.position);
    try std.testing.expect(state.saved_position == null);
}

test "window FP8 quantization includes the RoPE tail and index FP4 uses E8M0 scale bytes" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    const x = try device.buffer(64 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    const storage = try device.buffer(66, mtl.ResourceOptions.shared);
    defer storage.deinit();
    const decoded = try device.buffer(64 * 2, mtl.ResourceOptions.shared);
    defer decoded.deinit();
    @memset(x.slice(u16, 64), 0);
    @memset(x.slice(u16, 64)[32..64], bf(0.3));
    x.slice(u16, 64)[63] = bf(448);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    const args = kv.Pack{ .rows = 1, .width = 64, .stride = 64, .packed_stride = 66, .block = 32, .kind = .window_fp8, .inplace = 1 };
    try pipes.encodePack(enc, view(x), view(storage), args);
    try pipes.encodeUnpack(enc, view(storage), view(decoded), args);
    try submit(cb, enc);
    try std.testing.expectEqual(@as(u8, 127), storage.contents()[65]);
    try std.testing.expectEqual(@as(u8, 126), storage.contents()[63]);
    try std.testing.expectEqual(bf(0.3125), decoded.slice(u16, 64)[62]);
    try std.testing.expectEqualSlices(u16, x.slice(u16, 64), decoded.slice(u16, 64));
    @memset(x.slice(u16, 64)[0..32], bf(6));
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    try pipes.encodePack(enc2, view(x), view(storage), .{ .rows = 1, .width = 32, .stride = 32, .packed_stride = 17, .block = 32, .kind = .index_fp4, .inplace = 1 });
    try submit(cb2, enc2);
    for (storage.contents()[0..16]) |byte| try std.testing.expectEqual(@as(u8, 0x77), byte);
    try std.testing.expectEqual(@as(u8, 127), storage.contents()[16]);
}

test {
    _ = @import("csa2_test.zig");
    _ = @import("csa2_identity_test.zig");
}

test "cache pack rejects undersized borrowed input and output before encoding" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const small = try device.buffer(16, mtl.ResourceOptions.shared);
    defer small.deinit();
    const input = try device.buffer(64, mtl.ResourceOptions.shared);
    defer input.deinit();
    const args = kv.Pack{ .rows = 1, .width = 32, .stride = 32, .packed_stride = 33, .block = 32, .kind = .window_fp8 };
    const pipelines: *const kv.Pipelines = undefined;
    try std.testing.expectError(error.InvalidTensorView, pipelines.encodePack(undefined, view(small), view(input), args));
    try std.testing.expectError(error.InvalidTensorView, pipelines.encodePack(undefined, view(input), view(small), args));
    try std.testing.expectError(error.InvalidTensorView, pipelines.encodePack(undefined, view(input).shifted(1), view(input), args));
}

test "cache shapes reject overflow and nonfinite compressor epsilon" {
    try std.testing.expectError(error.InvalidCachePackShape, (kv.Pack{ .rows = 2, .width = 32, .stride = 32, .packed_stride = 33, .block = 32, .kind = .window_fp8, .first = std.math.maxInt(u32) }).validate());
    try std.testing.expectError(error.InvalidCompressorShape, (kv.Pool{ .rows = 1, .ratio = 1, .position = 0, .stride = 512, .input_f32 = 0, .eps = std.math.nan(f32) }).validate());
}

test "a resumed stream equals a fresh one: rollback then re-append matches a direct append" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var pipes = try kv.Pipelines.load(device);
    defer pipes.deinit();
    var fresh = try cache.State.init(device, .{ .capacity = 1024, .layers = 40 });
    defer fresh.deinit();
    var resumed = try cache.State.init(device, .{ .capacity = 1024, .layers = 40 });
    defer resumed.deinit();
    const chunk = try device.buffer(131 * 528, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    for (0..131) |r| @memset(chunk.contents()[r * 528 .. (r + 1) * 528], @intCast(r));

    // A fresh path: two hundred and sixty-two rows, appended straight through.
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try pipes.publishWindow(enc, view(chunk), try fresh.ring(0), 131, 0);
        try fresh.append(131);
        try pipes.publishWindow(enc, view(chunk), try fresh.ring(0), 131, 131);
        try fresh.append(131);
        try submit(cb, enc);
    }

    // A resumed path: the same rows, with a snapshot, a rollback and a re-append in the middle.
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        try pipes.publishWindow(enc, view(chunk), try resumed.ring(0), 131, 0);
        try resumed.append(131);
        try pipes.snapshot(enc, &resumed);
        try pipes.publishWindow(enc, view(chunk), try resumed.ring(0), 131, 131);
        try resumed.append(131);
        try pipes.rollback(enc, &resumed);
        try submit(cb, enc);
        const cb2 = queue.commandBuffer();
        const enc2 = cb2.compute(.serial);
        try pipes.snapshot(enc2, &resumed);
        try pipes.publishWindow(enc2, view(chunk), try resumed.ring(0), 131, 131);
        try resumed.append(131);
        try resumed.commit();
        try submit(cb2, enc2);
    }

    try std.testing.expectEqual(fresh.position, resumed.position);
    try std.testing.expectEqualSlices(u8, fresh.live.contents()[0..fresh.plan.mutableBytes()], resumed.live.contents()[0..resumed.plan.mutableBytes()]);
}

test "a plan with the drafter's three blocks allocates and rings at layer 42" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var state = try cache.State.init(device, .{ .capacity = 1024, .layers = cache.drafter_layers });
    defer state.deinit();
    const ring = try state.ring(42);
    try std.testing.expect(ring.offset == @as(usize, 42) * cache.window * cache.ring_stride);
    try std.testing.expectError(error.InvalidCacheCapacity, cache.State.init(device, .{ .capacity = 1024, .layers = 41 }));
}

test "the fused window KV glue is bit-identical to rms+rope+pack" {
    const ops = @import("ops.zig");
    const rope = @import("rope.zig");
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var kvp = try kv.Pipelines.load(device);
    defer kvp.deinit();
    var op = try ops.Pipelines.load(device);
    defer op.deinit();
    var rp = try rope.Pipelines.load(device, 32, 64, .{ .theta = 10000 }, .{ .theta = 10000 });
    defer rp.deinit();
    const width: u32 = 512;
    const stride: u32 = 512;
    const packed_stride: u32 = 528;
    const position: u32 = 5;
    for ([_]u32{ 1, 4, 8 }) |rows| {
        const x1 = try device.buffer(rows * width * 2, mtl.ResourceOptions.shared);
        defer x1.deinit();
        const x2 = try device.buffer(rows * width * 2, mtl.ResourceOptions.shared);
        defer x2.deinit();
        const w = try device.buffer(width * 2, mtl.ResourceOptions.shared);
        defer w.deinit();
        const y2 = try device.buffer(rows * width * 2, mtl.ResourceOptions.shared);
        defer y2.deinit();
        const c1 = try device.buffer(rows * packed_stride, mtl.ResourceOptions.shared);
        defer c1.deinit();
        const c2 = try device.buffer(rows * packed_stride, mtl.ResourceOptions.shared);
        defer c2.deinit();
        @memset(c1.contents()[0 .. rows * packed_stride], 0);
        @memset(c2.contents()[0 .. rows * packed_stride], 0);
        for (0..rows * width) |i| {
            const v = @as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) * 0.1875 +
                @as(f32, @floatFromInt(i % 7)) * 0.03125;
            x1.slice(u16, rows * width)[i] = bf(v);
            x2.slice(u16, rows * width)[i] = bf(v);
        }
        for (0..width) |i| w.slice(u16, width)[i] = bf(0.75 + @as(f32, @floatFromInt(i % 13)) * 0.0625);
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        ops.encodeRms(enc, &op, view(x1), view(w), view(x1), .{ .rows = rows, .width = width, .x_stride = stride, .y_stride = stride, .eps = 1e-20 });
        enc.barrier();
        try rp.encode(enc, view(x1), .{ .rows = rows, .heads = 1, .width = width, .rotary = 64, .stride = stride, .position = position, .step = 1, .inverse = 0, .spectrum = .normal });
        enc.barrier();
        try kvp.encodePack(enc, view(x1), view(c1), .{ .rows = rows, .width = width, .stride = stride, .packed_stride = packed_stride, .block = 32, .kind = .window_fp8 });
        try kvp.encodeGlue(enc, view(x2), view(w), view(rp.factors), view(c2), view(y2), .{ .rows = rows, .width = width, .stride = stride, .packed_stride = packed_stride, .block = 32, .position = position, .rotary = 64, .write_y = 1, .eps = 1e-20 });
        try submit(cb, enc);
        try std.testing.expectEqualSlices(u16, x1.slice(u16, rows * width)[0 .. rows * width], y2.slice(u16, rows * width)[0 .. rows * width]);
        try std.testing.expectEqualSlices(u8, c1.contents()[0 .. rows * packed_stride], c2.contents()[0 .. rows * packed_stride]);
    }
}
