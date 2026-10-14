const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv.zig");
const cache = @import("kv_state.zig");
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
    try std.testing.expectEqual(@as(usize, 2732032), plan.mutableBytes());
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
    @memset(partial.contents()[0 .. 4 * 512 * 4], 0);
    @memset(one_shot.contents()[0 .. 4 * 512 * 4], 0);
    @memset(weight.slice(u16, 512), bf(1));
    for (0..3 * 512) |i| {
        values.slice(f32, 3 * 512)[i] = @as(f32, @floatFromInt((i % 512) % 11)) - 5 + @as(f32, @floatFromInt(i / 512));
        gates.slice(f32, 3 * 512)[i] = @as(f32, @floatFromInt(i % 7)) * 0.25;
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try pipes.encodePool(enc, view(values), view(gates), view(partial), view(weight), view(out), .{ .rows = 1, .ratio = 2, .position = 0, .stride = 512, .input_f32 = 1 });
    try pipes.encodePool(enc, view(values).shifted(512 * 4), view(gates).shifted(512 * 4), view(partial), view(weight), view(out), .{ .rows = 2, .ratio = 2, .position = 1, .stride = 512, .input_f32 = 1 });
    try pipes.encodePool(enc, view(values), view(gates), view(one_shot), view(weight), view(out).shifted(512 * 2), .{ .rows = 3, .ratio = 2, .position = 0, .stride = 512, .input_f32 = 1 });
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
    try pipes.encodePool(enc, view(values), view(gates), view(partial), view(weight), view(out), .{ .rows = 2, .width = 2, .ratio = 2, .position = 0, .stride = 2, .input_f32 = 1 });
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
