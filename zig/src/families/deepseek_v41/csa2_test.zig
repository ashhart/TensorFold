const std = @import("std");
const mtl = @import("metal");
const csa2 = @import("csa2.zig");
const index = @import("csa2_index.zig");
const rope = @import("rope.zig");
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

test "sparse attention preserves a causal window across ring wrap and masks future compressed keys" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();
    const q = try device.buffer(3 * 512 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const ring = try device.buffer(128 * 528, mtl.ResourceOptions.shared);
    defer ring.deinit();
    const chunk = try device.buffer(3 * 528, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    const main = try device.buffer(201 * 288, mtl.ResourceOptions.shared);
    defer main.deinit();
    const positions = try device.buffer(6 * 4, mtl.ResourceOptions.shared);
    defer positions.deinit();
    const sink = try device.buffer(4, mtl.ResourceOptions.shared);
    defer sink.deinit();
    const out = try device.buffer(3 * 512 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(q.contents()[0 .. 3 * 512 * 2], 0);
    sink.slice(f32, 1)[0] = 0;
    for (0..128) |r| {
        @memset(ring.contents()[r * 528 .. r * 528 + 512], 0x38);
        @memset(ring.contents()[r * 528 + 512 .. (r + 1) * 528], 127);
    }
    for (0..3) |r| {
        @memset(chunk.contents()[r * 528 .. r * 528 + 512], 0x40);
        @memset(chunk.contents()[r * 528 + 512 .. (r + 1) * 528], 127);
        positions.slice(i32, 6)[2 * r] = 0;
        positions.slice(i32, 6)[2 * r + 1] = 200;
    }
    @memset(main.contents()[0..256], 0x66);
    @memset(main.contents()[256..288], 0x38);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q), view(ring), view(chunk), view(main), view(positions), view(sink), view(out), .{ .rows = 3, .heads = 1, .q_stride = 512, .position = 129, .chunk_rows = 3, .ratio = 2, .compressed_rows = 201, .topk = 2, .index_stride = 2 });
    try submit(cb, enc);
    for (0..3) |r| {
        const expected = bf((133 + @as(f32, @floatFromInt(r))) / 130);
        for (out.slice(u16, 3 * 512)[r * 512 .. (r + 1) * 512]) |v| try std.testing.expectEqual(expected, v);
    }
}

test "sparse QK cancellation retains a small score before sink normalization" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();
    const q = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const chunk = try device.buffer(528, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    const sink = try device.buffer(4, mtl.ResourceOptions.shared);
    defer sink.deinit();
    const out = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(q.slice(u16, 512), 0);
    q.slice(u16, 512)[0] = bf(0x1p24);
    q.slice(u16, 512)[1] = bf(-0x1p24);
    q.slice(u16, 512)[32] = bf(0x1p-6);
    @memset(chunk.contents()[0..512], 0x38);
    @memset(chunk.contents()[512..528], 127);
    sink.slice(f32, 1)[0] = 0;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    // No prior window or compressed keys: their unused bindings share the immutable chunk.
    try p.encode(enc, view(q), view(chunk), view(chunk), view(chunk), view(chunk), view(sink), view(out), .{ .rows = 1, .heads = 1, .q_stride = 512, .position = 0, .chunk_rows = 1, .ratio = 0, .compressed_rows = 0, .scale = 1 });
    try submit(cb, enc);
    // Exact dot = 2^-6; 1 / (1 + exp(-2^-6)) rounds to this BF16 value.
    for (out.slice(u16, 512)) |v| try std.testing.expectEqual(bf(0.50390625), v);
}

test "the first query retains BF16 probability arithmetic regardless of future prefill rows" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();
    const q = try device.buffer(32 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const ring = try device.buffer(128 * 33, mtl.ResourceOptions.shared);
    defer ring.deinit();
    const chunk = try device.buffer(128 * 33, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    const main = try device.buffer(18, mtl.ResourceOptions.shared);
    defer main.deinit();
    const positions = try device.buffer(4, mtl.ResourceOptions.shared);
    defer positions.deinit();
    const sink = try device.buffer(4, mtl.ResourceOptions.shared);
    defer sink.deinit();
    const out = try device.buffer(32 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(q.slice(u16, 32), 0);
    q.slice(u16, 32)[0] = bf(1);
    @memset(chunk.contents()[0..32], 0x38);
    chunk.contents()[32] = 127;
    @memset(main.contents()[0..16], 0x44);
    @memset(main.contents()[16..18], 0x38);
    positions.slice(i32, 1)[0] = 0;
    sink.slice(f32, 1)[0] = 0.005;
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q), view(ring), view(chunk), view(main), view(positions), view(sink), view(out), .{ .rows = 1, .heads = 1, .width = 32, .q_stride = 32, .position = 0, .chunk_rows = 1, .ring_stride = 33, .chunk_stride = 33, .main_stride = 18, .ratio = 1, .compressed_rows = 1, .topk = 1, .index_stride = 1, .scale = 1 });
    try submit(cb, enc);
    // BF16(exp(-1))*1 + 1*2; padded windows would move key 2 into a later online-softmax tile.
    for (out.slice(u16, 32)) |v| try std.testing.expectEqual(bf(1.5703125), v);
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    try p.encode(enc2, view(q), view(ring), view(chunk), view(main), view(positions), view(sink), view(out), .{ .rows = 1, .heads = 1, .width = 32, .q_stride = 32, .position = 0, .chunk_rows = 128, .ring_stride = 33, .chunk_stride = 33, .main_stride = 18, .ratio = 1, .compressed_rows = 1, .topk = 1, .index_stride = 1, .scale = 1 });
    try submit(cb2, enc2);
    for (out.slice(u16, 32)) |v| try std.testing.expectEqual(bf(1.5703125), v);
}

test "hierarchical candidate selection pins the newest partial block and subsequent reindex uses those blocks" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try index.Pipelines.load(device);
    defer p.deinit();
    var scratch = try index.Scratch.init(device, 1);
    defer scratch.deinit();
    const q = try device.buffer(32 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const weights = try device.buffer(2, mtl.ResourceOptions.shared);
    defer weights.deinit();
    const keys_count = 2048 * 8 + 3;
    const keys = try device.buffer(keys_count * 17, mtl.ResourceOptions.shared);
    defer keys.deinit();
    @memset(q.slice(u16, 32), bf(1));
    weights.slice(u16, 1)[0] = bf(1);
    for (0..keys_count) |r| {
        @memset(keys.contents()[r * 17 .. r * 17 + 16], 0x22);
        keys.contents()[r * 17 + 16] = if (r >= 2048 * 8) 126 else 127;
    }
    var args = index.Params{ .rows = 1, .heads = 1, .width = 32, .q_stride = 32, .weight_stride = 1, .position = keys_count - 1, .ratio = 1, .keys = keys_count, .key_stride = 17, .make_candidates = 1, .scale = 1 };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q), view(weights), view(keys), scratch, args);
    try submit(cb, enc);
    const Pick = extern struct { score: f32, position: i32 };
    const candidates = @as([*]Pick, @ptrCast(@alignCast(scratch.candidates().buffer.contents() + scratch.candidates().offset)))[0..2048];
    try std.testing.expectEqual(@as(i32, 2048), candidates[2047].position);
    for (candidates[0..2047], 0..) |pick, i| try std.testing.expectEqual(@as(i32, @intCast(i)), pick.position);
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    args.make_candidates = 0;
    args.use_candidates = 1;
    try p.encode(enc2, view(q), view(weights), view(keys), scratch, args);
    try submit(cb2, enc2);
    const selected = @as([*]i32, @ptrCast(@alignCast(scratch.positions().buffer.contents() + scratch.positions().offset)))[0..512];
    for (selected, 0..) |pos, i| try std.testing.expectEqual(@as(i32, @intCast(i)), pos);
}

test "RoPE rotates adjacent tail pairs with BF16 assignment and compressed positions use group starts" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try rope.Pipelines.load(device, 3, 32, .{ .theta = 10000 }, .{ .theta = 160000 });
    defer p.deinit();
    const x = try device.buffer(2 * 64 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memset(x.slice(u16, 128), bf(1));
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try std.testing.expectError(error.InvalidRotaryShape, p.encode(enc, view(x), .{ .rows = 2, .heads = 1, .width = 64, .rotary = 32, .stride = 64, .position = 1, .step = 2 }));
    try std.testing.expectError(error.InvalidRotaryShape, p.encode(enc, view(x), .{ .rows = 1, .heads = 1, .width = 64, .rotary = 16, .stride = 64, .position = 0 }));
    try std.testing.expectError(error.InvalidRotaryShape, p.encode(enc, view(x), .{ .rows = 2, .heads = 1, .width = 64, .rotary = 32, .stride = 64, .position = std.math.maxInt(u32), .step = std.math.maxInt(u32) }));
    try p.encode(enc, view(x), .{ .rows = 2, .heads = 1, .width = 64, .rotary = 32, .stride = 64, .position = 0, .step = 2 });
    try submit(cb, enc);
    for (x.slice(u16, 128)[0..96]) |v| try std.testing.expectEqual(bf(1), v);
    for (0..16) |pair| {
        const power: f32 = @floatCast(std.math.pow(f64, 10000, @as(f32, @floatFromInt(pair * 2)) / 32));
        const frequency: f32 = 1 / power;
        const angle: f32 = 2 * frequency;
        const c: f32 = @floatCast(@cos(@as(f64, angle)));
        const s: f32 = @floatCast(@sin(@as(f64, angle)));
        const a = bf(c - s);
        const b = bf(s + c);
        try std.testing.expectEqual(a, x.slice(u16, 128)[96 + pair * 2]);
        try std.testing.expectEqual(b, x.slice(u16, 128)[97 + pair * 2]);
    }
}

test "indexer visibility follows completed compressor groups for every query in a chunk" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try index.Pipelines.load(device);
    defer p.deinit();
    var scratch = try index.Scratch.init(device, 3);
    defer scratch.deinit();
    const q = try device.buffer(3 * 32 * 2, mtl.ResourceOptions.shared);
    defer q.deinit();
    const weights = try device.buffer(3 * 2, mtl.ResourceOptions.shared);
    defer weights.deinit();
    const keys = try device.buffer(3 * 17, mtl.ResourceOptions.shared);
    defer keys.deinit();
    @memset(q.slice(u16, 3 * 32), bf(1));
    @memset(weights.slice(u16, 3), bf(1));
    for (0..3) |r| {
        @memset(keys.contents()[r * 17 .. r * 17 + 16], 0x22);
        keys.contents()[r * 17 + 16] = 127;
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q), view(weights), view(keys), scratch, .{ .rows = 3, .heads = 1, .width = 32, .q_stride = 32, .weight_stride = 1, .position = 0, .ratio = 2, .keys = 3, .key_stride = 17, .scale = 1 });
    try submit(cb, enc);
    const selected = @as([*]i32, @ptrCast(@alignCast(scratch.positions().buffer.contents() +
        scratch.positions().offset)))[0 .. 3 * 512];
    for (selected[0..512]) |pos| try std.testing.expectEqual(@as(i32, -1), pos);
    for (1..3) |r| {
        try std.testing.expectEqual(@as(i32, 0), selected[r * 512]);
        for (selected[r * 512 + 1 .. (r + 1) * 512]) |pos| try std.testing.expectEqual(@as(i32, -1), pos);
    }
}

test "YaRN interpolates its correction band and inverse rotation conjugates the frequency" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try rope.Pipelines.load(device, 65537, 32, .{ .theta = 10000 }, .{ .theta = 160000, .original = 65536 });
    defer p.deinit();
    const x = try device.buffer(64 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memset(x.slice(u16, 64), bf(1));
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(x), .{ .rows = 1, .heads = 1, .width = 64, .rotary = 32, .stride = 64, .position = 65536, .inverse = 1, .spectrum = .compressed });
    try submit(cb, enc);
    const low: f32 = @floatCast(@max(@floor(32 * @log(@as(f64, 65536) / (32 * 2 * std.math.pi)) / (2 * @log(@as(f64, 160000)))), 0));
    const high: f32 = @floatCast(@min(@ceil(32 * @log(@as(f64, 65536) / (2 * std.math.pi)) / (2 * @log(@as(f64, 160000)))), 31));
    for ([_]usize{ 0, 10, 15 }) |pair| {
        const power: f32 = @floatCast(std.math.pow(f64, 160000, @as(f32, @floatFromInt(pair * 2)) / 32));
        const frequency: f32 = 1 / power;
        const smooth: f32 = 1 - std.math.clamp((@as(f32, @floatFromInt(pair)) - low) / @max(high - low, 1e-3), 0, 1);
        const slow: f32 = frequency / 16 * (1 - smooth);
        const fast: f32 = frequency * smooth;
        const angle: f32 = 65536 * (slow + fast);
        const c: f32 = @floatCast(@cos(@as(f64, angle)));
        const s: f32 = -@as(f32, @floatCast(@sin(@as(f64, angle))));
        try std.testing.expectEqual(bf(c - s), x.slice(u16, 64)[32 + pair * 2]);
        try std.testing.expectEqual(bf(s + c), x.slice(u16, 64)[33 + pair * 2]);
    }
}

test "RoPE retained source pair rounds the position-five imaginary consumer to pinned CUDA BF16" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try rope.Pipelines.load(device, 6, 64, .{ .theta = 10000 }, .{ .theta = 160000, .original = 65536 });
    defer p.deinit();
    const x = try device.buffer(512 * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memset(x.slice(u16, 512), bf(0));
    // Token-5/layer-0 rotary pair 21 crosses a BF16 tie under per-head cos; pinned source factors give .546875.
    x.slice(u16, 512)[490] = bf(1.15625);
    x.slice(u16, 512)[491] = bf(0.53125);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(x), .{ .rows = 1, .heads = 1, .width = 512, .rotary = 64, .stride = 512, .position = 5 });
    try submit(cb, enc);
    try std.testing.expectEqual(bf(0.546875), x.slice(u16, 512)[491]);
}
