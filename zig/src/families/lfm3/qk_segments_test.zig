//! A shared round's qk in one dispatch against lfm3_qk run stream by stream: queries, keys and values byte for byte.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const kernels = @import("kernels.zig");

const QkArgs = kernels.QkArgs;
const Segment = kernels.AttentionSegment;
const shared_memory = mtl.ResourceOptions.shared;

/// Three streams at different lengths in caches of different sizes; rows 1, 4 and 3.
const lens = [_]usize{ 0, 37, 4099 };
const caps = [_]usize{ 16, 64, 4200 };
const rows = [_]usize{ 1, 4, 3 };
const total = 8;
const kv_row = c.kv_heads * c.head_dim * 2;

const Buffers = struct {
    device: mtl.Device,
    list: [48]mtl.Buffer = undefined,
    count: usize = 0,

    fn make(self: *Buffers, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(bytes, shared_memory);
        self.list[self.count] = b;
        self.count += 1;
        return b;
    }

    fn deinit(self: *Buffers) void {
        for (self.list[0..self.count]) |b| b.deinit();
    }
};

/// BF16 noise in [-2, 2): an f32's top half, never a NaN or an infinity.
fn noise(b: mtl.Buffer, words: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    for (b.slice(u16, words)) |*w| w.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 4 - 2)) >> 16);
}

const Caches = struct { keys: [3]mtl.Buffer, values: [3]mtl.Buffer, queries: mtl.Buffer };

fn caches(pool: *Buffers) !Caches {
    var out: Caches = undefined;
    for (0..3) |i| {
        out.keys[i] = try pool.make(caps[i] * kv_row);
        out.values[i] = try pool.make(caps[i] * kv_row);
        noise(out.keys[i], caps[i] * kv_row / 2, 100 + i); // the same old rows in both sets
        noise(out.values[i], caps[i] * kv_row / 2, 200 + i);
    }
    out.queries = try pool.make(total * c.hidden * 2);
    @memset(out.queries.contents()[0 .. total * c.hidden * 2], 0xff);
    return out;
}

test "LFM3: a shared round's qk in one dispatch writes the queries, keys and values lfm3_qk writes stream by stream" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var k = try kernels.load(device);
    defer k.deinit();
    var pool: Buffers = .{ .device = device };
    defer pool.deinit();
    const q = try pool.make(total * c.hidden * 2);
    const kx = try pool.make(total * kv_row);
    const vx = try pool.make(total * kv_row);
    const qn = try pool.make(c.head_dim * 2);
    const kn = try pool.make(c.head_dim * 2);
    noise(q, total * c.hidden, 1);
    noise(kx, total * kv_row / 2, 2);
    noise(vx, total * kv_row / 2, 3);
    noise(qn, c.head_dim, 4);
    noise(kn, c.head_dim, 5);
    const table = try pool.make(3 * @sizeOf(Segment));
    const row_segments = try pool.make(total * 4);
    // A full layer's partial rotation and a sliding layer's whole one.
    for ([_]struct { theta: f32, pairs: u32 }{ .{ .theta = 1e6, .pairs = 16 }, .{ .theta = 1e4, .pairs = 64 } }) |rope| {
        const serial = try caches(&pool);
        const one = try caches(&pool);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        var base: usize = 0;
        e.setPipeline(k.get("lfm3_qk"));
        for (0..3) |i| {
            const args = QkArgs{ .len = @intCast(lens[i]), .cap = @intCast(caps[i]), .eps = c.eps, .log2_theta = std.math.log2(rope.theta), .rotary_pairs = rope.pairs };
            e.setBuffer(q, base * c.hidden * 2, 0);
            e.setBuffer(kx, base * kv_row, 1);
            e.setBuffer(vx, base * kv_row, 2);
            e.setBuffer(qn, 0, 3);
            e.setBuffer(kn, 0, 4);
            e.setBytes(std.mem.asBytes(&args), 5);
            e.setBuffer(serial.queries, base * c.hidden * 2, 6);
            e.setBuffer(serial.keys[i], 0, 7);
            e.setBuffer(serial.values[i], 0, 8);
            e.dispatchGroups(.of(c.query_heads + c.kv_heads, rows[i], 1), .of(32, 1, 1));
            base += rows[i];
        }
        const segments = table.slice(Segment, 3);
        const map = row_segments.slice(u32, total);
        base = 0;
        for (segments, 0..) |*seg, i| {
            seg.* = .{ .keys = one.keys[i].gpuAddress(), .values = one.values[i].gpuAddress(), .base = @intCast(base), .rows = @intCast(rows[i]), .args = .{ .len = @intCast(lens[i]), .cap = @intCast(caps[i]), .scale = 0, .window = 0 } };
            @memset(map[base..][0..rows[i]], @intCast(i));
            base += rows[i];
        }
        const args = QkArgs{ .len = 0, .cap = 0, .eps = c.eps, .log2_theta = std.math.log2(rope.theta), .rotary_pairs = rope.pairs };
        e.setPipeline(k.get("lfm3_qk_segments"));
        for (0..3) |i| {
            e.useResource(one.keys[i], mtl.ResourceUsage.write);
            e.useResource(one.values[i], mtl.ResourceUsage.write);
        }
        for ([_]mtl.Buffer{ q, kx, vx, qn, kn }, 0..) |b, slot| e.setBuffer(b, 0, slot);
        e.setBytes(std.mem.asBytes(&args), 5);
        e.setBuffer(one.queries, 0, 6);
        e.setBuffer(table, 0, 7);
        e.setBuffer(row_segments, 0, 8);
        e.dispatchGroups(.of(c.query_heads + c.kv_heads, total, 1), .of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |_| return error.GpuFailed;
        try std.testing.expectEqualSlices(u8, serial.queries.contents()[0 .. total * c.hidden * 2], one.queries.contents()[0 .. total * c.hidden * 2]);
        for (0..3) |i| {
            try std.testing.expectEqualSlices(u8, serial.keys[i].contents()[0 .. caps[i] * kv_row], one.keys[i].contents()[0 .. caps[i] * kv_row]);
            try std.testing.expectEqualSlices(u8, serial.values[i].contents()[0 .. caps[i] * kv_row], one.values[i].contents()[0 .. caps[i] * kv_row]);
        }
        // The queries were written: an unwritten word stays 0xffff.
        try std.testing.expect(serial.queries.slice(u16, 1)[0] != 0xffff);
    }
}
