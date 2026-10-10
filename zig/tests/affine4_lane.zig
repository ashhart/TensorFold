//! Prepared CPU affine words encode like direct shared core at every width, on the cooperative and reg kernels.
const std = @import("std");
const mtl = @import("metal");
const adapter = @import("affine4_lane");
const lane = @import("core_lane");
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const n = 96;
    const k = 256;
    const weights = try a.alloc(u16, n * k);
    for (weights, 0..) |*v, i| v.* = adapter.cpu.asBf16(@as(f32, @floatFromInt(@as(i32, @intCast((i * 73 + 11) % 511)) - 255)) / 128);
    var prepared = try adapter.cpu.prepare(a, weights, n, k);
    defer prepared.deinit();
    const order = try lane.RegOrder.init(device);
    defer order.deinit();
    for ([_]usize{ 1, 2, 4 }) |sk| {
        const wrapper = try adapter.Adapter.init(a, device, prepared, sk, null);
        defer wrapper.deinit();
        const reg = try adapter.Adapter.init(a, device, prepared, sk, order);
        defer reg.deinit();
        const packed_words = wrapper.words.slice(u32, prepared.words.len);
        const packed_metadata = wrapper.metadata.slice(u16, 2 * prepared.scales.len);
        for (0..n) |column| for (0..k / 64) |group| {
            const source = column * (k / 8) + group * 8;
            const target = ((column / 32) * (k / 64) + group) * 32 * 8 + (column % 32) * 8;
            if (!std.mem.eql(u32, prepared.words[source..][0..8], packed_words[target..][0..8])) return error.PackedWordsDiffer;
            const index = (group * n + column) * 2;
            if (packed_metadata[index] != prepared.scales[column * (k / 64) + group] or packed_metadata[index + 1] != prepared.biases[column * (k / 64) + group]) return error.PackedMetadataDiffer;
        };
        const layout = lane.Layout{ .n = n, .k = k, .format = .{ .bits = 4, .group = 64 }, .sk = sk, .precompute_sums = true, .cooperative = true };
        const direct = try lane.Projection.init(a, device, layout);
        defer direct.deinit();
        const x = try device.buffer(16 * k * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer x.deinit();
        for (x.slice(u16, 16 * k), 0..) |*v, i| v.* = adapter.cpu.asBf16(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 39) % 255)) - 127)) / 64);
        const actual = try device.buffer(16 * n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer actual.deinit();
        const expected = try device.buffer(16 * n * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        defer expected.deinit();
        for (1..17) |rows| {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            try wrapper.encode(e, .{ .buf = x }, .{ .buf = actual }, @intCast(rows));
            e.barrier();
            try direct.encode(e, .{ .buf = x }, .{ .buf = wrapper.words }, .{ .buf = wrapper.metadata }, .{ .buf = expected }, @intCast(rows));
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailure;
            if (!std.mem.eql(u8, actual.contents()[0 .. rows * n * 2], expected.contents()[0 .. rows * n * 2])) return error.AdapterBytesDiffer;
            const cb2 = queue.commandBuffer();
            const e2 = cb2.compute(.serial);
            try reg.encode(e2, .{ .buf = x }, .{ .buf = actual }, @intCast(rows));
            e2.end();
            cb2.commit();
            cb2.wait();
            if (cb2.failure() != null) return error.GpuFailure;
            if (!std.mem.eql(u8, actual.contents()[0 .. rows * n * 2], expected.contents()[0 .. rows * n * 2])) return error.RegBytesDiffer;
        }
    }
    std.debug.print("affine4 lane: 48 SK/width cases byte-equal to direct shared core on both kernels, no model loaded\n", .{});
}
