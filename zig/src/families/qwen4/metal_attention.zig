//! Native QSA uses the inference command session for compute and token-major cache blits in one ordered buffer.
const std = @import("std");
const qsa = @import("cuda_attention.zig");
const generation = @import("metal_attention_source.zig");
const Plan = @import("attention_plan.zig").Plan;
const wo = @import("core").weight_only;
pub const source = generation.source;
pub const Sizes = struct { bytes: [18]usize, score_stride: usize };
pub fn sizes(g: qsa.Geometry, p: qsa.Call) !Sizes {
    try generation.validate(g);
    const z = try qsa.sizes(g, p);
    const part = try std.math.mul(usize, try std.math.mul(usize, p.rows, g.heads), generation.parts);
    return .{ .score_stride = z.score_stride, .bytes = .{
        z.bytes[0],  1024,                                                                1024,        g.index_dim * 4,                        g.index_dim * 4,                      (3 + g.rotary / 2) * 4,
        z.bytes[7],  try std.math.mul(usize, try std.math.mul(usize, p.rows, g.kv), 512), z.bytes[8],  z.bytes[9],                             z.bytes[10],                          z.bytes[11],
        z.bytes[12], z.bytes[13],                                                         z.bytes[14], try std.math.mul(usize, part, 256 * 4), try std.math.mul(usize, part, 2 * 4), z.bytes[20],
    } };
}
pub fn Kernels(comptime metal: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: metal.Buffer, offset: usize = 0, len: usize };
        pub const names = generation.names;
        device: metal.Device,
        geometry: qsa.Geometry,
        pipes: [6]metal.Pipeline = undefined,
        count: usize = 0,
        pub fn load(a: std.mem.Allocator, device: metal.Device, g: qsa.Geometry) !Self {
            const text = try source(a, g);
            defer a.free(text);
            const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
            defer lib.deinit();
            var out: Self = .{ .device = device, .geometry = g };
            errdefer out.deinit();
            for (names, 0..) |name, i| {
                out.pipes[i] = try metal.Pipeline.init(device, lib, name, false);
                out.count += 1;
                const n: usize = if (i == 3) 1024 else if (i == 1) 128 else 256;
                if (out.pipes[i].maxThreads() < n or out.pipes[i].simdWidth() != 32) return error.Qwen4AttentionThreads;
            }
            return out;
        }
        /// The owner drains its inference session before releasing pipelines referenced by earlier encoders.
        pub fn deinit(k: *Self) void {
            for (k.pipes[0..k.count]) |pipe| pipe.deinit();
            k.* = undefined;
        }
        fn bind(e: metal.ComputeEncoder, views: []const View, indices: []const usize) void {
            for (views, indices) |v, i| e.setBuffer(v.buffer, v.offset, i);
        }
        fn ints(e: metal.ComputeEncoder, values: []const i32, index: usize) void {
            e.setBytes(std.mem.sliceAsBytes(values), index);
        }
        pub fn encode(k: *const Self, session: anytype, p: qsa.Call, v: [18]View) !void {
            const g = k.geometry;
            const z = try sizes(g, p);
            const plan = try Plan.init(g, p);
            // All range, context and alias admission precedes the first encoder or cache write.
            if (session.queue.deviceId() != k.device.id) return error.Qwen4AttentionContext;
            var ranges: [18]wo.Range = undefined;
            for (v, z.bytes, 0..) |view, need, i| {
                if (view.buffer.deviceId() != k.device.id) return error.Qwen4AttentionContext;
                if (view.offset > view.buffer.length() or view.len > view.buffer.length() - view.offset or view.len < need) return error.Qwen4AttentionBuffer;
                const ptr = try std.math.add(usize, @intCast(view.buffer.gpuAddress()), view.offset);
                const alignment: usize = if (i == 0 or i == 17) 2 else if (i >= 6 and i <= 12) 16 else 4;
                if (ptr == 0 or ptr % alignment != 0) return error.Qwen4AttentionAlignment;
                _ = try std.math.add(usize, ptr, view.len);
                ranges[i] = .{ .address = ptr, .len = view.len };
            }
            for (6..18) |i| for (0..i) |j| {
                const x = ranges[i];
                const y = ranges[j];
                if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.Qwen4AttentionAlias;
            };
            var e = try session.encoder();
            e.setPipeline(k.pipes[0]);
            bind(e, &.{ v[0], v[1], v[2], v[3], v[5], v[6], v[7], v[10] }, &.{ 0, 2, 3, 4, 6, 7, 8, 9 });
            ints(e, plan.positions[0..p.rows], 1);
            e.setBytes(std.mem.asBytes(&p.eps), 5);
            e.dispatchGroups(metal.Size.of(1, g.heads + g.kv + g.index_heads, p.rows), metal.Size.of(256, 1, 1));
            e.barrier();
            const kv_row = g.kv * 256 * 2;
            try session.copy(v[8].buffer, v[8].offset + p.start * kv_row, v[7].buffer, v[7].offset, p.rows * kv_row);
            for (0..p.rows) |r| {
                const pack_row = v[0].offset + r * g.projectionWidth() * 2;
                try session.copy(v[9].buffer, v[9].offset + (p.start + r) * kv_row, v[0].buffer, pack_row + (2 * g.heads + g.kv) * 256 * 2, kv_row);
                try session.copy(v[11].buffer, v[11].offset + (p.start + r) * g.index_dim * 2, v[0].buffer, pack_row + (2 * (g.heads + g.kv) * 256 + g.index_heads * g.index_dim) * 2, g.index_dim * 2);
            }
            e = try session.encoder();
            if (plan.pool_count > 0) {
                e.setPipeline(k.pipes[1]);
                bind(e, &.{ v[11], v[4], v[5] }, &.{ 0, 2, 4 });
                const first: i32 = @intCast(plan.first_pool);
                e.setBytes(std.mem.asBytes(&first), 1);
                e.setBytes(std.mem.asBytes(&p.eps), 3);
                e.setBuffer(v[12].buffer, v[12].offset + plan.first_pool * g.index_dim * 2, 5);
                e.dispatchGroups(metal.Size.of(1, plan.pool_count, 1), metal.Size.of(128, 1, 1));
                e.barrier();
            }
            const shape = [_]i32{ @intCast(p.rows), @intCast(z.score_stride) };
            e.setPipeline(k.pipes[2]);
            bind(e, &.{ v[10], v[12], v[13] }, &.{ 0, 2, 5 });
            ints(e, &.{@intCast(p.rows)}, 1);
            ints(e, &.{@intCast(z.score_stride)}, 3);
            ints(e, plan.complete[0..p.rows], 4);
            e.dispatchGroups(metal.Size.of((z.score_stride + 7) / 8, (p.rows + 7) / 8, 1), metal.Size.of(256, 1, 1));
            e.barrier();
            e.setPipeline(k.pipes[3]);
            bind(e, &.{ v[13], v[14] }, &.{ 0, 4 });
            ints(e, &shape, 1);
            ints(e, plan.complete[0..p.rows], 2);
            ints(e, plan.ends[0..p.rows], 3);
            e.dispatchGroups(metal.Size.of(p.rows, 1, 1), metal.Size.of(1024, 1, 1));
            e.barrier();
            e.setPipeline(k.pipes[4]);
            bind(e, &.{ v[6], v[8], v[9], v[14], v[15], v[16] }, &.{ 0, 1, 3, 4, 9, 10 });
            ints(e, &.{ 1, @intCast(g.kv), @intCast(p.capacity), 256 }, 2);
            ints(e, &.{ @intCast(p.rows), @intCast(try g.ids()) }, 5);
            ints(e, plan.keys[0..p.rows], 6);
            ints(e, plan.sparse[0..p.rows], 7);
            const scale: f32 = 0.0625;
            e.setBytes(std.mem.asBytes(&scale), 8);
            e.dispatchGroups(metal.Size.of(g.heads, p.rows, generation.parts), metal.Size.of(256, 1, 1));
            e.barrier();
            e.setPipeline(k.pipes[5]);
            bind(e, &.{ v[15], v[16], v[0], v[17] }, &.{ 0, 1, 2, 3 });
            e.dispatchGroups(metal.Size.of(1, g.heads, p.rows), metal.Size.of(256, 1, 1));
            e.barrier();
        }
    };
}
