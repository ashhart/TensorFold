//! The production launches replaced by each chain, executed on the same guarded inputs and activation types.
const std = @import("std");
const mtl = @import("metal");
const source = @import("exl3_chain_source.zig");
const sources = @import("kernel_sources");
const Ref = @import("rig.zig").Ref;

fn bind(e: mtl.ComputeEncoder, r: Ref, extra: usize, at: usize) void {
    e.setBuffer(r.buf, r.off + extra, at);
}

pub fn For(comptime ex: type) type {
    return struct {
        const Self = @This();
        rotate_in: mtl.Pipeline,
        rotate_out: mtl.Pipeline,
        rms: mtl.Pipeline,
        swiglu: mtl.Pipeline,
        rope: mtl.Pipeline,
        glue: mtl.Pipeline,
        combine: mtl.Pipeline,

        pub fn load(device: mtl.Device) !Self {
            const core = try source.library(device, sources.core_exl3_projection);
            defer core.deinit();
            const ops = try source.library(device, sources.deepseek_v41_ops);
            defer ops.deinit();
            const rope_lib = try source.library(device, sources.deepseek_v41_rope);
            defer rope_lib.deinit();
            const glue_lib = try source.library(device, sources.deepseek_v41_kv_glue);
            defer glue_lib.deinit();
            var made: std.ArrayList(mtl.Pipeline) = .empty;
            defer made.deinit(std.heap.page_allocator);
            errdefer for (made.items) |p| p.deinit();
            const libs = [_]mtl.Library{ core, core, ops, ops, rope_lib, glue_lib, ops };
            const names = [_][:0]const u8{ "tf_exl3_rotate_in", "tf_exl3_rotate_out", "dsv41_rms", "dsv41_swiglu", "dsv41_rope", "dsv41_kv_glue", "dsv41_combine" };
            for (libs, names) |lib, name| {
                const p = try mtl.Pipeline.init(device, lib, name, false);
                made.append(std.heap.page_allocator, p) catch |err| {
                    p.deinit();
                    return err;
                };
            }
            return .{ .rotate_in = made.items[0], .rotate_out = made.items[1], .rms = made.items[2], .swiglu = made.items[3], .rope = made.items[4], .glue = made.items[5], .combine = made.items[6] };
        }

        pub fn deinit(s: *Self) void {
            for ([_]mtl.Pipeline{ s.rotate_in, s.rotate_out, s.rms, s.swiglu, s.rope, s.glue, s.combine }) |p| p.deinit();
        }

        fn out(s: *const Self, p: anytype, e: mtl.ComputeEncoder, a: ex.Args, sums_offset: usize, routes: Ref, routes_offset: usize, y: Ref, y_offset: usize) void {
            e.setPipeline(s.rotate_out);
            bind(e, p.sums, sums_offset, 0);
            bind(e, p.sv, 0, 1);
            bind(e, p.bias, 0, 2);
            bind(e, y, y_offset, 3);
            e.setValue(a, 4);
            bind(e, p.destinations, 0, 5);
            bind(e, routes, routes_offset, 6);
            bind(e, p.descriptors, 0, 7);
            e.dispatchGroups(.{ .width = (@as(usize, a.n) + 511) / 512, .height = a.rows }, .{ .width = 128 });
            e.barrier();
        }

        fn input(s: *const Self, p: anytype, e: mtl.ComputeEncoder, x: Ref, x_offset: usize, rotated_offset: usize, n: usize, stride: usize, indexed: bool, route_offset: usize) void {
            var a = p.args(n);
            a.x_stride = stride;
            a.indexed = @intFromBool(indexed);
            e.setPipeline(s.rotate_in);
            bind(e, x, x_offset, 0);
            bind(e, p.su, 0, 1);
            bind(e, p.rotated, rotated_offset, 2);
            e.setValue(a, 3);
            bind(e, p.ids, 0, 4);
            bind(e, p.routes, route_offset, 5);
            bind(e, p.descriptors, 0, 6);
            e.dispatchGroups(.{ .width = (@as(usize, a.k) + 511) / 512, .height = n }, .{ .width = 128 });
            e.barrier();
        }

        fn norm(s: *const Self, p: anytype, e: mtl.ComputeEncoder, x: Ref, x_offset: usize, y_offset: usize, rows: usize, stride: usize) void {
            const f = p.f.?;
            e.setPipeline(s.rms);
            bind(e, x, x_offset, 0);
            bind(e, p.norm_weight, 0, 1);
            bind(e, if (p.kind == .rms) p.work else p.y, y_offset, 2);
            e.setValue(extern struct { rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32, weight_f32: u32 }{ .rows = @intCast(rows), .width = @intCast(f.shape.k), .x_stride = @intCast(stride), .y_stride = @intCast(if (p.kind == .rms) f.stride else f.shape.k), .eps = f.eps, .weight_f32 = 1 }, 3);
            e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
            e.barrier();
        }

        fn rotate(s: *const Self, p: anytype, e: mtl.ComputeEncoder, x: Ref, rows: usize, stride: usize) void {
            const f = p.f.?;
            for (0..rows) |row| {
                e.setPipeline(s.rope);
                bind(e, x, row * stride * 2, 0);
                e.setValue([9]u32{ 1, @intCast(f.shape.k / f.head_width), @intCast(f.head_width), @intCast(f.rotary), @intCast(stride), @intCast(f.factorOffset(row) / f.rotary), 1, @intFromBool(f.inverse), 0 }, 1);
                bind(e, p.factors, 0, 2);
                e.dispatchThreads(.{ .width = f.rotary / 2, .height = f.shape.k / f.head_width }, .{ .width = 32 });
            }
            e.barrier();
        }

        fn activate(s: *const Self, p: anytype, e: mtl.ComputeEncoder, gate: Ref, up: Ref, offset_g: usize, offset_u: usize, output: Ref, output_offset: usize, rows: usize, stride: usize, weighted: bool) void {
            const k = p.f.?.shape.k;
            e.setPipeline(s.swiglu);
            bind(e, gate, offset_g, 0);
            bind(e, up, offset_u, 1);
            bind(e, p.weights, 0, 2);
            bind(e, output, output_offset, 3);
            e.setValue([6]u32{ @intCast(rows), @intCast(k), @intCast(stride), @intCast(k), @intFromBool(weighted), 0 }, 4);
            e.dispatchThreads(.{ .width = k, .height = rows }, .{ .width = 256 });
            e.barrier();
        }

        pub fn encode(s: *const Self, p: anytype, e: mtl.ComputeEncoder) void {
            const f = p.f.?;
            const k = f.shape.k;
            const n = f.shape.rows;
            e.useResource(p.su.buf, mtl.ResourceUsage.read);
            e.useResource(p.sv.buf, mtl.ResourceUsage.read);
            e.useResource(p.bias.buf, mtl.ResourceUsage.read);
            switch (p.kind) {
                .swiglu => {
                    s.activate(p, e, p.x, p.up, 0, 0, p.activation, 0, n, f.stride, f.weighted);
                    for (0..n) |row| s.input(p, e, p.activation, row * k * 2, @as(usize, f.destinations[row]) * k * 2, 1, k, true, row * 4);
                },
                .out_in => {
                    var a = p.args(n);
                    a.indexed = 1;
                    a.output_ids = 1;
                    s.out(p, e, a, 0, p.routes, 0, p.y, 0);
                    var next = p.args(1);
                    next.k = @intCast(n * k);
                    next.x_stride = n * k;
                    e.setPipeline(s.rotate_in);
                    bind(e, p.y, 0, 0);
                    bind(e, p.su, 0, 1);
                    bind(e, p.rotated, 0, 2);
                    e.setValue(next, 3);
                    bind(e, p.ids, 0, 4);
                    bind(e, p.routes, 0, 5);
                    bind(e, p.descriptors, 0, 6);
                    e.dispatchGroups(.{ .width = (n * k + 511) / 512 }, .{ .width = 128 });
                    e.barrier();
                },
                .rms => {
                    s.norm(p, e, p.work, 0, 0, n, f.stride);
                    s.input(p, e, p.work, 0, 0, n, f.stride, false, 0);
                },
                .out_rope => {
                    s.out(p, e, p.args(n), 0, p.routes, 0, p.y, 0);
                    s.rotate(p, e, p.y, n, k);
                },
                .rope_in => {
                    s.rotate(p, e, p.work, n, f.stride);
                    s.input(p, e, p.work, 0, 0, n, f.stride, true, 0);
                },
                .gate_up => {
                    var a = p.args(14 * n);
                    a.indexed = 1;
                    s.out(p, e, a, 0, p.ids, 0, p.activation, 0);
                    s.activate(p, e, p.activation, p.activation, 0, 6 * n * k * 2, p.y, 0, 6 * n, k, f.weighted);
                    s.activate(p, e, p.activation, p.activation, 12 * n * k * 2, 13 * n * k * 2, p.y, 6 * n * k * 2, n, k, false);
                    s.input(p, e, p.y, 0, 0, 6 * n, k, true, 0);
                    const route_bytes = p.rig.bytes(p.route_work, f.routes.len * 4);
                    for (0..n) |row| std.mem.writeInt(u32, route_bytes[(6 * n + row) * 4 ..][0..4], @intCast(f.expert_count), .little);
                    for (0..n) |row| {
                        var shared = p.args(1);
                        shared.x_stride = k;
                        e.setPipeline(s.rotate_in);
                        bind(e, p.y, (6 * n + row) * k * 2, 0);
                        bind(e, p.su, f.expert_count * k * 2, 1);
                        bind(e, p.rotated, (6 * n + row) * k * 2, 2);
                        e.setValue(shared, 3);
                        bind(e, p.ids, 0, 4);
                        bind(e, p.routes, 0, 5);
                        bind(e, p.descriptors, 0, 6);
                        e.dispatchGroups(.{ .width = (k + 511) / 512 }, .{ .width = 128 });
                        e.barrier();
                    }
                },
                .down => {
                    const route_bytes = p.rig.bytes(p.route_work, f.routes.len * 4);
                    for (0..n) |row| std.mem.writeInt(u32, route_bytes[(6 * n + row) * 4 ..][0..4], @intCast(f.expert_count), .little);
                    var a = p.args(7 * n);
                    a.indexed = 1;
                    s.out(p, e, a, 0, p.route_work, 0, p.activation, 0);
                    e.setPipeline(s.combine);
                    bind(e, p.activation, 0, 0);
                    bind(e, p.activation, 6 * n * k * 2, 1);
                    bind(e, p.routes, 0, 2);
                    bind(e, p.y, 0, 3);
                    e.setValue([8]u32{ @intCast(n), @intCast(k), @intCast(6 * k), @intCast(k), @intCast(k), @intCast(k), 0, 6 }, 4);
                    e.dispatchThreads(.{ .width = k, .height = n }, .{ .width = 256 });
                },
                .qkv => for (0..n) |row| {
                    var a = p.args(2);
                    a.indexed = 1;
                    s.out(p, e, a, 2 * row * k * 4, p.ids, 0, p.y, 2 * row * k * 2);
                    s.norm(p, e, p.y, 2 * row * k * 2, 2 * row * k * 2, 1, k);
                    s.input(p, e, p.y, 2 * row * k * 2, row * k * 2, 1, k, false, 0);
                    e.setPipeline(s.glue);
                    bind(e, p.y, (2 * row + 1) * k * 2, 0);
                    bind(e, p.kv_weight, 0, 1);
                    bind(e, p.factors, 0, 2);
                    bind(e, p.key_pack, row * 528, 3);
                    bind(e, p.key_pack, row * 528, 4);
                    e.setValue(extern struct { rows: u32, width: u32, stride: u32, packed_stride: u32, block: u32, position: u32, rotary: u32, write_y: u32, eps: f32 }{ .rows = 1, .width = 512, .stride = @intCast(k), .packed_stride = 528, .block = 32, .position = @intCast(f.position + row), .rotary = @intCast(f.rotary), .write_y = 0, .eps = f.eps }, 5);
                    e.dispatchGroups(.{ .width = 16 }, .{ .width = 256 });
                    e.barrier();
                },
            }
        }
    };
}
