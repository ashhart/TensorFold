//! Geometry-checked HC uses the maintained Metal norm, mix and dense activation bodies.
const std = @import("std");
const layout = @import("hc_layout.zig");
const wo = @import("core").weight_only;
const generation = @import("../flashnext/hybrid_hc_sources.zig");
pub const Control = @import("hc_control_source.zig").Control;
pub fn source(a: std.mem.Allocator, g: layout.Geometry) ![]u8 {
    try g.validate();
    return generation.metal(a, @embedFile("../../../kernels/metal/flashnext/q4_hc_norm_none_4e0640a899baccfe.metal"), @embedFile("../../../kernels/metal/prefill/fn_prompt.metal"), g.hidden, g.streams);
}
pub fn Kernels(comptime metal: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: metal.Buffer, offset: usize = 0, len: usize };
        pub const names = [_][]const u8{ "qwen4_hc_sums", "qwen4_hc_normed", "qwen4_hc_mix", "qwen4_dense_act" };
        device: metal.Device,
        pipes: [4]metal.Pipeline = undefined,
        count: usize = 0,
        geometry: layout.Geometry,
        control: Control = .production,
        pub fn load(a: std.mem.Allocator, device: metal.Device, g: layout.Geometry) !Self {
            return loadControl(a, device, g, .production);
        }
        pub fn loadControl(a: std.mem.Allocator, device: metal.Device, g: layout.Geometry, control: Control) !Self {
            const text = try source(a, g);
            defer a.free(text);
            const controlled = try @import("hc_control_source.zig").source(a, text, control);
            defer a.free(controlled);
            const lib = try metal.Library.fromSource(device, controlled, metal.CompileOptions.mlx());
            defer lib.deinit();
            var out: Self = .{ .device = device, .geometry = g, .control = control };
            errdefer out.deinit();
            for (names) |name| {
                const pipe = try metal.Pipeline.init(device, lib, name, false);
                out.pipes[out.count] = pipe;
                out.count += 1;
                if (pipe.maxThreads() < 256 or pipe.simdWidth() != 32) return error.Qwen4HcThreads;
            }
            return out;
        }
        /// The owner drains its queue before releasing pipelines referenced by earlier encoders.
        pub fn deinit(k: *Self) void {
            for (k.pipes[0..k.count]) |pipe| pipe.deinit();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, e: metal.ComputeEncoder, op: layout.Op, p: layout.Call, views: []const View) !void {
            return k.testEncode(e, op, p, views, 0);
        }
        pub fn testEncode(k: *const Self, e: metal.ComputeEncoder, op: layout.Op, p: layout.Call, views: []const View, repeat: u32) !void {
            if (e.deviceId() != k.device.id) return error.Qwen4HcContext;
            if (views.len > 4) return error.Qwen4HcBuffer;
            var ranges: [4]wo.Range = undefined;
            for (views, 0..) |v, i| {
                if (v.buffer.deviceId() != k.device.id) return error.Qwen4HcContext;
                if (v.offset > v.buffer.length() or v.len > v.buffer.length() - v.offset) return error.Qwen4HcBuffer;
                ranges[i] = .{ .address = try std.math.add(usize, @intCast(v.buffer.gpuAddress()), v.offset), .len = v.len };
            }
            try layout.validateMetal(k.geometry, p, op, ranges[0..views.len]);
            e.setPipeline(k.pipes[@backingInt(op)]);
            if (k.control == .repeat) e.setBytes(std.mem.asBytes(&repeat), 31);
            const d = k.geometry.hidden;
            const wide = d * k.geometry.streams;
            switch (op) {
                .sum => {
                    e.setBuffer(views[0].buffer, views[0].offset, 0);
                    e.setBuffer(views[0].buffer, views[0].offset, 1);
                    e.setBuffer(views[1].buffer, views[1].offset, 2);
                    e.dispatchGroups(metal.Size.of(d / 256, p.rows, 1), metal.Size.of(256, 1, 1));
                },
                .norm => {
                    for (views[0..3], 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
                    e.setBytes(std.mem.asBytes(&p.eps), 3);
                    e.setBuffer(views[3].buffer, views[3].offset, 4);
                    e.dispatchThreads(metal.Size.of(wide, p.rows, 1), metal.Size.of(256, 1, 1));
                },
                .mix => {
                    for (views[0..3], 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
                    e.dispatchThreads(metal.Size.of(d, p.rows, 1), metal.Size.of(256, 1, 1));
                },
                .act => {
                    for (views, 0..) |v, i| e.setBuffer(v.buffer, v.offset, i);
                    e.dispatchThreads(metal.Size.of(p.rows * p.width, 1, 1), metal.Size.of(256, 1, 1));
                },
            }
            e.barrier();
        }
    };
}
