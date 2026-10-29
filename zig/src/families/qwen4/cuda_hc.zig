//! Qwen4 HC and dense activation use the existing Flash Next CUDA image with geometry-checked arguments.
const std = @import("std");
const layout = @import("hc_layout.zig");
const wo = @import("core").weight_only;
pub fn Kernels(comptime cuda: type) type {
    return struct {
        const Self = @This();
        pub const names = [_][:0]const u8{ "fn_hc_writeback0", "fn_hc_normalize_cols", "fn_hc_dense_mix8", "fn_shared_swiglu" };
        pub const View = struct { buffer: *const cuda.DeviceBuffer, offset: usize = 0, len: usize };
        module: cuda.Module,
        f: [4]cuda.Function = undefined,
        geometry: layout.Geometry,
        dense_bound: bool = false,
        dense_repeat: u32 = 0,
        pub fn load(d: *const cuda.Driver, image: []const u8, geometry: layout.Geometry) !Self {
            try geometry.validate();
            var out: Self = .{ .module = try cuda.Module.load(d, image), .geometry = geometry };
            errdefer out.module.unload();
            for (names, 0..) |name, i| {
                out.f[i] = try out.module.function(name);
                if (try out.f[i].attribute(.max_threads_per_block) < 256) return error.Qwen4HcThreads;
            }
            return out;
        }
        /// The caller retains this function's module until every queued HC use drains.
        pub fn bindDense(k: *Self, function: cuda.Function) !void {
            if (function.d != k.module.d) return error.Qwen4HcContext;
            if (try function.attribute(.max_threads_per_block) < 256) return error.Qwen4HcThreads;
            k.f[3] = function;
            k.dense_bound = true;
        }
        pub fn close(k: *Self, s: cuda.Stream) !void {
            if (s.d != k.module.d) return error.Qwen4HcContext;
            try s.synchronize();
            k.module.unload();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, s: cuda.Stream, op: layout.Op, p: layout.Call, views: []const View) !void {
            if (s.d != k.module.d) return error.Qwen4HcContext;
            if (views.len > 4) return error.Qwen4HcBuffer;
            var ranges: [4]wo.Range = undefined;
            var a: cuda.Args = .{};
            for (views, 0..) |v, i| {
                if (v.buffer.d != s.d) return error.Qwen4HcContext;
                if (v.offset > v.buffer.len or v.len > v.buffer.len - v.offset) return error.Qwen4HcBuffer;
                const address = try std.math.add(usize, @intCast(v.buffer.ptr), v.offset);
                ranges[i] = .{ .address = address, .len = v.len };
                a.add(@as(u64, @intCast(address)));
            }
            try layout.validate(k.geometry, p, op, ranges[0..views.len]);
            const d = k.geometry.hidden;
            const streams = k.geometry.streams;
            var configuration: cuda.Config = .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } };
            switch (op) {
                .sum => {
                    a.add(@as(c_int, @intCast(d)));
                    a.add(@as(c_int, @intCast(streams)));
                    configuration.grid = .{ .x = @intCast(p.rows), .y = @intCast(d / 256) };
                },
                .norm => {
                    a.add(p.eps);
                    a.add(@as(c_int, @intCast(d)));
                    a.add(@as(c_int, @intCast(streams)));
                    configuration.grid = .{ .x = @intCast(p.rows), .y = @intCast(streams), .z = @intCast(d / 256) };
                },
                .mix => {
                    a.add(@as(c_int, @intCast(d)));
                    a.add(@as(c_int, @intCast(streams)));
                    configuration.grid = .{ .x = @intCast(p.rows), .y = @intCast((d / 32 + 7) / 8) };
                },
                .act => {
                    a.add(@as(c_int, @intCast(p.width)));
                    if (k.dense_bound) a.add(k.dense_repeat);
                    configuration.grid = .{ .x = @intCast(p.rows), .y = @intCast((p.width + 255) / 256) };
                },
            }
            try cuda.launch.launch(k.f[@backingInt(op)], configuration, s, &a);
        }
    };
}
