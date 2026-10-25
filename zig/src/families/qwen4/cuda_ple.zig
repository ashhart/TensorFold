//! Native CUDA lexical gates and dilated convolution publish every row's tail on the owning inference stream.
const std = @import("std");
const cfg = @import("config.zig");
const wo = @import("core").weight_only;
pub fn Kernels(comptime cuda: type) type {
    return struct {
        const Self = @This();
        pub const names = [_][:0]const u8{ "qwen4_ple_gate", "qwen4_ple_conv" };
        pub const View = struct { buffer: *const cuda.DeviceBuffer, offset: usize = 0, len: usize };
        module: cuda.Module,
        f: [2]cuda.Function,
        c: *const cfg.Config,
        controls: ?[8]cuda.Function = null,
        pub fn load(d: *const cuda.Driver, image: []const u8, c: *const cfg.Config) !Self {
            return loadControls(d, image, c, false);
        }
        pub fn loadControls(d: *const cuda.Driver, image: []const u8, c: *const cfg.Config, controls: bool) !Self {
            var m = try cuda.Module.load(d, image);
            errdefer m.unload();
            var functions: [2]cuda.Function = undefined;
            for (names, 0..) |name, i| {
                functions[i] = try m.function(name);
                if (try functions[i].attribute(.max_threads_per_block) < 512) return error.Qwen4PleThreads;
            }
            var out: Self = .{ .module = m, .f = functions, .c = c };
            if (controls) {
                var all: [8]cuda.Function = undefined;
                for (@import("ple_control_source.zig").names, 0..) |name, i| {
                    all[i] = try m.function(name);
                    if (try all[i].attribute(.max_threads_per_block) < 512) return error.Qwen4PleThreads;
                }
                out.controls = all;
            }
            return out;
        }
        pub fn close(k: *Self, s: cuda.Stream) !void {
            if (s.d != k.module.d) return error.Qwen4PleContext;
            try s.synchronize();
            k.module.unload();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, b: anytype, rows: usize, v: [12]View, tail_stride: usize) !void {
            return k.testEncode(b, rows, v, tail_stride, .production, 0);
        }
        pub fn testEncode(k: *const Self, b: anytype, rows: usize, v: [12]View, tail_stride: usize, control: @import("ple_control_source.zig").Control, repeat: u32) !void {
            const c = k.c;
            const wide = c.wide();
            const functions = if (control == .production) k.f else (k.controls orelse return error.Qwen4PleControls)[(@backingInt(control) - 1) * 2 ..][0..2].*;
            const tail_rows = (c.ple_conv - 1) * c.ngram;
            if (rows == 0 or rows > 128 or rows * wide > std.math.maxInt(c_int) or tail_stride % 16 != 0) return error.Qwen4PleGeometry;
            const tail = try std.math.mul(usize, tail_rows, wide * 2);
            const kept = try std.math.add(usize, try std.math.mul(usize, rows - 1, tail_stride), tail);
            const needs = [_]usize{ rows * (wide + c.hidden) * 2, rows * wide * 2, wide * 4, wide * 4, wide * 4, rows * wide * 2, rows * wide * 2, rows * c.streams * 4, wide * c.ple_conv * 4, tail, tail + rows * wide * 2, kept };
            if (b.stream.d != k.module.d) return error.Qwen4PleContext;
            var pointers: [12]u64 = undefined;
            var ranges: [12]wo.Range = undefined;
            for (v, needs, 0..) |view, n, i| {
                if (view.buffer.d != k.module.d) return error.Qwen4PleContext;
                if (view.offset > view.buffer.len or view.len > view.buffer.len - view.offset or view.len < n) return error.Qwen4PleBuffer;
                const address = try std.math.add(usize, @intCast(view.buffer.ptr), view.offset);
                if (address == 0 or address % @as(usize, if ((i >= 2 and i <= 4) or i == 7 or i == 8) 4 else 2) != 0) return error.Qwen4PleAlignment;
                _ = try std.math.add(usize, address, view.len);
                pointers[i] = address;
                ranges[i] = .{ .address = address, .len = view.len };
            }
            for ([_]usize{ 5, 6, 7, 10, 11 }) |i| for (0..12) |j| {
                if (i == j) continue;
                const x = ranges[i];
                const y = ranges[j];
                if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.Qwen4PleAlias;
            };
            var a: cuda.Args = .{};
            for ([_]u64{ pointers[0], pointers[0] + wide * 2, pointers[1], pointers[2], pointers[3], pointers[5], pointers[7] }) |ptr| a.add(ptr);
            a.add(c.eps);
            a.add(@as(c_int, @intCast(c.hidden)));
            a.add(@as(c_int, @intCast(c.streams)));
            if (control != .production) a.add(repeat);
            try cuda.launch.launch(functions[0], .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, b.stream, &a);
            a = .{};
            for ([_]u64{ pointers[5], pointers[7], pointers[4], pointers[9], pointers[8], pointers[1], pointers[1], pointers[6] }) |ptr| a.add(ptr);
            a.add(c.eps);
            for ([_]usize{ c.hidden, c.streams, c.ple_conv, c.ngram }) |n| a.add(@as(c_int, @intCast(n)));
            if (control != .production) a.add(repeat);
            try cuda.launch.launch(functions[1], .{ .grid = .{ .x = @intCast(rows), .y = @intCast((wide + 511) / 512) }, .block = .{ .x = 512 } }, b.stream, &a);
            try b.copy(v[10].buffer.*, v[10].offset, v[9].buffer.*, v[9].offset, tail);
            try b.copy(v[10].buffer.*, v[10].offset + tail, v[6].buffer.*, v[6].offset, rows * wide * 2);
            for (0..rows) |r| try b.copy(v[11].buffer.*, v[11].offset + r * tail_stride, v[10].buffer.*, v[10].offset + (r + 1) * wide * 2, tail);
        }
    };
}
