//! PLE specializes maintained Flash Next bodies, with explicit F32 convolution weights and kept row tails.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
const cfg = @import("config.zig");
const hc = @import("hc_layout.zig");
const wo = @import("core").weight_only;
pub const names = [_][]const u8{ "qwen4_ple_gate", "qwen4_ple_conv" };
pub fn source(a: std.mem.Allocator, c: *const cfg.Config) ![]u8 {
    try (hc.Geometry{ .hidden = c.hidden, .streams = c.streams }).validate();
    if (c.ple_conv < 1 or c.ngram < 2) return error.Qwen4PleGeometry;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const original = @embedFile("../../../kernels/metal/flashnext/q4_ple_gate_6b4b48a5b2338042.metal");
    const conv = @embedFile("../../../kernels/metal/flashnext/q4_ple_conv_f30f7bbf4be76b4e.metal");
    var gate = try edit.once(mem, try edit.function(original, "custom_kernel_q4_ple_gate_6b4b48a5b2338042_bfloat16_t_bfloat16_t_float_float_float_floatc_bfloat16_t_bfloat16_t"), "custom_kernel_q4_ple_gate_6b4b48a5b2338042_bfloat16_t_bfloat16_t_float_float_float_floatc_bfloat16_t_bfloat16_t", names[0]);
    var body = try edit.once(mem, try edit.function(conv, "custom_kernel_q4_ple_conv_f30f7bbf4be76b4e_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t"), "custom_kernel_q4_ple_conv_f30f7bbf4be76b4e_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t", names[1]);
    for ([_][2][]const u8{
        .{ "constexpr int S = 4;", try std.fmt.allocPrint(mem, "constexpr int S = {d};", .{c.streams}) },
        .{ "constexpr int D = 2560;", try std.fmt.allocPrint(mem, "constexpr int D = {d};", .{c.hidden}) },
        .{ "[[kernel]]", "[[max_total_threads_per_threadgroup(256)]] [[kernel]]" },
    }) |e| {
        gate = try edit.once(mem, gate, e[0], e[1]);
        body = try edit.once(mem, body, e[0], e[1]);
    }
    body = try edit.once(mem, body, "constexpr int TAPS = 4;", try std.fmt.allocPrint(mem, "constexpr int TAPS = {d};", .{c.ple_conv}));
    body = try edit.once(mem, body, "constexpr int DIL = 3;", try std.fmt.allocPrint(mem, "constexpr int DIL = {d};", .{c.ngram}));
    gate = try edit.once(mem, gate, "float(bfloat(gd)) / float(bfloat(metal::precise::sqrt(float(D))))", "float(bfloat(gd)) / metal::precise::sqrt(float(D))");
    // torch SiLU reads the BF16 convolution result and rounds the activation once.
    body = try edit.once(mem, body, "float(bfloat(yb * bsig(yb)))", "float(bfloat(yb >= 0.0f ? yb / (1.0f + metal::precise::exp(-yb)) : (yb * metal::precise::exp(yb)) / (1.0f + metal::precise::exp(yb))))");
    return std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bfloat16_t;\n", try edit.function(original, "inline float bsig("), "\n", gate, "\n", body, "\n" });
}
pub fn Kernels(comptime metal: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: metal.Buffer, offset: usize = 0, len: usize };
        device: metal.Device,
        c: *const cfg.Config,
        pipes: [2]metal.Pipeline = undefined,
        count: usize = 0,
        control: @import("ple_control_source.zig").Control = .production,
        pub fn load(a: std.mem.Allocator, d: metal.Device, c: *const cfg.Config) !Self {
            return loadControl(a, d, c, .production);
        }
        pub fn loadControl(a: std.mem.Allocator, d: metal.Device, c: *const cfg.Config, control: @import("ple_control_source.zig").Control) !Self {
            const production = try source(a, c);
            defer a.free(production);
            const text = try @import("ple_control_source.zig").metal(a, production, control);
            defer a.free(text);
            const lib = try metal.Library.fromSource(d, text, metal.CompileOptions.mlx());
            defer lib.deinit();
            var out: Self = .{ .device = d, .c = c, .control = control };
            errdefer out.deinit();
            for (names, 0..) |name, i| {
                out.pipes[i] = try metal.Pipeline.init(d, lib, name, false);
                out.count += 1;
                if (out.pipes[i].maxThreads() < 256 or out.pipes[i].simdWidth() != 32) return error.Qwen4PleThreads;
            }
            return out;
        }
        /// The owner drains every inference command before releasing the pipelines.
        pub fn deinit(k: *Self) void {
            for (k.pipes[0..k.count]) |pipe| pipe.deinit();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, session: anytype, rows: usize, v: [11]View, tail_stride: usize) !void {
            return k.testEncode(session, rows, v, tail_stride, 0);
        }
        pub fn testEncode(k: *const Self, session: anytype, rows: usize, v: [11]View, tail_stride: usize, repeat: u32) !void {
            const c = k.c;
            const wide = c.wide();
            if (rows == 0 or rows > 128 or rows * wide > std.math.maxInt(i32) or tail_stride % 16 != 0) return error.Qwen4PleGeometry;
            const tail = (c.ple_conv - 1) * c.ngram * wide * 2;
            const needs = [_]usize{ rows * (wide + c.hidden) * 2, rows * wide * 2, wide * 4, wide * 4, wide * 4, rows * wide * 2, rows * wide * 2, (tail + rows * wide * 2), wide * c.ple_conv * 4, tail, (rows - 1) * tail_stride + tail };
            if (session.queue.deviceId() != k.device.id) return error.Qwen4PleContext;
            var ranges: [11]wo.Range = undefined;
            for (v, needs, 0..) |view, n, i| {
                if (view.buffer.deviceId() != k.device.id) return error.Qwen4PleContext;
                if (view.offset > view.buffer.length() or view.len > view.buffer.length() - view.offset or view.len < n) return error.Qwen4PleBuffer;
                const address = try std.math.add(usize, @intCast(view.buffer.gpuAddress()), view.offset);
                if (address == 0 or address % @as(usize, if ((i >= 2 and i <= 4) or i == 8) 4 else 2) != 0) return error.Qwen4PleAlignment;
                _ = try std.math.add(usize, address, view.len);
                ranges[i] = .{ .address = address, .len = view.len };
            }
            for ([_]usize{ 5, 6, 7, 10 }) |i| for (0..11) |j| {
                if (i == j) continue;
                const x = ranges[i];
                const y = ranges[j];
                if (if (x.address <= y.address) y.address - x.address < x.len else x.address - y.address < y.len) return error.Qwen4PleAlias;
            };
            var e = try session.encoder();
            e.setPipeline(k.pipes[0]);
            if (k.control == .repeat) @import("metal_control.zig").bind(e, repeat);
            for (v[0..5], 0..) |view, i| e.setBuffer(view.buffer, view.offset, i);
            e.setBytes(std.mem.asBytes(&c.eps), 5);
            e.setBuffer(v[5].buffer, v[5].offset, 6);
            e.setBuffer(v[6].buffer, v[6].offset, 7);
            e.dispatchGroups(metal.Size.of(c.streams, rows, 1), metal.Size.of(256, 1, 1));
            e.barrier();
            try session.copy(v[7].buffer, v[7].offset, v[9].buffer, v[9].offset, tail);
            try session.copy(v[7].buffer, v[7].offset + tail, v[6].buffer, v[6].offset, rows * wide * 2);
            e = try session.encoder();
            e.setPipeline(k.pipes[1]);
            if (k.control == .repeat) @import("metal_control.zig").bind(e, repeat);
            const bind = [_]View{ v[7], v[8], v[5], v[1], v[1] };
            for (bind, 0..) |view, i| e.setBuffer(view.buffer, view.offset, i);
            e.dispatchThreads(metal.Size.of(wide, rows, 1), metal.Size.of(256, 1, 1));
            e.barrier();
            for (0..rows) |r| try session.copy(v[10].buffer, v[10].offset + r * tail_stride, v[7].buffer, v[7].offset + (r + 1) * wide * 2, tail);
        }
    };
}
