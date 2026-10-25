//! Native CUDA branch and output gates share explicit policy parameters and drain-owned module lifetime.
const std = @import("std");
pub fn Adapter(comptime h: type, comptime cuda: type) type {
    return struct {
        pub const Control = enum(u2) { production, nan, repeat, tail };
        pub const names = [_][:0]const u8{ "qwen4_hc_pointwise", "qwen4_hc_pointwise_bad_nan", "qwen4_hc_pointwise_bad_repeat", "qwen4_hc_pointwise_bad_tail" };
        pub const View = struct { buffer: *const cuda.DeviceBuffer, offset: usize = 0, len: usize };
        pub const Kernels = struct {
            module: cuda.Module,
            f: [4]cuda.Function = undefined,
            controls: bool,
            pub fn load(d: *const cuda.Driver, image: []const u8, controls: bool) !Kernels {
                var out: Kernels = .{ .module = try cuda.Module.load(d, image), .controls = controls };
                errdefer out.module.unload();
                for (names[0..if (controls) 4 else 1], 0..) |name, i| {
                    out.f[i] = try out.module.function(name);
                    if (try out.f[i].attribute(.max_threads_per_block) < 256) return error.HcPointwiseThreads;
                }
                return out;
            }
            pub fn close(k: *Kernels, s: cuda.Stream) !void {
                if (s.d != k.module.d) return error.HcPointwiseContext;
                try s.synchronize();
                k.module.unload();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, s: cuda.Stream, p: h.Params, v: [4]View) !void {
                return k.testEncode(s, p, v, .production, 0);
            }
            pub fn testEncode(k: *const Kernels, s: cuda.Stream, p: h.Params, v: [4]View, control: Control, repeat: c_int) !void {
                if (s.d != k.module.d) return error.HcPointwiseContext;
                if (control != .production and !k.controls) return error.HcPointwiseControl;
                var ranges: [4]h.Range = undefined;
                var args: cuda.Args = .{};
                for (v, 0..) |view, i| {
                    if (view.buffer.d != s.d) return error.HcPointwiseContext;
                    if (view.offset > view.buffer.len or view.len > view.buffer.len - view.offset) return error.HcPointwiseBuffer;
                    const address = try std.math.add(usize, @intCast(view.buffer.ptr), view.offset);
                    ranges[i] = .{ .address = address, .len = view.len };
                    args.add(@as(u64, @intCast(address)));
                }
                const n = try h.validate(p, ranges);
                args.add(p.rows);
                args.add(p.hidden);
                args.add(p.streams);
                args.add(@as(c_int, @backingInt(p.op)));
                args.add(@as(c_int, @backingInt(p.round_mid)));
                args.add(p.gate_stride);
                args.add(p.gate_dim);
                args.add(repeat);
                try cuda.launch.launch(k.f[@backingInt(control)], .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &args);
            }
        };
    };
}
