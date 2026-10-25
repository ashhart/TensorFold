//! Hyper-connection activation launches retain the consuming stream and module ownership.
const std = @import("std");
pub fn Adapter(comptime h: type, comptime cuda: type) type {
    return struct {
        pub const Control = enum(u2) { production, nan, repeat, tail };
        pub const names = [_][:0]const u8{ "qwen4_hc_activation", "qwen4_hc_activation_bad_nan", "qwen4_hc_activation_bad_repeat", "qwen4_hc_activation_bad_tail" };
        pub const View = struct {
            buffer: *const cuda.DeviceBuffer,
            offset: usize = 0,
            len: usize,
            fn range(v: View, d: *const cuda.Driver) !h.Range {
                if (v.buffer.d != d) return error.HcActivationContext;
                if (v.offset > v.buffer.len or v.len > v.buffer.len - v.offset) return error.HcActivationBuffer;
                return .{ .address = try std.math.add(usize, @intCast(v.buffer.ptr), v.offset), .len = v.len };
            }
        };
        pub const Kernels = struct {
            module: cuda.Module,
            fns: [names.len]cuda.Function = undefined,
            controls: bool,
            pub fn load(d: *const cuda.Driver, image: []const u8, controls: bool) !Kernels {
                var out: Kernels = .{ .module = try cuda.Module.load(d, image), .controls = controls };
                errdefer out.module.unload();
                for (names[0..if (controls) names.len else 1], 0..) |name, i| {
                    out.fns[i] = try out.module.function(name);
                    if (try out.fns[i].attribute(.max_threads_per_block) < 256) return error.HcActivationThreads;
                }
                return out;
            }
            pub fn close(k: *Kernels, stream: cuda.Stream) !void {
                if (stream.d != k.module.d) return error.HcActivationContext;
                try stream.synchronize();
                k.module.unload();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, s: cuda.Stream, p: h.Params, x: View, out: View) !void {
                return k.testEncode(s, p, x, out, .production, 0);
            }
            pub fn testEncode(k: *const Kernels, s: cuda.Stream, p: h.Params, x: View, out: View, control: Control, repeat: c_int) !void {
                if (control != .production and !k.controls) return error.HcActivationControl;
                if (s.d != k.module.d) return error.HcActivationContext;
                const xr = try x.range(s.d);
                const yr = try out.range(s.d);
                const n = try h.validate(p, xr, yr);
                var args: cuda.Args = .{};
                args.add(@as(u64, @intCast(xr.address)));
                args.add(@as(u64, @intCast(yr.address)));
                args.add(p.rows);
                args.add(p.cols);
                args.add(p.streams);
                args.add(p.injection);
                args.add(repeat);
                try cuda.launch.launch(k.fns[@backingInt(control)], .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &args);
            }
        };
    };
}
