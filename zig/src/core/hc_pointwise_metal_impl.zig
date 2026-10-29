//! Native Metal branch and output gates validate all borrowed views before recording their first binding.
const std = @import("std");
pub fn Adapter(comptime h: type, comptime metal: type) type {
    return struct {
        pub const Control = enum(u2) { production, nan, repeat, tail };
        pub const names = [_][]const u8{ "qwen4_hc_pointwise", "qwen4_hc_pointwise_bad_nan", "qwen4_hc_pointwise_bad_repeat", "qwen4_hc_pointwise_bad_tail" };
        pub const View = struct { buffer: metal.Buffer, offset: usize = 0, len: usize };
        pub const source = @import("hc_activation_metal_impl.zig").Adapter(h, metal).source;
        pub const Kernels = struct {
            device: metal.Device,
            pipes: [4]metal.Pipeline = undefined,
            count: usize = 0,
            pub fn load(device: metal.Device, text: []const u8, controls: bool) !Kernels {
                const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
                defer lib.deinit();
                var out: Kernels = .{ .device = device };
                errdefer out.deinit();
                for (names[0..if (controls) 4 else 1]) |name| {
                    const pipe = try metal.Pipeline.init(device, lib, name, false);
                    out.pipes[out.count] = pipe;
                    out.count += 1;
                    if (pipe.maxThreads() < 256) return error.HcPointwiseThreads;
                }
                return out;
            }
            /// The owner drains command buffers before pipelines used by those buffers can be released.
            pub fn deinit(k: *Kernels) void {
                for (k.pipes[0..k.count]) |pipe| pipe.deinit();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, e: metal.ComputeEncoder, p: h.Params, v: [4]View) !void {
                return k.testEncode(e, p, v, .production, 0);
            }
            pub fn testEncode(k: *const Kernels, e: metal.ComputeEncoder, p: h.Params, v: [4]View, control: Control, repeat: c_int) !void {
                if (@backingInt(control) >= k.count) return error.HcPointwiseControl;
                if (e.deviceId() != k.device.id) return error.HcPointwiseContext;
                var ranges: [4]h.Range = undefined;
                for (v, 0..) |view, i| {
                    if (view.buffer.deviceId() != k.device.id) return error.HcPointwiseContext;
                    if (view.offset > view.buffer.length() or view.len > view.buffer.length() - view.offset) return error.HcPointwiseBuffer;
                    ranges[i] = .{ .address = try std.math.add(usize, @intCast(view.buffer.gpuAddress()), view.offset), .len = view.len };
                }
                const n = try h.validate(p, ranges);
                var params = p;
                params.repeat = repeat;
                e.setPipeline(k.pipes[@backingInt(control)]);
                for (v, 0..) |view, i| e.setBuffer(view.buffer, view.offset, i);
                e.setBytes(std.mem.asBytes(&params), 4);
                e.dispatchGroups(metal.Size.of((n + 255) / 256, 1, 1), metal.Size.of(256, 1, 1));
                e.barrier();
            }
        };
    };
}
