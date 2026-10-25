//! Hyper-connection activations bind checked device views before recording any command.
const std = @import("std");
pub fn Adapter(comptime h: type, comptime metal: type) type {
    return struct {
        pub const Control = enum(u2) { production, nan, repeat, tail };
        pub const names = [_][]const u8{ "qwen4_hc_activation", "qwen4_hc_activation_bad_nan", "qwen4_hc_activation_bad_repeat", "qwen4_hc_activation_bad_tail" };
        pub const View = struct {
            buffer: metal.Buffer,
            offset: usize = 0,
            len: usize,
            fn range(v: View, device: metal.Device) !h.Range {
                if (v.buffer.deviceId() != device.id) return error.HcActivationContext;
                if (v.offset > v.buffer.length() or v.len > v.buffer.length() - v.offset) return error.HcActivationBuffer;
                return .{ .address = try std.math.add(usize, @intCast(v.buffer.gpuAddress()), v.offset), .len = v.len };
            }
        };
        pub fn source(a: std.mem.Allocator, raw: []const u8, activation: []const u8, wrapper: []const u8, controls: bool) ![]u8 {
            const first = "// TF_WEIGHT_ONLY_HEADER";
            const second = "// TF_HC_ACTIVATION_HEADER";
            if (std.mem.count(u8, wrapper, first) != 1 or std.mem.count(u8, wrapper, second) != 1) return error.HcActivationSource;
            const body = try std.mem.replaceOwned(u8, a, wrapper, first, raw);
            defer a.free(body);
            const inserted = try std.mem.replaceOwned(u8, a, body, second, activation);
            defer a.free(inserted);
            return std.mem.concat(a, u8, &.{ if (controls) "#define TF_KERNEL_CONTRACT 1\n" else "", inserted });
        }
        pub const Kernels = struct {
            device: metal.Device,
            pipes: [names.len]metal.Pipeline = undefined,
            count: usize = 0,
            pub fn load(device: metal.Device, text: []const u8, controls: bool) !Kernels {
                const lib = try metal.Library.fromSource(device, text, metal.CompileOptions.mlx());
                defer lib.deinit();
                var out: Kernels = .{ .device = device };
                errdefer out.deinit();
                for (names[0..if (controls) names.len else 1]) |name| {
                    const pipe = try metal.Pipeline.init(device, lib, name, false);
                    out.pipes[out.count] = pipe;
                    out.count += 1;
                    if (pipe.maxThreads() < 256) return error.HcActivationThreads;
                }
                return out;
            }
            /// The owner drains its command buffers before releasing the pipelines.
            pub fn deinit(k: *Kernels) void {
                for (k.pipes[0..k.count]) |pipe| pipe.deinit();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, e: metal.ComputeEncoder, p: h.Params, x: View, out: View) !void {
                return k.testEncode(e, p, x, out, .production, 0);
            }
            pub fn testEncode(k: *const Kernels, e: metal.ComputeEncoder, p: h.Params, x: View, out: View, control: Control, repeat: c_int) !void {
                if (@backingInt(control) >= k.count) return error.HcActivationControl;
                if (e.deviceId() != k.device.id) return error.HcActivationContext;
                const n = try h.validate(p, try x.range(k.device), try out.range(k.device));
                var params = p;
                params.repeat = repeat;
                e.setPipeline(k.pipes[@backingInt(control)]);
                e.setBuffer(x.buffer, x.offset, 0);
                e.setBuffer(out.buffer, out.offset, 1);
                e.setBytes(std.mem.asBytes(&params), 2);
                e.dispatchGroups(metal.Size.of((n + 255) / 256, 1, 1), metal.Size.of(256, 1, 1));
                e.barrier();
            }
        };
    };
}
