//! Shared launch adapter instantiated in its owning core or native family module.
const std = @import("std");
pub fn Adapter(comptime wo: type, comptime mtl: type) type {
    return struct {
        pub const Control = enum(u3) { production, nan, repeat, tail, sums };
        pub const names = [_][]const u8{
            "tf_weight_only_projection",          "tf_weight_only_projection_bad_nan",  "tf_weight_only_projection_bad_repeat",
            "tf_weight_only_projection_bad_tail", "tf_weight_only_projection_bad_sums",
        };
        pub const View = struct {
            buffer: mtl.Buffer,
            offset: usize = 0,
            len: usize,
            fn range(v: View, device: mtl.Device) !wo.Range {
                if (v.buffer.deviceId() != device.id) return error.WeightOnlyContext;
                if (v.offset > v.buffer.length() or v.len > v.buffer.length() - v.offset) return error.WeightOnlyBuffer;
                return .{ .address = try std.math.add(usize, @intCast(v.buffer.gpuAddress()), v.offset), .len = v.len };
            }
        };
        pub fn source(a: std.mem.Allocator, header: []const u8, wrapper: []const u8, controls: bool) ![]u8 {
            const marker = "// TF_WEIGHT_ONLY_HEADER";
            if (std.mem.count(u8, wrapper, marker) != 1) return error.WeightOnlySource;
            const defined = try std.mem.concat(a, u8, &.{ if (controls) "#define TF_KERNEL_CONTRACT 1\n" else "", header });
            defer a.free(defined);
            return std.mem.replaceOwned(u8, a, wrapper, marker, defined);
        }
        pub const Kernels = struct {
            device: mtl.Device,
            pipes: [names.len]mtl.Pipeline = undefined,
            count: usize = 0,
            pub fn load(device: mtl.Device, text: []const u8, controls: bool) !Kernels {
                const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
                defer lib.deinit();
                var out: Kernels = .{ .device = device };
                errdefer out.deinit();
                for (names[0..if (controls) names.len else 1]) |name| {
                    const pipe = try mtl.Pipeline.init(device, lib, name, false);
                    out.pipes[out.count] = pipe;
                    out.count += 1;
                    if (pipe.maxThreads() < 128 or pipe.simdWidth() != 32) return error.WeightOnlyThreads;
                }
                return out;
            }
            /// The owner must drain its command buffers before releasing these pipelines.
            pub fn deinit(k: *Kernels) void {
                for (k.pipes[0..k.count]) |pipe| pipe.deinit();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, enc: mtl.ComputeEncoder, p: wo.Params, x: View, w: View, s: View, b: View, out: View) !void {
                return k.testEncode(enc, p, x, w, s, b, out, .production, 0);
            }
            pub fn testEncode(k: *const Kernels, enc: mtl.ComputeEncoder, p: wo.Params, x: View, w: View, s: View, b: View, out: View, control: Control, repetition: c_int) !void {
                if (@backingInt(control) >= k.count) return error.WeightOnlyControl;
                if (enc.deviceId() != k.device.id) return error.WeightOnlyContext;
                const views = [_]View{ x, w, s, b, out };
                const ranges = [_]wo.Range{ try x.range(k.device), try w.range(k.device), try s.range(k.device), try b.range(k.device), try out.range(k.device) };
                try wo.buffers(p, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4]);
                enc.setPipeline(k.pipes[@backingInt(control)]);
                for (views, 0..) |v, i| enc.setBuffer(v.buffer, v.offset, i);
                enc.setBytes(std.mem.asBytes(&p), 5);
                enc.setBytes(std.mem.asBytes(&repetition), 6);
                enc.dispatchGroups(mtl.Size.of((@as(usize, @intCast(p.n)) + 3) / 4, @intCast(p.rows), 1), mtl.Size.of(128, 1, 1));
                enc.barrier();
            }
        };
    };
}
