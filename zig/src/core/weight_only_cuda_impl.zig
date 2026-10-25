//! Shared launch adapter instantiated in its owning core or native family module.
const std = @import("std");
pub fn Adapter(comptime wo: type, comptime cuda: type) type {
    return struct {
        pub const Control = enum(u3) { production, nan, repeat, tail, sums };
        pub const names = [_][:0]const u8{
            "tf_qwen4_projection",          "tf_qwen4_projection_bad_nan",  "tf_qwen4_projection_bad_repeat",
            "tf_qwen4_projection_bad_tail", "tf_qwen4_projection_bad_sums",
        };
        pub const View = struct {
            buffer: *const cuda.DeviceBuffer,
            offset: usize = 0,
            len: usize,
            fn range(v: View, d: *const cuda.Driver) !wo.Range {
                if (v.buffer.d != d) return error.WeightOnlyContext;
                if (v.offset > v.buffer.len or v.len > v.buffer.len - v.offset) return error.WeightOnlyBuffer;
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
                    if (try out.fns[i].attribute(.max_threads_per_block) < 128) return error.WeightOnlyThreads;
                }
                return out;
            }
            /// A failed drain retains the module so queued functions cannot outlive it; the caller owns retry/teardown.
            pub fn close(k: *Kernels, stream: cuda.Stream) !void {
                if (stream.d != k.module.d) return error.WeightOnlyContext;
                try stream.synchronize();
                k.module.unload();
                k.* = undefined;
            }
            pub fn encode(k: *const Kernels, stream: cuda.Stream, p: wo.Params, x: View, w: View, s: View, b: View, out: View) !void {
                return k.testEncode(stream, p, x, w, s, b, out, .production, 0);
            }
            /// Contract images alone expose controls; production cannot select a planted variant.
            pub fn testEncode(k: *const Kernels, stream: cuda.Stream, p: wo.Params, x: View, w: View, s: View, b: View, out: View, control: Control, repetition: c_int) !void {
                if (control != .production and !k.controls) return error.WeightOnlyControl;
                if (stream.d != k.module.d) return error.WeightOnlyContext;
                const d = stream.d;
                const ranges = [_]wo.Range{ try x.range(d), try w.range(d), try s.range(d), try b.range(d), try out.range(d) };
                try wo.buffers(p, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4]);
                var args: cuda.Args = .{};
                for (ranges) |v| args.add(@as(u64, @intCast(v.address)));
                args.add(p);
                args.add(repetition);
                try cuda.launch.launch(k.fns[@backingInt(control)], .{
                    .grid = .{ .x = @intCast((@as(usize, @intCast(p.n)) + 3) / 4), .y = @intCast(p.rows) },
                    .block = .{ .x = 128 },
                }, stream, &args);
            }
        };
    };
}
