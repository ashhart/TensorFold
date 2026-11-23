//! Native Metal row pullbacks retain their pipelines until the owning command session proves completion.
const std = @import("std");
const rows = @import("core").living_rows;
const wo = @import("core").weight_only;
pub fn Kernels(comptime M: type) type {
    return Adapter(M, rows);
}
pub fn Adapter(comptime M: type, comptime spec: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: M.Buffer, offset: usize = 0, len: usize };
        device: M.Device,
        pipelines: [spec.names.len]M.Pipeline = undefined,
        count: usize = 0,
        pub fn load(a: std.mem.Allocator, device: M.Device) !Self {
            return loadTest(a, device, .production);
        }
        pub fn loadTest(a: std.mem.Allocator, device: M.Device, control: rows.Control) !Self {
            const source = try @import("train_source.zig").controlled(a, spec, control != .production);
            defer a.free(source);
            const lib = try M.Library.fromSource(device, source, M.CompileOptions.mlx());
            defer lib.deinit();
            var out: Self = .{ .device = device };
            errdefer out.deinitDrained();
            for (spec.names) |name| {
                var buffer: [192]u8 = undefined;
                const symbol = try std.fmt.bufPrint(&buffer, "{s}{s}", .{ name, rows.suffixes[@backingInt(control)] });
                const pipe = try M.Pipeline.init(device, lib, symbol, false);
                out.pipelines[out.count] = pipe;
                out.count += 1;
                if (pipe.maxThreads() < spec.threads) return error.LivingRowsThreads;
            }
            return out;
        }
        pub fn deinitDrained(k: *Self) void {
            for (k.pipelines[0..k.count]) |pipe| pipe.deinit();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, encoder: M.ComputeEncoder, op: spec.Op, params: spec.Params, views: [11]View) !void {
            if (encoder.deviceId() != k.device.id) return error.LivingRowsContext;
            var ranges: [11]wo.Range = undefined;
            for (views, &ranges) |v, *r| {
                if (v.buffer.deviceId() != k.device.id) return error.LivingRowsContext;
                if (v.offset > v.buffer.length() or v.len > v.buffer.length() - v.offset) return error.LivingRowsBuffer;
                r.* = .{ .address = try std.math.add(usize, @intCast(v.buffer.gpuAddress()), v.offset), .len = v.len };
            }
            try spec.validate(op, params, ranges);
            encoder.setPipeline(k.pipelines[@backingInt(op)]);
            for (views, 0..) |v, i| encoder.setBuffer(v.buffer, v.offset, i);
            encoder.setBytes(std.mem.asBytes(&params), 11);
            encoder.dispatchThreads(M.Size.of(params.u[15], 1, 1), M.Size.of(spec.threads, 1, 1));
            encoder.barrier();
        }
    };
}
