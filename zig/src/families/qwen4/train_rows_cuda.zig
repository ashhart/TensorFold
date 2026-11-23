//! Row pullbacks use the inference stream; checked views precede every native CUDA launch.
const std = @import("std");
const rows = @import("core").living_rows;
const wo = @import("core").weight_only;
pub fn Kernels(comptime C: type) type {
    return Adapter(C, rows);
}
pub fn Adapter(comptime C: type, comptime spec: type) type {
    return struct {
        const Self = @This();
        pub const View = struct { buffer: C.DeviceBuffer, offset: usize = 0, len: usize };
        module: C.Module,
        functions: [spec.names.len]C.Function,
        pub fn load(d: *const C.Driver, image: []const u8) !Self {
            return loadTest(d, image, .production);
        }
        pub fn loadTest(d: *const C.Driver, image: []const u8, control: rows.Control) !Self {
            var out: Self = .{ .module = try C.Module.load(d, image), .functions = undefined };
            errdefer out.module.unload();
            for (spec.names, &out.functions) |name, *f| {
                var buffer: [192]u8 = undefined;
                const symbol = try std.fmt.bufPrintSentinel(&buffer, "{s}{s}", .{ name, rows.suffixes[@backingInt(control)] }, 0);
                f.* = try out.module.function(symbol);
                if (try f.attribute(.max_threads_per_block) < spec.threads) return error.LivingRowsThreads;
            }
            return out;
        }
        pub fn deinitDrained(k: *Self) void {
            k.module.unload();
            k.* = undefined;
        }
        pub fn encode(k: *const Self, stream: C.Stream, op: spec.Op, params: spec.Params, views: [11]View) !void {
            if (stream.d != k.module.d) return error.LivingRowsContext;
            var ranges: [11]wo.Range = undefined;
            var pointers: [11]u64 = undefined;
            for (views, &ranges, &pointers) |v, *r, *ptr| {
                if (v.buffer.d != stream.d) return error.LivingRowsContext;
                if (v.offset > v.buffer.len or v.len > v.buffer.len - v.offset) return error.LivingRowsBuffer;
                ptr.* = try std.math.add(u64, v.buffer.ptr, v.offset);
                r.* = .{ .address = std.math.cast(usize, ptr.*) orelse return error.LivingRowsBuffer, .len = v.len };
            }
            try spec.validate(op, params, ranges);
            var args: C.Args = .{};
            for (pointers) |ptr| args.add(ptr);
            args.add(params);
            try C.launch.launch(k.functions[@backingInt(op)], .{ .grid = .{ .x = @intCast((@as(usize, params.u[15]) + spec.threads - 1) / spec.threads) }, .block = .{ .x = spec.threads } }, stream, &args);
        }
    };
}
