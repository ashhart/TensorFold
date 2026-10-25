//! Qwen4 uses Flash Next's maintained 128-wide recurrence with runtime heads, exact weight reads and explicit gating.
const std = @import("std");
const wo = @import("core").weight_only;
pub const Gate = enum(c_int) { silu, sigmoid };
pub const Control = @import("gdn_control_source.zig").Control;
pub const Params = struct { rows: usize, keys: usize, values: usize, eps: f32, gate: Gate = .silu, weight_flags: u4 = 3, row_stride: usize = 0 };
/// Widened storage retains checkpoint dtypes at the convolution and norm rounding nodes.
pub fn weightFlags(conv_bf16: bool, norm_bf16: bool) u4 {
    return @as(u4, if (norm_bf16) 3 else 7) | @as(u4, if (conv_bf16) 8 else 0);
}
pub const Sizes = struct { projection: usize, conv_state: usize, snapshots: usize, conv_weight: usize, state: usize, scalar: usize, norm: usize, output: usize, sums: usize };
pub fn sizes(p: Params) !Sizes {
    if (p.rows == 0 or p.rows > 128 or p.keys == 0 or p.values == 0 or p.keys > p.values or p.values % p.keys != 0 or !std.math.isFinite(p.eps) or p.eps <= 0) return error.Qwen4GdnShape;
    if (p.weight_flags & 4 != 0 and p.weight_flags & 2 == 0) return error.Qwen4GdnShape;
    const channels = try std.math.add(usize, try std.math.mul(usize, p.keys, 256), try std.math.mul(usize, p.values, 128));
    const width = try std.math.add(usize, channels, try std.math.mul(usize, p.values, 130));
    if (channels > std.math.maxInt(c_int) / 4 or width > std.math.maxInt(c_int)) return error.Qwen4GdnShape;
    const state = try std.math.mul(usize, p.values, 128 * 128 * 4);
    const conv = try std.math.mul(usize, channels, 6);
    if (p.row_stride != 0 and (p.row_stride < @max(state, conv) or p.row_stride % 16 != 0)) return error.Qwen4GdnShape;
    return .{
        .projection = try std.math.mul(usize, try std.math.mul(usize, p.rows, width), 2),
        .conv_state = try std.math.mul(usize, channels, 6),
        .conv_weight = try std.math.mul(usize, channels, if (p.weight_flags & 1 != 0) @as(usize, 16) else 8),
        .state = try std.math.mul(usize, p.values, 128 * 128 * 4),
        .snapshots = try extent(p, state),
        .scalar = try std.math.mul(usize, p.values, 4),
        .norm = if (p.weight_flags & 2 != 0) 512 else 256,
        .output = try std.math.mul(usize, try std.math.mul(usize, p.rows, p.values), 256),
        .sums = try std.math.mul(usize, try std.math.mul(usize, p.rows, p.values), 16),
    };
}
pub fn extent(p: Params, bytes: usize) !usize {
    if (p.rows == 0) return error.Qwen4GdnShape;
    return std.math.add(usize, try std.math.mul(usize, p.rows - 1, if (p.row_stride == 0) bytes else p.row_stride), bytes);
}
/// Save raw convolution inputs for every accepted prefix on the same inference queue.
pub fn saveConv(backend: anytype, p: Params, dest: anytype, projection: anytype, prior: anytype) !void {
    const z = try sizes(p);
    const width = z.conv_state / 3;
    const input_stride = z.projection / p.rows;
    const stride = if (p.row_stride == 0) z.conv_state else p.row_stride;
    if (dest.len < try extent(p, z.conv_state) or projection.len < z.projection or prior.len < z.conv_state) return error.Qwen4GdnBuffer;
    for (0..p.rows) |r| for (0..3) |tap| {
        const at = r + tap + 1;
        if (at < 3) try backend.copy(dest.buffer, dest.offset + r * stride + tap * width, prior.buffer, prior.offset + at * width, width) else try backend.copy(dest.buffer, dest.offset + r * stride + tap * width, projection.buffer, projection.offset + (at - 3) * input_stride, width);
    };
}
fn overlaps(a: wo.Range, b: wo.Range) bool {
    if (a.len == 0 or b.len == 0) return false;
    return if (a.address <= b.address) b.address - a.address < a.len else a.address - b.address < b.len;
}
pub fn validate(p: Params, ranges: [10]wo.Range) !void {
    const s = try sizes(p);
    const bytes = [_]usize{ s.projection, s.conv_state, s.conv_weight, s.state, s.scalar, s.scalar, s.norm, s.output, s.sums, s.snapshots };
    for (ranges, bytes, 0..) |range, need, i| {
        if (range.address == 0 or range.len < need) return error.Qwen4GdnBuffer;
        _ = std.math.add(usize, range.address, range.len) catch return error.Qwen4GdnBuffer;
        const align_to: usize = if (i == 0 or i == 1 or i == 7 or (i == 2 and p.weight_flags & 1 == 0) or (i == 6 and p.weight_flags & 2 == 0)) 2 else 4;
        if (range.address % align_to != 0) return error.Qwen4GdnAlignment;
    }
    for (7..10) |i| for (0..i) |j| {
        // The recurrence may consume and publish exactly the same state allocation after all input cells are loaded.
        if (p.rows == 1 and i == 9 and j == 3 and ranges[i].address == ranges[j].address and ranges[i].len == ranges[j].len) continue;
        if (overlaps(ranges[i], ranges[j])) return error.Qwen4GdnAlias;
    };
}
pub fn Kernels(comptime cuda: type) type {
    return struct {
        const Self = @This();
        module: cuda.Module,
        function: cuda.Function,
        controls: ?[4]cuda.Function = null,
        pub fn load(driver: *const cuda.Driver, image: []const u8) !Self {
            return loadControls(driver, image, false);
        }
        pub fn loadControls(driver: *const cuda.Driver, image: []const u8, controls: bool) !Self {
            var module = try cuda.Module.load(driver, image);
            errdefer module.unload();
            const function = try module.function("qwen4_gdn_chain");
            if (try function.attribute(.max_threads_per_block) < 1024) return error.Qwen4GdnThreads;
            var out: Self = .{ .module = module, .function = function };
            if (controls) {
                var fns: [4]cuda.Function = undefined;
                for (&fns, @import("gdn_control_source.zig").cuda_names) |*f, name| {
                    f.* = try module.function(name);
                    if (try f.attribute(.max_threads_per_block) < 1024) return error.Qwen4GdnThreads;
                }
                out.controls = fns;
            }
            return out;
        }
        pub fn close(self: *Self, stream: cuda.Stream) !void {
            if (stream.d != self.module.d) return error.Qwen4GdnContext;
            try stream.synchronize();
            self.module.unload();
            self.* = undefined;
        }
        pub fn encode(self: *Self, stream: cuda.Stream, p: Params, ranges: [10]wo.Range) !void {
            return self.testEncode(stream, p, ranges, .production, 0);
        }
        pub fn testEncode(self: *Self, stream: cuda.Stream, p: Params, ranges: [10]wo.Range, control: Control, repeat: u32) !void {
            if (stream.d != self.module.d) return error.Qwen4GdnContext;
            if (repeat > std.math.maxInt(c_int)) return error.Qwen4GdnShape;
            const function = if (control == .production) self.function else (self.controls orelse return error.Qwen4GdnControls)[@backingInt(control) - 1];
            try validate(p, ranges);
            var args: cuda.Args = .{};
            for (ranges[0..7]) |range| args.add(@as(u64, @intCast(range.address)));
            args.add(p.eps);
            args.add(@as(c_int, @intCast(p.rows)));
            for (ranges[7..]) |range| args.add(@as(u64, @intCast(range.address)));
            args.add(@as(c_int, @intCast(p.keys)));
            args.add(@as(c_int, @intCast(p.values)));
            args.add(@as(c_int, @backingInt(p.gate)));
            args.add(@as(c_int, p.weight_flags));
            args.add(@as(u64, @intCast((if (p.row_stride == 0) (try sizes(p)).state else p.row_stride) / 4)));
            if (control != .production) args.add(@as(c_int, @intCast(repeat)));
            try cuda.launch.launch(function, .{ .grid = .{ .x = @intCast(p.values) }, .block = .{ .x = 1024 } }, stream, &args);
        }
    };
}
