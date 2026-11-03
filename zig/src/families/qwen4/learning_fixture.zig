//! Test-only owned recording buffers execute the same scalar device body, never a production inference path.
const std = @import("std");
const core = @import("core");
const rows = core.living_rows;
const a = std.testing.allocator;
extern fn tf_living_gemma_cpu(c_uint, rows.Params, [*][*]u8) void;
extern fn tf_living_rows_cpu(c_uint, rows.Params, [*][*]u8) void;
extern fn tf_living_gdn_cpu(c_uint, rows.Params, [*][*]u8) void;
extern fn tf_living_qsa_cpu(c_uint, rows.Params, [*][*]u8) void;
pub fn Case(comptime spec: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        p: rows.Params,
        op: spec.Op,
        data: [11][]align(16) u8,
        pub fn init(op: spec.Op, p: rows.Params) !Self {
            return initWithAllocator(a, op, p);
        }
        pub fn initWithAllocator(allocator: std.mem.Allocator, op: spec.Op, p: rows.Params) !Self {
            var out: Self = .{ .a = allocator, .op = op, .p = try spec.finish(op, p), .data = undefined };
            const n = try spec.needs(op, p);
            var count: usize = 0;
            errdefer for (out.data[0..count]) |v| allocator.free(v);
            for (&out.data, n.bytes) |*v, size| {
                v.* = try allocator.alignedAlloc(u8, .@"16", @max(size, 16));
                @memset(v.*, 0);
                count += 1;
            }
            return out;
        }
        pub fn deinit(c: *Self) void {
            for (c.data) |v| c.a.free(v);
        }
        pub fn run(c: *Self) !void {
            var ranges: [11]core.weight_only.Range = undefined;
            var ptrs: [11][*]u8 = undefined;
            for (c.data, &ranges, &ptrs) |v, *r, *p| {
                r.* = .{ .address = @intFromPtr(v.ptr), .len = v.len };
                p.* = v.ptr;
            }
            try spec.validate(c.op, c.p, ranges);
            if (spec == core.living_rows) tf_living_rows_cpu(@intCast(@backingInt(c.op)), c.p, &ptrs) else if (spec == core.living_gdn) tf_living_gdn_cpu(@intCast(@backingInt(c.op)), c.p, &ptrs) else if (spec == core.living_qsa) tf_living_qsa_cpu(@intCast(@backingInt(c.op)), c.p, &ptrs) else tf_living_gemma_cpu(@intCast(@backingInt(c.op)), c.p, &ptrs);
        }
        pub fn put(c: *Self, at: usize, bytes: []const u8) !void {
            try std.testing.expect(c.data[at].len >= bytes.len);
            @memcpy(c.data[at][0..bytes.len], bytes);
        }
        pub fn floats(c: *Self, at: usize) []f32 {
            return std.mem.bytesAsSlice(f32, c.data[at]);
        }
    };
}
pub fn near(got: f32, want: f64) !void {
    if (!std.math.isFinite(got) or @abs(@as(f64, got) - want) > 2e-7 + 3e-5 * @abs(want)) return error.NativeMixerGradientMismatch;
}
