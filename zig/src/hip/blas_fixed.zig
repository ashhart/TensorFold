//! BF16/FP32 hipBLASLt products with one single-row algorithm retained across request widths.
const std = @import("std");
const platform = @import("sys");
const Stream = @import("stream.zig").Stream;
const Shape = @import("blas.zig").Shape;
const P = ?*anyopaque;
const C = ?*const anyopaque;
const Algo = extern struct { data: [16]u8, max_workspace: usize };
const Result = extern struct { algo: Algo, workspace: usize, state: c_int, waves: f32, reserved: [4]c_int };
const Api = struct {
    hipblasLtCreate: *const fn (*P) callconv(.c) c_int,
    hipblasLtDestroy: *const fn (P) callconv(.c) c_int,
    hipblasLtMatmulDescCreate: *const fn (*P, c_int, c_int) callconv(.c) c_int,
    hipblasLtMatmulDescDestroy: *const fn (P) callconv(.c) c_int,
    hipblasLtMatmulDescSetAttribute: *const fn (P, c_int, C, usize) callconv(.c) c_int,
    hipblasLtMatrixLayoutCreate: *const fn (*P, c_int, u64, u64, i64) callconv(.c) c_int,
    hipblasLtMatrixLayoutDestroy: *const fn (P) callconv(.c) c_int,
    hipblasLtMatrixLayoutSetAttribute: *const fn (P, c_int, C, usize) callconv(.c) c_int,
    hipblasLtMatmulPreferenceCreate: *const fn (*P) callconv(.c) c_int,
    hipblasLtMatmulPreferenceDestroy: *const fn (P) callconv(.c) c_int,
    hipblasLtMatmulAlgoGetHeuristic: *const fn (P, P, P, P, P, P, P, c_int, *Result, *c_int) callconv(.c) c_int,
    hipblasLtMatmul: *const fn (P, P, C, C, P, C, P, C, C, P, P, P, *const Algo, P, usize, @import("abi.zig").Stream) callconv(.c) c_int,
};
const Plan = struct { a: P = null, b: P = null, out: P = null, n: usize, k: usize, algo: Algo = undefined };
pub const Dimensions = struct { n: usize, k: usize };

pub const Fixed = struct {
    lib: platform.DynLib,
    api: Api,
    handle: P,
    desc: P,
    stream: Stream,
    plans: [8]Plan = undefined,
    count: usize = 0,

    pub fn open(stream: Stream, dimensions: []const Dimensions) !Fixed {
        if (dimensions.len == 0 or dimensions.len > 8) return error.Invalid;
        for (dimensions, 0..) |d, i| {
            _ = try Shape.of(1, d.n, d.k);
            for (dimensions[0..i]) |previous| if (previous.n == d.n and previous.k == d.k) return error.Invalid;
        }
        for (platform.optional_libraries.spec(.hipblaslt).names) |name| {
            var lib = platform.DynLib.open(name) catch continue;
            errdefer lib.close();
            var api: Api = undefined;
            const info = @typeInfo(Api).@"struct";
            inline for (info.field_names, info.field_types) |field, T|
                @field(api, field) = lib.lookup(T, field) orelse return platform.optional_libraries.unavailable(.hipblaslt);
            var handle: P = null;
            try check(api.hipblasLtCreate(&handle));
            errdefer _ = api.hipblasLtDestroy(handle);
            var desc: P = null;
            try check(api.hipblasLtMatmulDescCreate(&desc, 2, 0)); // FP32 compute and scalar.
            errdefer _ = api.hipblasLtMatmulDescDestroy(desc);
            const trans: c_int = 112;
            try check(api.hipblasLtMatmulDescSetAttribute(desc, 0, &trans, @sizeOf(c_int)));
            var pref: P = null;
            try check(api.hipblasLtMatmulPreferenceCreate(&pref)); // Default: zero workspace.
            defer _ = api.hipblasLtMatmulPreferenceDestroy(pref);
            var fixed: Fixed = .{ .lib = lib, .api = api, .handle = handle, .desc = desc, .stream = stream };
            errdefer fixed.destroyPlans();
            for (dimensions) |d| {
                var plan: Plan = .{ .n = d.n, .k = d.k };
                errdefer fixed.destroyPlan(plan);
                try check(api.hipblasLtMatrixLayoutCreate(&plan.a, 14, d.k, d.n, @intCast(d.k)));
                try check(api.hipblasLtMatrixLayoutCreate(&plan.b, 14, d.k, 1, @intCast(d.k)));
                try check(api.hipblasLtMatrixLayoutCreate(&plan.out, 14, d.n, 1, @intCast(d.n)));
                var result: Result = undefined;
                var count: c_int = 0;
                try check(api.hipblasLtMatmulAlgoGetHeuristic(handle, desc, plan.a, plan.b, plan.out, plan.out, pref, 1, &result, &count));
                if (count != 1 or result.state != 0 or result.workspace != 0) return error.NoFixedHipblasltAlgorithm;
                plan.algo = result.algo;
                fixed.plans[fixed.count] = plan;
                fixed.count += 1;
            }
            return fixed;
        }
        return platform.optional_libraries.unavailable(.hipblaslt);
    }
    fn destroyPlan(f: *const Fixed, p: Plan) void {
        if (p.out != null) _ = f.api.hipblasLtMatrixLayoutDestroy(p.out);
        if (p.b != null) _ = f.api.hipblasLtMatrixLayoutDestroy(p.b);
        if (p.a != null) _ = f.api.hipblasLtMatrixLayoutDestroy(p.a);
    }
    fn destroyPlans(f: *const Fixed) void {
        for (f.plans[0..f.count]) |p| f.destroyPlan(p);
    }
    pub fn close(f: *Fixed) void {
        f.stream.synchronize() catch {};
        f.destroyPlans();
        _ = f.api.hipblasLtMatmulDescDestroy(f.desc);
        _ = f.api.hipblasLtDestroy(f.handle);
        f.lib.close();
        f.* = undefined;
    }
    /// The owner is confined to its stream; descriptor changes and submissions must not race on host threads.
    pub fn project(f: *const Fixed, x: u64, weight: u64, out: u64, m: usize, n: usize, k: usize) !void {
        _ = try Shape.of(m, n, k);
        try validate(x, weight, out, m, n, k);
        const p = for (f.plans[0..f.count]) |*plan| {
            if (plan.n == n and plan.k == k) break plan;
        } else return error.Invalid;
        const cols: u64 = m;
        try check(f.api.hipblasLtMatrixLayoutSetAttribute(p.b, 5, &cols, @sizeOf(u64)));
        try check(f.api.hipblasLtMatrixLayoutSetAttribute(p.out, 5, &cols, @sizeOf(u64)));
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(f.api.hipblasLtMatmul(f.handle, f.desc, &one, @ptrFromInt(weight), p.a, @ptrFromInt(x), p.b, &zero, @ptrFromInt(out), p.out, @ptrFromInt(out), p.out, &p.algo, null, 0, f.stream.handle));
    }
};

fn validate(x: u64, w: u64, y: u64, m: usize, n: usize, k: usize) !void {
    if (x == 0 or w == 0 or y == 0 or x % 2 != 0 or w % 2 != 0 or y % 2 != 0) return error.Invalid;
    const xb = std.math.mul(u64, m, k * 2) catch return error.Invalid;
    const wb = std.math.mul(u64, n, k * 2) catch return error.Invalid;
    const yb = std.math.mul(u64, m, n * 2) catch return error.Invalid;
    const xe = std.math.add(u64, x, xb) catch return error.Invalid;
    const we = std.math.add(u64, w, wb) catch return error.Invalid;
    const ye = std.math.add(u64, y, yb) catch return error.Invalid;
    if ((x < ye and y < xe) or (w < ye and y < we)) return error.Invalid;
}
fn check(status: c_int) !void {
    if (status != 0) {
        std.log.err("hipBLASLt status {d}", .{status});
        return error.HipblasltFailed;
    }
}

test "HIP fixed BLAS: ABI, shapes and aliases are checked without library calls" {
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Algo));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Result));
    const stream: Stream = undefined;
    try std.testing.expectError(error.Invalid, Fixed.open(stream, &.{}));
    try std.testing.expectError(error.Invalid, Fixed.open(stream, &.{.{ .n = 0, .k = 1 }}));
    try std.testing.expectError(error.Invalid, Fixed.open(stream, &.{ .{ .n = 1, .k = 1 }, .{ .n = 1, .k = 1 } }));
    const f: Fixed = undefined;
    try std.testing.expectError(error.Invalid, f.project(2, 4, 6, 0, 1, 1));
    try std.testing.expectError(error.Invalid, f.project(100, 200, 102, 2, 2, 2));
    try std.testing.expectError(error.Invalid, f.project(std.math.maxInt(u64) - 1, 100, 200, 1, 1, 2));
}
