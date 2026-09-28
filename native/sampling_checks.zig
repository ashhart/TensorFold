//! Independent Python Metal fixtures exercise both sampler mappings and radix top-k.
const std = @import("std");
const mx = @import("mlx.zig");
const gpu = @import("gpu_sampling.zig");
const Case = struct {
    key: []const u8,
    op: []const u8,
    k: usize,
    seed: u64 = 0,
    temperature: f64 = 1,
    p: f64 = 0.95,
    mapped: bool = false,
    positions: []const i32 = &.{},
};
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var store = @import("checkpoint.zig").Store.init(64);
    defer store.deinit();
    var path: [4096]u8 = undefined;
    try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    for (cases.value) |case| {
        errdefer std.debug.print("Failed sampling fixture {s}: {s}, k={d}, seed={d}, temperature={d}, mapped={}\n", .{ case.key, case.op, case.k, case.seed, case.temperature, case.mapped });
        var s = mx.Scope{};
        defer s.deinit();
        const x = try store.field(case.key, "x");
        if (std.mem.eql(u8, case.op, "topk")) {
            const out = try gpu.topk(&kernels, &s, x, @intCast(case.k));
            try equal(&s, out[0], try store.field(case.key, "indices"));
            try equal(&s, out[1], try store.field(case.key, "values"));
        } else if (std.mem.eql(u8, case.op, "sample")) {
            const out = try gpu.sample(&kernels, &s, x, case.positions, .{ .seed = case.seed, .temperature = case.temperature, .top_k = case.k, .top_p = case.p }, if (case.mapped) try store.field(case.key, "ids") else null);
            try equal(&s, out, try store.field(case.key, "expected"));
        } else if (std.mem.eql(u8, case.op, "cpu_sample")) {
            const out = try @import("sampling.zig").rowsMapped(&kernels, &s, x, case.positions, .{ .seed = case.seed, .temperature = case.temperature, .top_k = case.k, .top_p = case.p }, if (case.mapped) try store.field(case.key, "ids") else null);
            defer mx.allocator.free(out);
            try equal(&s, try s.cast(try s.ints(out), mx.c.MLX_UINT32), try store.field(case.key, "expected"));
        } else return error.UnknownFixtureOperation;
    }
    std.debug.print("PASS: {d} Python CPU/Metal sampling/top-k fixtures match bit for bit.\n", .{cases.value.len});
}
pub fn equal(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    if (mx.dtype(a) != mx.dtype(b) or !std.mem.eql(c_int, mx.shape(a), mx.shape(b))) return error.FixtureShapeMismatch;
    const x = try s.contiguous(try s.cast(a, mx.f32t));
    const y = try s.contiguous(try s.cast(b, mx.f32t));
    try mx.evalMany(&.{ x, y }, false);
    const n = mx.c.mlx_array_size(x);
    if (n == 0) return;
    const av = mx.c.mlx_array_data_float32(x)[0..n];
    const bv = mx.c.mlx_array_data_float32(y)[0..n];
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(av), std.mem.sliceAsBytes(bv))) {
        var differences: usize = 0;
        var maximum: f32 = 0;
        for (av, bv, 0..) |a_value, b_value, i| {
            if (@as(u32, @bitCast(a_value)) != @as(u32, @bitCast(b_value))) {
                if (differences == 0) std.debug.print("First mismatch at {d}: {d} vs {d}\n", .{ i, a_value, b_value });
                differences += 1;
                maximum = @max(maximum, @abs(a_value - b_value));
            }
        }
        std.debug.print("Mismatch: {d}/{d} values, max abs diff {d}, shape {any}\n", .{ differences, n, maximum, mx.shape(a) });
        return error.FixtureMismatch;
    }
}
