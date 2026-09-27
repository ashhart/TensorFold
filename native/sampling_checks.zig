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
    try store.loadFile(try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    for (cases.value) |case| {
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
        } else return error.UnknownFixtureOperation;
    }
    std.debug.print("PASS: {d} Python Metal sampling/top-k fixtures match bit for bit.\n", .{cases.value.len});
}
pub fn equal(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    if (mx.dtype(a) != mx.dtype(b) or !std.mem.eql(c_int, mx.shape(a), mx.shape(b))) return error.FixtureShapeMismatch;
    const x = try s.contiguous(try s.cast(a, mx.f32t));
    const y = try s.contiguous(try s.cast(b, mx.f32t));
    try mx.evalMany(&.{ x, y }, false);
    const n = mx.c.mlx_array_size(x);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(x)[0..n]), std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(y)[0..n]))) return error.FixtureMismatch;
}
