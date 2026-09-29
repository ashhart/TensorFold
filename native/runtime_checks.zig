const std = @import("std");
const mx = @import("mlx.zig");

fn arithmetic(stream: mx.c.mlx_stream) !void {
    const a = mx.c.mlx_array_new_data(&[_]f32{ 1, 2, -3, 0.5 }, &[_]c_int{4}, 1, mx.c.MLX_FLOAT32);
    defer mx.free(a);
    const b = mx.c.mlx_array_new_data(&[_]f32{ 3, -4, 2, 0.25 }, &[_]c_int{4}, 1, mx.c.MLX_FLOAT32);
    defer mx.free(b);
    var sum = mx.c.mlx_array_new();
    defer mx.free(sum);
    try mx.check(mx.c.mlx_add(&sum, a, b, stream));
    try mx.eval(sum);
    const actual = mx.c.mlx_array_data_float32(sum)[0..4];
    if (!std.mem.eql(f32, actual, &.{ 4, -2, -1, 0.75 })) return error.RuntimeArithmeticMismatch;
}

pub fn check(io: std.Io, path: []const u8) !void {
    try mx.checkVersion();
    const cpu = mx.c.mlx_default_cpu_stream_new();
    defer _ = mx.c.mlx_stream_free(cpu);
    try arithmetic(cpu);
    var available: bool = false;
    try mx.check(mx.c.mlx_metal_is_available(&available));
    if (available) {
        try mx.init();
        defer mx.shutdown();
        try arithmetic(mx.stream);
    }
    const bytes = try std.json.Stringify.valueAlloc(mx.allocator, .{
        .mlx_version = @import("native_runtime").mlx_version,
        .cpu_arithmetic = true,
        .metal_available = available,
        .metal_arithmetic = available,
        .tensor_units = available and mx.tensor_units,
    }, .{ .whitespace = .indent_2 });
    defer mx.allocator.free(bytes);
    if (std.fs.path.dirname(path)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
    std.debug.print("PASS: loaded MLX-C CPU arithmetic; Metal {s}. Capabilities: {s}\n", .{ if (available) "arithmetic passed" else "unavailable (GPU checks not run)", path });
}
