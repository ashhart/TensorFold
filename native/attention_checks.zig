const std = @import("std");
const mx = @import("mlx.zig");
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    if (!mx.tensor_units) return error.TensorHardwareRequired;
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var store = @import("checkpoint.zig").Store.init(64);
    defer store.deinit();
    var path: [4096]u8 = undefined;
    try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { key: []const u8, length: i32, scale: f32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    const equal = @import("sampling_checks.zig").equal;
    for (cases.value) |case| {
        var s = mx.Scope{};
        defer s.deinit();
        const q = try store.field(case.key, "q");
        const k = try s.slice(try store.field(case.key, "k"), 2, 0, case.length);
        const v = try s.slice(try store.field(case.key, "v"), 2, 0, case.length);
        const out = try @import("lanes.zig").sdpa(&kernels, &s, q, k, v, case.scale);
        try equal(&s, out, try store.field(case.key, "expected"));
        const count = mx.dim(q, 2);
        for (0..@intCast(count)) |row| {
            const j: i32 = @intCast(row);
            const end = case.length - count + j + 1;
            const single = try @import("lanes.zig").sdpa(&kernels, &s, try s.slice(q, 2, j, j + 1), try s.slice(k, 2, 0, end), try s.slice(v, 2, 0, end), case.scale);
            try equal(&s, single, try s.slice(out, 2, j, j + 1));
        }
    }
    std.debug.print("PASS: {d} long-context tensor attention fixtures and all serial rows match bit for bit.\n", .{cases.value.len});
}
