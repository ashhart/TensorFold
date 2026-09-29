const std = @import("std");
const mx = @import("mlx.zig");
const dense = @import("deepseek_dense.zig");
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { key: []const u8, group: i32, scalar_ok: bool, rows: []i32 };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var calls: usize = 0;
    for (cases.value) |case| {
        errdefer std.debug.print("Failed calibrated SIMD case {s}, group {d}\n", .{ case.key, case.group });
        var store = @import("checkpoint.zig").Store.init(64);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.key }), "", "");
        const p = dense.Projection{ .weights = .{ try store.get("weight"), try store.get("scales"), try store.get("biases") }, .group = case.group };
        var dispatch = dense.Dense{};
        defer dispatch.deinit();
        try dispatch.prepare(&kernels, &.{p});
        var values = dispatch.checked.valueIterator();
        if (values.next().?.* != case.scalar_ok) return error.CalibrationMismatch;
        try dispatch.prepare(&kernels, &.{p});
        if (dispatch.checked.count() != 1) return error.DuplicateCalibration;
        for (case.rows) |rows| {
            var s = mx.Scope{};
            defer s.deinit();
            const x = try s.slice(try store.get("x"), 0, 0, rows);
            const expected = try store.get(try std.fmt.bufPrint(&path, "out{d}", .{rows}));
            try @import("sampling_checks.zig").equal(&s, try dispatch.apply(&kernels, &s, x, p), expected);
            // Physical SIMD group counts must preserve the calibrated reduction.
            for ([_]i32{ 1, 2, 4, 8, 16 }) |groups| {
                try @import("sampling_checks.zig").equal(&s, try dense.launch(&kernels, &s, x, p, false, groups), expected);
                calls += 1;
            }
            calls += 1;
        }
    }
    std.debug.print("PASS: {d} calibrated SIMD projections and {d} row/group dispatches match upstream bit for bit.\n", .{ cases.value.len, calls });
}
