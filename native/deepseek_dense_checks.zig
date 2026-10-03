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
    const Case = struct { key: []const u8, group: i32, bits: i32 = 4, scalar_ok: bool, rows: []i32, members: []i32 = &.{} };
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    var calls: usize = 0;
    for (cases.value) |case| {
        errdefer std.debug.print("Failed calibrated SIMD case {s}, group {d}\n", .{ case.key, case.group });
        var store = @import("checkpoint.zig").Store.init(64);
        defer store.deinit();
        try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.key }), "", "");
        const p = dense.Projection{ .weights = .{ try store.get("weight"), try store.get("scales"), try store.get("biases") }, .group = case.group, .bits = case.bits };
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
            const equal = @import("sampling_checks.zig").equal;
            if (case.bits == 4 or case.scalar_ok) try equal(&s, try dispatch.apply(&kernels, &s, x, p), expected);
            if (case.bits != 4 and rows <= 128) {
                const was_tensor = mx.tensor_units;
                mx.tensor_units = false;
                defer mx.tensor_units = was_tensor;
                var linear = try @import("lanes.zig").Linear.initFormat(&s, p.weights[0], p.weights[1], p.weights[2], .{ .bits = p.bits, .group_size = p.group });
                defer linear.deinit();
                const fallback = try store.get(try std.fmt.bufPrint(&path, "fallback{d}", .{rows}));
                _ = kernels.affine.checked.remove(p.key());
                try equal(&s, try s.reshape(try linear.apply(&kernels, &s, .{ .x = x }), mx.shape(expected)), if (case.scalar_ok) expected else fallback);
                if (kernels.affine.checked.get(p.key()).? != case.scalar_ok) return error.CalibrationMismatch;
                try kernels.affine.checked.put(mx.allocator, p.key(), false);
                try equal(&s, try s.reshape(try linear.apply(&kernels, &s, .{ .x = x }), mx.shape(fallback)), fallback);
                try std.testing.expectError(error.SimdScalarMismatch, kernels.affine.apply(&kernels, &s, x, p));
                _ = kernels.affine.checked.remove(p.key());
                calls += 2;
                if (case.members.len > 0) {
                    const Linear = @import("lanes.zig").Linear;
                    var members: [4]Linear = undefined;
                    var initialized: usize = 0;
                    defer for (members[0..initialized]) |*member| member.deinit();
                    var offset: i32 = 0;
                    for (case.members) |width| {
                        members[initialized] = try Linear.initFormat(&s, try s.slice(p.weights[0], 0, offset, offset + width), try s.slice(p.weights[1], 0, offset, offset + width), try s.slice(p.weights[2], 0, offset, offset + width), .{ .bits = p.bits, .group_size = p.group });
                        initialized += 1;
                        offset += width;
                    }
                    const compatible = try Linear.prepareSimdGroup(&kernels, members[0..initialized]);
                    if (compatible != case.scalar_ok) return error.StackCalibrationMismatch;
                    try kernels.affine.checked.put(mx.allocator, p.key(), false);
                    if (try Linear.prepareSimdGroup(&kernels, members[0..initialized])) return error.StackFallbackMismatch;
                    offset = 0;
                    for (members[0..initialized]) |member| {
                        const want = try s.slice(if (compatible) expected else fallback, 1, offset, offset + member.n);
                        try equal(&s, try s.reshape(try member.simdBitsRows(&kernels, &s, x, p.key()[4], compatible), mx.shape(want)), want);
                        const fallback_part = try s.slice(fallback, 1, offset, offset + member.n);
                        try equal(&s, try s.reshape(try member.simdBitsRows(&kernels, &s, x, p.key()[4], false), mx.shape(fallback_part)), fallback_part);
                        offset += member.n;
                        calls += 2;
                    }
                    _ = kernels.affine.checked.remove(p.key());
                }
            }
            // Physical SIMD group counts must preserve the calibrated reduction.
            for ([_]i32{ 1, 2, 4, 8, 16 }) |groups| {
                if (groups > mx.simd_groups) continue;
                try @import("sampling_checks.zig").equal(&s, try dense.launch(&kernels, &s, x, p, false, groups), expected);
                calls += 1;
            }
            calls += 1;
        }
    }
    std.debug.print("PASS: {d} calibrated SIMD projections and {d} row/group dispatches match upstream bit for bit.\n", .{ cases.value.len, calls });
}
