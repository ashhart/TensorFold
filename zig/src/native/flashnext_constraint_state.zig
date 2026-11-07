//! Borrowed model views and shape arguments return before the ordinary driver can take its next request.
const std = @import("std");
const keys = [_][]const u8{ "Kc_shape", "IDS_shape" };

pub fn Scope(comptime Engine: type, comptime Buffer: type) type {
    const Model = @typeInfo(@FieldType(Engine, "m")).pointer.child;
    const Run = @typeInfo(@FieldType(Engine, "r")).pointer.child;
    return struct {
        model: Model,
        gdn: @FieldType(Run, "gdn_pipe"),
        shapes: [keys.len]?Buffer,
        pub fn capture(e: *Engine) @This() {
            var s: @This() = .{ .model = e.m.*, .gdn = e.r.gdn_pipe, .shapes = undefined };
            for (keys, &s.shapes) |key, *shape| shape.* = e.r.shapes.get(key);
            return s;
        }
        pub fn restore(s: @This(), e: *Engine) void {
            for (keys, s.shapes) |key, shape| {
                if (shape) |buffer| e.r.shapes.getPtr(key).?.* = buffer else _ = e.r.shapes.remove(key);
            }
            e.m.* = s.model;
            e.r.gdn_pipe = s.gdn;
            e.hostMode();
        }
    };
}

test "Flash Next constraint: all borrowed views and shape arguments return on an error" {
    const Model = struct { pos: usize, input: u32, weights: [2]u32, rows: [3]u32 };
    const Run = struct { shapes: std.StringHashMapUnmanaged(u32) = .empty, gdn_pipe: ?u32 = null };
    const Engine = struct {
        m: *Model,
        r: *Run,
        host_calls: usize = 0,
        pub fn hostMode(e: *@This()) void {
            e.host_calls += 1;
        }
        fn fail(e: *@This()) !void {
            const scope = Scope(@This(), u32).capture(e);
            defer scope.restore(e);
            e.m.* = .{ .pos = 99, .input = 100, .weights = .{ 101, 102 }, .rows = .{ 103, 104, 105 } };
            e.r.gdn_pipe = 7;
            try e.r.shapes.put(std.testing.allocator, "Kc_shape", 200);
            try e.r.shapes.put(std.testing.allocator, "IDS_shape", 201);
            return error.GpuFailed;
        }
    };
    const original: Model = .{ .pos = 12, .input = 21, .weights = .{ 31, 32 }, .rows = .{ 41, 42, 43 } };
    var model = original;
    var run: Run = .{};
    defer run.shapes.deinit(std.testing.allocator);
    try run.shapes.put(std.testing.allocator, "Kc_shape", 1);
    var engine: Engine = .{ .m = &model, .r = &run };
    try std.testing.expectError(error.GpuFailed, engine.fail());
    try std.testing.expectEqualDeep(original, model);
    try std.testing.expectEqual(@as(?u32, 1), run.shapes.get("Kc_shape"));
    try std.testing.expect(run.shapes.get("IDS_shape") == null and run.gdn_pipe == null);
    try run.shapes.put(std.testing.allocator, "IDS_shape", 2);
    run.gdn_pipe = 3;
    try std.testing.expectError(error.GpuFailed, engine.fail());
    try std.testing.expectEqual(@as(?u32, 2), run.shapes.get("IDS_shape"));
    try std.testing.expectEqual(@as(?u32, 3), run.gdn_pipe);
    try std.testing.expectEqualDeep(original, model);
    try std.testing.expectEqual(@as(usize, 2), engine.host_calls);
}
