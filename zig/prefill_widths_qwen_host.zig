//! The real Qwen CUDA prompt loop over a recording engine whose forward finishes before returning.
const std = @import("std");
const q = @import("qwen35");
const lanes = @import("lanes");
const cuda = @import("cuda");
const Driver = @import("prefill_widths_cuda_fake.zig").Driver;
const t = std.testing;
const Cache = struct { delta: i64 = 0 };
const Engine = struct {
    fail: bool = false,
    stream: ?cuda.Stream = null,
    pub fn forward(e: *@This(), _: anytype, _: []const u32, _: q.engine.Keep) !void {
        if (e.fail) return error.PlantedForward;
        if (e.stream) |stream| try stream.synchronize();
    }
};
const Backend = struct {
    gpa: std.mem.Allocator = t.allocator,
    e: *Engine,
    cache: Cache = .{},
    prefix: ?struct {
        pub fn keep(_: @This(), _: *Cache, _: []const u32, _: usize) !void {}
    } = null,
    pages: ?struct {
        pub fn log(_: @This(), _: usize, _: u32) void {}
    } = null,
    pub fn cacheOf(b: *@This(), _: *lanes.Stream) !*Cache {
        return &b.cache;
    }
};
test "CUDA prefill widths: Qwen target cold cuts, one pass, resumed cuts and a failed forward" {
    const cap = q.engine.max_rows;
    const ids = try t.allocator.alloc(u32, 2 * cap + 17);
    defer t.allocator.free(ids);
    @memset(ids, 1);
    for ([_]usize{ 0, cap, 0 }, 0..) |from, case| {
        var d = Driver.init();
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
        var b: Backend = .{ .e = &e };
        const prompt = if (case == 2) ids[0..7] else ids;
        var s = try lanes.Stream.init(t.allocator, .{ .id = "qwen", .prompt = prompt, .chunks = &.{13}, .max_new = 1 });
        defer s.deinit(t.allocator);
        var f: q.prefill.Fill = .{ .at = from, .media = null };
        while (!try q.prefill.chunk(&b, &s, &f, 0)) {}
        const expected: []const u32 = if (case == 2) &.{7} else if (from == 0) &.{ 13, cap, cap, 4 } else &.{ cap, 17 };
        try t.expectEqualSlices(u32, expected, s.prefill_widths.items);
        try t.expectEqual(expected.len, Driver.calls);
    }
    var e: Engine = .{ .fail = true };
    var b: Backend = .{ .e = &e };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "failed", .prompt = &.{1}, .max_new = 1 });
    defer s.deinit(t.allocator);
    var f: q.prefill.Fill = .{ .at = 0, .media = null };
    try t.expectError(error.PlantedForward, q.prefill.chunk(&b, &s, &f, 0));
    try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
}

test "CUDA prefill widths: Qwen failed completion wait records no target pass" {
    var d = Driver.init();
    var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
    var b: Backend = .{ .e = &e };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "wait", .prompt = &.{1}, .max_new = 1 });
    defer s.deinit(t.allocator);
    var f: q.prefill.Fill = .{ .at = 0, .media = null };
    Driver.fail = true;
    try t.expectError(error.NotReady, q.prefill.chunk(&b, &s, &f, 0));
    try t.expectEqual(@as(usize, 0), f.at);
    try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
}
