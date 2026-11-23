//! Nemotron's served completion call over a recording engine and the existing CUDA stream wait.
const std = @import("std");
const n = @import("nemotron");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Driver = @import("prefill_widths_cuda_fake.zig").Driver;
const t = std.testing;
const Engine = struct {
    gpa: std.mem.Allocator = t.allocator,
    stream: cuda.Stream,
    last_chunk: usize = 0,
    pos: usize = 0,
    expected_chunks: []const u32 = &.{},
    fail: bool = false,
    cancel: bool = false,
    pub fn prefillFrom(e: *@This(), ids: []const u32, _: usize, _: anytype, _: anytype, _: anytype, _: anytype, chunks: []const u32) !u32 {
        try t.expectEqualSlices(u32, e.expected_chunks, chunks);
        if (e.fail) return error.PlantedForward;
        if (e.cancel) return error.Cancelled;
        try e.stream.synchronize(); // cuda_prompt.prefillFrom downloads the first token, then waits.
        e.pos = ids.len;
        return 1;
    }
};
test "CUDA prefill widths: Nemotron one target pass and resumed pass report only rows run" {
    const ids: [n.state.pass_rows + 7]u32 = @splat(1);
    for ([_]usize{ 0, n.state.pass_rows, 0 }, 0..) |from, case| {
        var d = Driver.init();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "nemo", .prompt = if (case == 0) ids[0..7] else if (case == 2) ids[0..n.state.pass_rows] else &ids, .max_new = 1 });
        defer s.deinit(t.allocator);
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null }, .last_chunk = from, .pos = from };
        try t.expectEqual(@as(u32, 1), try n.Lanes.servedPrefill(&e, null, &s, from, null));
        try t.expectEqualSlices(u32, &.{if (case == 2) n.state.pass_rows else 7}, s.prefill_widths.items);
        try t.expectEqual(@as(usize, 1), Driver.calls);
    }
}
test "CUDA prefill widths: Nemotron multi-chunk serial, background and segmented calls stay unrecorded" {
    const ids: [2 * n.state.pass_rows + 17]u32 = @splat(1);
    for ([_]usize{ 0, n.state.pass_rows, 0 }, 0..) |from, case| {
        var d = Driver.init();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "skipped", .prompt = &ids, .max_new = 1 });
        defer s.deinit(t.allocator);
        const last = if (case == 1) ids.len - 1 else ids.len - 17;
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null }, .last_chunk = last, .pos = from };
        _ = try n.Lanes.servedPrefill(&e, null, &s, from, null);
        try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
        try t.expectEqual(@as(usize, 1), Driver.calls);
    }
}
test "CUDA prefill widths: Nemotron failed launch, readback and cancellation append nothing" {
    for (0..3) |case| {
        var d = Driver.init();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "failed", .prompt = &.{1}, .max_new = 1 });
        defer s.deinit(t.allocator);
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null }, .fail = case == 0, .cancel = case == 2 };
        Driver.fail = case == 1;
        const result = n.Lanes.servedPrefill(&e, null, &s, 0, null);
        if (case == 0) try t.expectError(error.PlantedForward, result) else if (case == 1) try t.expectError(error.NotReady, result) else try t.expectError(error.Cancelled, result);
        try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
    }
}

test "CUDA prefill widths: Nemotron forwards named starts from the actual stream" {
    var d = Driver.init();
    const points = [_]u32{ 1, 2, 3 };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "starts", .prompt = &.{ 1, 2, 3, 4 }, .chunks = &points, .max_new = 1 });
    defer s.deinit(t.allocator);
    var e: Engine = .{ .stream = .{ .d = &d, .handle = null }, .expected_chunks = &points };
    _ = try n.Lanes.servedPrefill(&e, null, &s, 0, null);
}
