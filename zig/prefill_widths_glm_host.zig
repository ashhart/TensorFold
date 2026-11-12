//! GLM's actual served step and batched-pass bodies over recording leader calls and real CUDA stream waits.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Driver = @import("prefill_widths_cuda_fake.zig").Driver;
const t = std.testing;
const Lane = struct { slot: u32 = 0, first: ?u64 = null, fill: ?usize = null };
const Leader = struct {
    e: *Engine,
    fail: bool = false,
    rows: usize = glm.engine.prompt_rows,
    pub fn processors(_: @This(), _: *const u8) !void {}
    pub fn bind(_: @This(), _: u32) !void {}
    pub fn chunk(x: @This(), ids: []const u32, at: usize, _: bool, _: anytype, _: anytype, _: anytype) !glm.follow.Chunk {
        if (x.fail) return error.PlantedForward;
        const rows = @min(x.rows, ids.len - at);
        const last = at + rows == ids.len;
        if (last) try x.e.stream.synchronize(); // Engine.prefillChunk draws the final row before returning.
        return .{ .rows = rows, .first = if (last) 1 else null };
    }
    pub fn prefillMany(x: @This(), _: []const glm.engine.Prompt, tokens: []u32) !void {
        if (x.fail) return error.PlantedForward;
        try x.e.stream.synchronize(); // Engine.prefillMany draws each stream before returning.
        @memset(tokens, 1);
    }
};
const Engine = struct {
    gpa: std.mem.Allocator = t.allocator,
    stream: cuda.Stream,
    prompt_logprobs: ?glm.prompt_logprobs.Want = null,
};
const Backend = struct {
    x: Leader,
    from: usize = 0,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    processor_stage: u8 = 0,
    pub fn resetProcessors(b: *@This()) !*const u8 {
        return &b.processor_stage;
    }
    pub fn addProcessor(_: *@This(), s: *const lanes.Stream, _: u64, _: []const u32, _: []const u32) !void {
        try t.expectEqual(@as(f32, 1), s.penalties.repetition);
    }
    pub fn promptProcessor(_: *@This(), s: *const lanes.Stream) !void {
        try t.expectEqual(@as(f32, 1), s.penalties.repetition);
    }
    pub fn opening(b: *@This(), _: *lanes.Stream) !?glm.engine.Resume {
        return if (b.from > 0) .{ .id = 1, .at = b.from } else null;
    }
    pub fn images(_: *@This(), _: *lanes.Stream, _: anytype) !void {}
    pub fn marksOf(_: *@This(), _: *lanes.Stream, _: anytype) ?glm.engine.Marks {
        return null;
    }
    pub fn take(_: *@This(), token: u32) u64 {
        return token;
    }
};
test "CUDA prefill widths: GLM stepped cold, solo, one-pass and resumed real cuts" {
    const cap = glm.engine.prompt_rows;
    const ids = try t.allocator.alloc(u32, 2 * cap + 17);
    defer t.allocator.free(ids);
    @memset(ids, 1);
    for ([_]usize{ 0, cap, 0, 0 }, 0..) |from, case| {
        var d = Driver.init();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "glm", .prompt = if (case == 2) ids[0..7] else ids, .max_new = 1 });
        defer s.deinit(t.allocator);
        s.settle_step = true;
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
        var b: Backend = .{ .x = .{ .e = &e, .rows = if (case == 3) glm.engine.solo_rows else cap }, .from = from };
        defer b.lanes.deinit(t.allocator);
        try b.lanes.put(t.allocator, &s, .{});
        while (!try glm.lanes.Lanes.prefillStep(&b, &s)) {}
        const expected: []const u32 = if (case == 2) &.{7} else if (case == 3) &.{ glm.engine.solo_rows, 17 } else if (from == 0) &.{ cap, cap, 17 } else &.{ cap, 17 };
        try t.expectEqualSlices(u32, expected, s.prefill_widths.items);
        try t.expectEqual(expected.len, Driver.calls);
        try t.expectEqual(@as(?u64, 1), b.lanes.get(&s).?.first);
    }
}
test "CUDA prefill widths: GLM skips unsettled intermediate chunks" {
    const ids: [2 * glm.engine.prompt_rows + 17]u32 = @splat(1);
    var d = Driver.init();
    var s = try lanes.Stream.init(t.allocator, .{ .id = "unsettled", .prompt = &ids, .max_new = 1 });
    defer s.deinit(t.allocator);
    var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
    var b: Backend = .{ .x = .{ .e = &e } };
    defer b.lanes.deinit(t.allocator);
    try b.lanes.put(t.allocator, &s, .{});
    for (0..2) |_| {
        try t.expect(!try glm.lanes.Lanes.prefillStep(&b, &s));
        try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
        try t.expectEqual(@as(usize, 0), Driver.calls);
    }
    try t.expect(try glm.lanes.Lanes.prefillStep(&b, &s));
    try t.expectEqualSlices(u32, &.{17}, s.prefill_widths.items);
    try t.expectEqual(@as(usize, 1), Driver.calls);
}
test "CUDA prefill widths: GLM failed launch, final readback and intermediate wait append nothing" {
    const ids: [glm.engine.prompt_rows + 1]u32 = @splat(1);
    for (0..3) |case| {
        var d = Driver.init();
        var s = try lanes.Stream.init(t.allocator, .{ .id = "failed", .prompt = if (case == 2) &ids else ids[0..1], .max_new = 1 });
        defer s.deinit(t.allocator);
        s.settle_step = true;
        var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
        var b: Backend = .{ .x = .{ .e = &e, .fail = case == 0 } };
        defer b.lanes.deinit(t.allocator);
        try b.lanes.put(t.allocator, &s, .{});
        Driver.fail = case > 0;
        const result = glm.lanes.Lanes.prefillStep(&b, &s);
        if (case == 0) try t.expectError(error.PlantedForward, result) else try t.expectError(error.NotReady, result);
        try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
    }
}
test "CUDA prefill widths: GLM batched pass credits each stream's own rows once after readback" {
    var d = Driver.init();
    var e: Engine = .{ .stream = .{ .d = &d, .handle = null } };
    var b: Backend = .{ .x = .{ .e = &e } };
    defer b.lanes.deinit(t.allocator);
    var left = try lanes.Stream.init(t.allocator, .{ .id = "left", .prompt = &.{ 1, 2 }, .max_new = 1 });
    defer left.deinit(t.allocator);
    var right = try lanes.Stream.init(t.allocator, .{ .id = "right", .prompt = &.{ 1, 2, 3 }, .max_new = 1 });
    defer right.deinit(t.allocator);
    try b.lanes.put(t.allocator, &left, .{});
    try b.lanes.put(t.allocator, &right, .{});
    var tokens: [2]u32 = undefined;
    const parts = [_]glm.engine.Prompt{ .{ .slot = 0, .ids = left.prompt() }, .{ .slot = 1, .ids = right.prompt() } };
    b.x.fail = true;
    try t.expectError(error.PlantedForward, glm.lanes.Lanes.many(&b, &.{ &left, &right }, &parts, &tokens));
    b.x.fail = false;
    Driver.fail = true;
    try t.expectError(error.NotReady, glm.lanes.Lanes.many(&b, &.{ &left, &right }, &parts, &tokens));
    try t.expectEqual(@as(usize, 0), left.prefill_widths.items.len + right.prefill_widths.items.len);
    Driver.fail = false;
    try glm.lanes.Lanes.many(&b, &.{ &left, &right }, &parts, &tokens);
    try t.expectEqualSlices(u32, &.{2}, left.prefill_widths.items);
    try t.expectEqualSlices(u32, &.{3}, right.prefill_widths.items);
}
