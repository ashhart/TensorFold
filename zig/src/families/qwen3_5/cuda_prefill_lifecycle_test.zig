//! The actual Drafted backend constructs its final prompt absorb after first-token sampling.
const std = @import("std");
const lanes = @import("lanes");
const cuda = @import("cuda");
const trace = cuda.prefill_attribution;
const adapter = @import("cuda_prefill_trace.zig");
const be = lanes.backend;
const dr = lanes.drafter;
const t = std.testing;
var receipts: [16]trace.ledger.Row = undefined;
var count: usize = 0;
fn sink(row: trace.ledger.Row) void {
    receipts[count] = row;
    count += 1;
}
fn noEvents(_: *cuda.abi.Event, _: c_uint) callconv(.c) i32 {
    return 1;
}
fn first(_: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
    return 11;
}
fn masked(_: *anyopaque, _: *lanes.Stream, _: u64, _: []const u32) anyerror!u64 {
    return 22;
}
fn read(_: *anyopaque, handle: u64) anyerror!u32 {
    return @intCast(handle);
}
fn prepare(_: *anyopaque, _: []const u32) anyerror!void {}
fn features(_: *anyopaque, s: *lanes.Stream, _: []const u32, start: u64, rows: u32) anyerror!be.Features {
    try t.expectEqual(@as(u64, s.prompt_len - 1), start);
    try t.expectEqual(@as(u32, 1), rows);
    return .{ .buffer = 1, .offset = start, .rows = rows, .row_bytes = 4, .space = .device, .dtype = .bf16 };
}
const Draft = struct {
    fns: struct { gather_taps: struct { handle: cuda.abi.Function = null } = .{}, gather_rows: struct { handle: cuda.abi.Function = null } = .{} } = .{},
    absorbed: usize = 0,
    follow: u32 = 0,
    fn taps(_: *anyopaque) []const u32 {
        return &.{};
    }
    fn facts(_: *anyopaque) dr.Facts {
        return .{ .depth = 1, .step_ms = 1 };
    }
    fn open(_: *anyopaque, _: *lanes.Stream) anyerror!void {}
    fn absorb(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d: *@This() = @ptrCast(@alignCast(p));
        try t.expectEqual(@as(usize, 1), items.len);
        try t.expect(items[0].prompt);
        d.absorbed += items[0].features.rows;
        d.follow = items[0].follow[0];
    }
    fn hold(_: *anyopaque, _: []const dr.Hold) anyerror!void {}
    fn held(_: *anyopaque, _: []const *lanes.Stream, _: []const []u32) anyerror!void {}
    fn release(_: *anyopaque, _: *lanes.Stream) void {}
    const table: dr.Drafter.VTable = .{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release };
};
pub fn run(comptime L: type, backend: anytype) !void {
    for ([_]bool{ false, true }) |with_mask| {
        var lane: L = .{ .gpa = t.allocator, .io = t.io, .e = undefined, .capacity = 32, .target = undefined };
        var target = backend(&lane);
        var table = target.vtable.*;
        table.first = first;
        table.first_masked = masked;
        table.read = read;
        table.prepare_features = prepare;
        table.features = features;
        target.vtable = &table;
        var driver: cuda.Driver = undefined;
        driver.api.cuEventCreate = noEvents;
        try trace.Trace.create(t.allocator, t.io, &driver, null, true);
        defer adapter.close();
        const state = trace.owner.?;
        state.sink = sink;
        count = 0;
        var tokens: [9]u32 = @splat(3);
        var stream = try lanes.Stream.init(t.allocator, .{ .id = "first-before-absorb", .prompt = &tokens, .max_new = 1 });
        defer stream.deinit(t.allocator);
        try t.expect(state.enter(&stream, tokens.len, 4096, 8192));
        state.leave();
        var d: Draft = .{};
        const drafter = adapter.wrapDrafter(&d, .{ .ptr = &d, .vtable = &Draft.table });
        var drafted = try lanes.drafted.Drafted.init(t.allocator, adapter.wrapBackend(L, target), drafter);
        defer drafted.deinit();
        try drafted.lanes.put(t.allocator, &stream, .{});
        try drafter.open(&stream);
        const b = drafted.backend();
        const drawn = if (with_mask) try b.vtable.first_masked.?(b.ptr, &stream, tokens.len, &.{1}) else try b.first(&stream, tokens.len);
        try t.expectEqual(@as(u64, if (with_mask) 22 else 11), drawn);
        try b.draftIn(t.allocator, &.{.{ .stream = &stream, .follow = &.{}, .first = .{ .handle = drawn }, .rows = null, .start = tokens.len, .position = tokens.len + 1, .depth = 0 }});
        try t.expectEqual(@as(usize, 1), d.absorbed);
        try t.expectEqual(@as(u32, @intCast(drawn)), d.follow);
        var trailing: usize = 0;
        for (receipts[0..count]) |row| if (row.stage == .trailing_absorb and row.start_token == tokens.len - 1 and row.rows == 1) {
            trailing += 1;
        };
        try t.expectEqual(@as(usize, 1), trailing);
    }
}
