//! Serial image sequences, reported seeds, solo byte equality and early termination through the actual route.
const std = @import("std");
const h = @import("image_test_harness.zig");
const json = @import("json");
const t = std.testing;

fn body(a: std.mem.Allocator, count: usize, seed: ?u64) ![]const u8 {
    return std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":\"hi\",\"negative_prompt\":\"fog\",\"size\":\"512x768\",\"steps\":8,\"guidance\":2,\"n\":{d}{s}}}", .{ count, if (seed) |s| try std.fmt.allocPrint(a, ",\"seed\":{d}", .{s}) else "" });
}
fn seedOf(item: json.Value) !u64 {
    const value = item.get("seed") orelse return error.MissingSeed;
    return std.fmt.parseInt(u64, value.int, 10);
}

fn sequence(count: usize, seed: ?u64) !void {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stub: h.Stub = .{};
    const reply = try h.ask(a, &stub, try body(a, count, seed));
    try t.expectEqual(@as(u16, 200), reply.status);
    const data = reply.value.get("data").?.array;
    try t.expectEqual(count, data.len);
    try t.expectEqual(count, stub.calls);
    try t.expectEqual(count, stub.finished);
    try t.expectEqual(@as(usize, 1), stub.maximum_active);
    for (data, 0..) |item, index| {
        const reported = try seedOf(item);
        try t.expectEqual(stub.seeds[index], reported);
        try t.expectEqual((seed orelse 42) + @as(u64, @intCast(index)), reported);
        var solo: h.Stub = .{};
        const single = try h.ask(a, &solo, try body(a, 1, reported));
        try t.expectEqual(@as(u16, 200), single.status);
        try t.expectEqualStrings(item.strField("b64_json").?, single.value.get("data").?.array[0].strField("b64_json").?);
    }
}

test "image sequence: n from one through four is serial, ordered and byte-equal to each solo seed" {
    for (1..5) |count| try sequence(count, 9007199254740993);
}

test "image sequence: omitted seeds are reported exactly and each picture replays alone" {
    try sequence(4, null);
}

test "image sequence: full-width single seed is exact and a multi-image overflow refuses before submission" {
    try sequence(1, std.math.maxInt(u64));
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stub: h.Stub = .{};
    const out = try h.ask(a, &stub, try body(a, 2, std.math.maxInt(u64)));
    try t.expectEqual(@as(u16, 400), out.status);
    try t.expect(std.mem.indexOf(u8, out.value.get("error").?.strField("message").?, "seed plus n - 1") != null);
    try t.expectEqual(@as(usize, 0), stub.calls);
}

test "image sequence: invalid counts refuse by name before the engine sees them" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "0", "5", "-1", "true", "1.0", "null", "\"2\"" }) |count| {
        var stub: h.Stub = .{};
        const request = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":\"hi\",\"n\":{s}}}", .{count});
        const out = try h.ask(a, &stub, request);
        try t.expectEqual(@as(u16, 400), out.status);
        try t.expect(std.mem.indexOf(u8, out.value.get("error").?.strField("message").?, "n must be from 1 to 4") != null);
        try t.expectEqual(@as(usize, 0), stub.calls);
    }
}

test "image sequence: noninteger or out-of-range seeds refuse before submission" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "-1", "true", "1.0", "null", "\"2\"", "18446744073709551616" }) |seed| {
        var stub: h.Stub = .{};
        const request = try std.fmt.allocPrint(a, "{{\"model\":\"m\",\"prompt\":\"hi\",\"seed\":{s}}}", .{seed});
        const out = try h.ask(a, &stub, request);
        try t.expectEqual(@as(u16, 400), out.status);
        try t.expect(std.mem.indexOf(u8, out.value.get("error").?.strField("message").?, "seed") != null);
        try t.expectEqual(@as(usize, 0), stub.calls);
    }
}

test "image sequence: a refusal or failure on the second picture discards partial results and stops" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { stub: h.Stub, status: u16, message: []const u8 }{
        .{ .stub = .{ .reject_at = 2 }, .status = 400, .message = "refused" },
        .{ .stub = .{ .reject_at = 2, .reject = error.Busy }, .status = 503, .message = "busy" },
        .{ .stub = .{ .fail_at = 2 }, .status = 500, .message = "fake image pass failed" },
        .{ .stub = .{ .cancel_at = 2 }, .status = 503, .message = "cancelled" },
    }) |case| {
        var stub = case.stub;
        const out = try h.ask(a, &stub, try body(a, 4, 7));
        try t.expectEqual(case.status, out.status);
        try t.expect(out.value.get("data") == null and out.value.get("error") != null);
        try t.expect(std.mem.indexOf(u8, out.value.get("error").?.strField("message").?, case.message) != null);
        try t.expectEqual(@as(usize, 2), stub.calls);
        try t.expectEqual(@as(usize, 0), stub.active);
        try t.expect(stub.pending == null);
    }
}

test "image sequence: a disconnected second picture cancels and settles before request storage is released" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stub: h.Stub = .{ .hold_at = 2, .leave_at = 2 };
    _ = try h.ask(a, &stub, try body(a, 4, 7));
    try t.expectEqual(@as(usize, 2), stub.calls);
    try t.expectEqual(@as(usize, 2), stub.finished);
    try t.expectEqual(@as(usize, 1), stub.cancels);
    try t.expectEqual(@as(usize, 0), stub.active);
    try t.expect(stub.pending == null);
}

test "image sequence: a disconnect between pictures never submits the next one" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var stub: h.Stub = .{ .leave_at = 1 };
    _ = try h.ask(a, &stub, try body(a, 4, 7));
    try t.expectEqual(@as(usize, 1), stub.calls);
    try t.expectEqual(@as(usize, 1), stub.finished);
    try t.expectEqual(@as(usize, 0), stub.cancels);
}
