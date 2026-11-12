//! The ledger's reasons in the scrape and /v1/load: a fixed label set, the shared names the client codes use.
const std = @import("std");
const api = @import("engine_api");
const wait = @import("wait_reason.zig");
const load_metrics = @import("load_metrics.zig");
const ledger = api.load_ledger;
const reasons = ledger.reasons;

fn snapshot() ledger.Snapshot {
    return .{ .instance = @splat('a'), .start_s = 1793200000, .parked = false, .lanes = 2, .http = @splat(0) };
}

/// The label values `family` carries in `body`, each line's value checked to be 0.
fn labels(body: []const u8, family: []const u8, out: *std.ArrayList([]const u8)) !void {
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| {
        const head = "tensorfold:";
        if (!std.mem.startsWith(u8, line, head) or !std.mem.startsWith(u8, line[head.len..], family)) continue;
        const rest = line[head.len + family.len ..];
        if (rest.len == 0 or rest[0] != '{') continue; // another family that starts with the same name
        try std.testing.expect(std.mem.startsWith(u8, rest, "{reason=\""));
        const end = std.mem.indexOf(u8, rest, "\"} ").?;
        try std.testing.expectEqualStrings("0", rest[end + 3 ..]);
        try out.append(std.testing.allocator, rest["{reason=\"".len..end]);
    }
}

test "wait reason ledger: a fresh host's scrape shows every wait reason at 0 and no other label" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    var s = snapshot();
    s.requests = .{}; // what a fresh host publishes: nothing queued
    try load_metrics.render(&out.writer, s, .{ .model = "m" });
    var seen: std.ArrayList([]const u8) = .empty;
    defer seen.deinit(std.testing.allocator);
    try labels(out.written(), "requests_waiting_by_reason", &seen);
    try std.testing.expectEqual(std.enums.values(reasons.Wait).len, seen.items.len);
    for (std.enums.values(reasons.Wait), seen.items) |r, label| try std.testing.expectEqualStrings(@tagName(r), label);
    seen.clearRetainingCapacity();
    try labels(out.written(), "request_refusals_total", &seen);
    try std.testing.expectEqual(@as(usize, 4), seen.items.len);
    seen.clearRetainingCapacity();
    try labels(out.written(), "request_failures_total", &seen);
    try std.testing.expectEqual(std.enums.values(reasons.Failure).len, seen.items.len);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "tensorfold:restore_fallbacks_total 0\n") != null);
}

test "wait reason ledger: the capacity codes are the internal reasons' and the refusal counter's names" {
    const shared = [_][]const u8{ wait.engine_busy, wait.engine_unavailable, wait.memory_pressure };
    for (shared, [_]reasons.Wait{ .engine_busy, .engine_unavailable, .memory_pressure }) |code, r| try std.testing.expectEqualStrings(code, @tagName(r));
    for (shared, [_]reasons.Refusal{ .engine_busy, .engine_unavailable, .memory_pressure }) |code, r| try std.testing.expectEqualStrings(code, @tagName(r));
    for (wait.all) |w| try std.testing.expect(std.meta.stringToEnum(reasons.Wait, w.code) == null); // queues nothing
}

test "wait reason ledger: /v1/load names each queued reason and each refusal" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var s = snapshot();
    s.refusals[@backingInt(reasons.Refusal.memory_pressure)] = 2;
    s.requests = .{ .waiting = 3 };
    s.requests.?.waiting_by_reason[@backingInt(reasons.Wait.kv_pool)] = 2;
    s.requests.?.waiting_by_reason[@backingInt(reasons.Wait.memory_floor)] = 1;
    const body = try @import("json").stringify(a, try load_metrics.object(a, s, 1793200001.5, "m"), .{});
    const flat = try std.mem.replaceOwned(u8, a, body, " ", "");
    for ([_][]const u8{ "\"waiting\":3", "\"kv_pool\":2", "\"memory_floor\":1", "\"memory_pressure\":2", "\"invalid_request\":0", "\"waiting_by_reason\":{", "\"refusals\":{" }) |key| {
        if (std.mem.indexOf(u8, flat, key) == null) std.debug.print("missing {s} in {s}\n", .{ key, flat });
        try std.testing.expect(std.mem.indexOf(u8, flat, key) != null);
    }
}
