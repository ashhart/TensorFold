//! Authenticated, read-only sizing diagnostics; unknown counters and plans are omitted rather than guessed.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json.zig");
const process = @import("process_memory.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

pub fn get(srv: *Server, conn: *Conn, a: Allocator) !void {
    const value = try snapshot(a, srv.info, process.read(), srv.engine.memory(false));
    routes.sendValue(conn, a, 200, value);
}

fn snapshot(a: Allocator, info: api.Info, proc: ?process.Snapshot, device: ?api.Memory) !json.Value {
    const o = try json.newObject(a);
    try o.put(a, "schema_version", try json.intValue(a, 1));
    if (info.context_window > 0) try o.put(a, "context_window", try json.intValue(a, info.context_window));
    try o.put(a, "context_fitted", .{ .bool = info.context_fitted });
    if (proc) |p| {
        const m = try json.newObject(a);
        try m.put(a, "source", .{ .string = "darwin_rusage_info_v4" });
        try m.put(a, "physical_footprint_bytes", try json.intValue(a, p.physical_footprint_bytes));
        try m.put(a, "lifetime_peak_physical_footprint_bytes", try json.intValue(a, p.lifetime_peak_physical_footprint_bytes));
        try o.put(a, "process", .{ .object = m });
    }
    if (device) |d| {
        const m = try json.newObject(a);
        try m.put(a, "active_bytes", try json.intValue(a, d.active));
        try m.put(a, "cache_bytes", try json.intValue(a, d.cache));
        try m.put(a, "peak_bytes", try json.intValue(a, d.peak));
        try m.put(a, "peak_scope", .{ .string = "since_start_or_health_reset" });
        try o.put(a, "device", .{ .object = m });
    }
    if (info.prompt_cache_plan) |plan| {
        const m = try json.newObject(a);
        try m.put(a, "source", .{ .string = @tagName(plan.source) });
        try m.put(a, "budget_bytes", try json.intValue(a, plan.budget_bytes));
        try m.put(a, "explicit_budget", .{ .bool = plan.explicit_budget });
        if (plan.over_cap) |over| try m.put(a, "over_cap", .{ .bool = over });
        inline for (.{ "ram_bytes", "ready_footprint_bytes", "cap_bytes", "room_bytes", "margin_bytes" }) |field| {
            if (@field(plan, field)) |n| try m.put(a, field, try json.intValue(a, n));
        }
        try o.put(a, "prompt_cache_plan", .{ .object = m });
    }
    return .{ .object = o };
}

test "unknown scopes are omitted and known zero budget remains visible" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const unknown = try snapshot(a, .{}, null, null);
    for ([_][]const u8{ "process", "device", "prompt_cache_plan", "context_window" }) |field|
        try std.testing.expect(unknown.get(field) == null);
    const zero = try snapshot(a, .{ .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = 0, .explicit_budget = true } }, null, null);
    const plan = zero.get("prompt_cache_plan").?;
    try std.testing.expectEqual(@as(i64, 0), plan.get("budget_bytes").?.int64().?);
    try std.testing.expect(plan.get("ram_bytes") == null);
    try std.testing.expect(plan.get("ready_footprint_bytes") == null);
    try std.testing.expect(plan.get("over_cap") == null);
}

test "process lifetime peak and resettable device peak retain independent scopes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try snapshot(arena.allocator(), .{ .context_window = 8192, .context_fitted = true }, .{ .physical_footprint_bytes = 5, .lifetime_peak_physical_footprint_bytes = 12 }, .{ .active = 2, .cache = 1, .peak = 3 });
    try std.testing.expectEqual(@as(i64, 12), value.get("process").?.get("lifetime_peak_physical_footprint_bytes").?.int64().?);
    try std.testing.expectEqual(@as(i64, 3), value.get("device").?.get("peak_bytes").?.int64().?);
    try std.testing.expectEqualStrings("since_start_or_health_reset", value.get("device").?.get("peak_scope").?.string);
}

test "physical cache plan preserves the applied sizing inputs without runtime or identity fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try snapshot(arena.allocator(), .{ .prompt_cache_plan = .{
        .source = .physical_footprint,
        .budget_bytes = 8,
        .explicit_budget = true,
        .over_cap = true,
        .ram_bytes = 100,
        .ready_footprint_bytes = 60,
        .cap_bytes = 70,
        .room_bytes = 6,
        .margin_bytes = 4,
    } }, null, null);
    const plan = value.get("prompt_cache_plan").?;
    try std.testing.expectEqual(@as(i64, 6), plan.get("room_bytes").?.int64().?);
    try std.testing.expect(plan.get("over_cap").?.bool);
    try std.testing.expectEqual(@as(usize, 9), plan.object.count());
    try std.testing.expectEqual(@as(usize, 3), value.object.count());
}
