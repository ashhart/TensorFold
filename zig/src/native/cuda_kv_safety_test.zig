//! Startup's actual room, page plan, kept budget and store configuration, with allocation counters.
const std = @import("std");
const api = @import("engine_api");
const F = struct {
    pub const snap_save = 1;
    pub fn snap_slots(n: *usize, _: usize) !u64 {
        n.* += 1;
        return 8 << 20;
    }
};
pub fn run(comptime planner: type, disabled: bool) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var context: u8 = 0;
    const loaded = .{ .ctx = @as(*anyopaque, &context), .host_snaps = true, .snap_bytes = @as(u64, 8 << 20) };
    const layout: @import("core").kv_pool.Layout = .{ .fixed_bytes = 64, .token_bytes = 4, .page_tokens = 512, .snap_bytes = 8 << 20 };
    const o: api.Open = .{ .dir = "", .model_type = "n", .lanes = 1, .keep = if (disabled) 0 else null, .prompt_cache_gib = 1 };
    for ([_]u64{ 4 << 20, 8 << 20, 12 << 20 }) |available| {
        var problem: []const u8 = "";
        var allocations: usize = 0;
        const left = try planner.startupRoom(F, arena.allocator(), loaded, o, available, 1 << 30, &problem);
        if (left) |room| {
            const plan = (try planner.plan(F, arena.allocator(), layout, room, o, 1024, &problem, true, planner.enginePaged(loaded))).?;
            const kept = planner.kept(F, loaded, plan, o, room, 0, 1 << 30, planner.host(F, loaded));
            var store: api.prompt_cache.Store = undefined;
            if (kept.copied > 0) try planner.store(F, &store, &allocations, kept);
        }
        if (!disabled and available < 8 << 20) {
            try std.testing.expect(left == null);
            try std.testing.expect(std.mem.indexOf(u8, problem, "staging") != null);
            try std.testing.expectEqual(@as(usize, 0), allocations);
        } else {
            try std.testing.expectEqual(available - if (disabled) @as(u64, 0) else 8 << 20, left.?);
            try std.testing.expectEqual(@as(usize, if (disabled) 0 else 1), allocations);
        }
    }
}
