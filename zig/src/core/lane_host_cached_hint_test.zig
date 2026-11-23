//! A resident-prefix estimate changes which fill starts. The backend still owns restoration and usage.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fill = @import("lane_host_fill.zig");
const ends = @import("lane_host_end.zig");
const Host = @import("lane_host.zig").LaneHost;
const a = std.testing.allocator;
const prefix_rows = 8_192;
const suffix_rows = 128;

const Fixture = struct {
    prefix: std.ArrayList(u32) = .empty,
    resident: bool = true,
    cached: [2]?u32 = .{ null, null },

    fn fromTarget(ptr: *anyopaque) *Fixture {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @ptrCast(@alignCast(target.prefill_hook_ctx.?));
    }

    fn hint(ptr: *anyopaque, s: *const lanes.Stream) u32 {
        const f = fromTarget(ptr);
        if (!f.resident or s.prompt_len < prefix_rows) return 0;
        if (!std.mem.eql(u32, f.prefix.items, s.prompt()[0..prefix_rows])) return 0;
        return prefix_rows;
    }

    fn prefill(ptr: *anyopaque, s: *lanes.Stream, max_rows: u32) !bool {
        const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const f = fromTarget(ptr);
        const actual = hint(ptr, s);
        if (actual > 0) {
            // The fake target reads these saved token rows in seed() and reports the number actually restored.
            s.reuse.at = actual;
            s.reuse.saved = &f.prefix;
            target.prefill_chunks = 1;
        }
        defer target.prefill_chunks = 16;
        return target.stepped().vtable.prefill_step.?(ptr, s, max_rows);
    }

    fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
        const f: *Fixture = @ptrCast(@alignCast(ptr));
        if (e.* == .prefilled) f.cached[id - 1] = e.prefilled;
    }
};

fn run(evict: bool) !void {
    const cold = try a.alloc(u32, 4_096);
    defer a.free(cold);
    @memset(cold, 3);
    const warm = try a.alloc(u32, prefix_rows + suffix_rows);
    defer a.free(warm);
    @memset(warm, 4);
    var f: Fixture = .{};
    try f.prefix.appendSlice(a, warm[0..prefix_rows]);
    defer f.prefix.deinit(a);
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 16, .prefill_hook_ctx = &f };
    defer target.deinit();
    var vt = target.stepped().vtable.*;
    vt.prefill_cached_hint = Fixture.hint;
    vt.prefill_step = Fixture.prefill;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, .{ .ptr = &target, .vtable = &vt }, clock.clock());
    defer core.deinit();
    var host = Host.init(a, std.testing.io, &core, .{ .lanes = 2 });
    defer {
        ends.closeAll(&host);
        host.stop();
    }
    const requests = [_]api.Request{
        .{ .prompt = cold, .max_tokens = 1, .drafts = false },
        .{ .prompt = warm, .max_tokens = 1, .drafts = false },
    };
    for (&requests, 1..) |*r, id| {
        try host.engine().submit(@intCast(id), r, .{ .ctx = &f, .event = Fixture.event });
        try std.testing.expect(host.admitOne());
    }
    const first = fill.next(&host);
    try std.testing.expectEqual(@as(api.Id, 2), host.filling.items[first].id);
    try std.testing.expectEqual(@as(u32, 0), host.filling.items[first].stream.cached);
    try std.testing.expectEqual(@as(u32, 0), host.filling.items[first].stream.fill_at.load(.monotonic));

    // Sustained shorter choices still yield to a cold request at its existing aging boundary.
    host.filling.items[0].fill_wait = fill.preference_rounds;
    try std.testing.expectEqual(@as(api.Id, 1), host.filling.items[fill.next(&host)].id);
    const warm_first = fill.next(&host);
    try std.testing.expectEqual(@as(api.Id, 2), host.filling.items[warm_first].id);
    if (evict) f.resident = false;
    _ = fill.step(&host, warm_first);
    if (evict) {
        // A cold first step leaves more than 4096 rows. The previous estimate must stop influencing rank.
        try std.testing.expectEqual(@as(api.Id, 1), host.filling.items[fill.next(&host)].id);
        try std.testing.expectEqual(@as(u32, 0), host.filling.items[1].stream.cached);
    }
    while (host.filling.items.len > 0) _ = fill.step(&host, fill.next(&host));
    try std.testing.expectEqual(@as(?u32, 0), f.cached[0]);
    try std.testing.expectEqual(@as(?u32, if (evict) 0 else prefix_rows), f.cached[1]);
}

test "resident prefix hint: a short warm suffix gets the first fill before a smaller cold prompt" {
    try run(false);
}

test "resident prefix hint: eviction before restore reprices the fill and reports zero cached" {
    try run(true);
}
