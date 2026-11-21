//! E24 controls for the shared-cut wiring and the native admission/eviction replay.
const std = @import("std");
const trace = @import("cache_agent_trace.zig");
const run = @import("cache_replay_run.zig");
const t = std.testing;

test "agent cache replay: native and selector adapter agree per request on long shared-system traces" {
    for (std.enums.values(trace.Shape)) |shape| {
        var requests = try trace.make(t.allocator, shape);
        defer requests.deinit();
        for (trace.budgets) |budget| {
            var native: run.Replay = undefined;
            native.init(t.allocator, .main, budget, .{});
            defer native.deinit();
            var adapter: run.Replay = undefined;
            adapter.init(t.allocator, .main, budget, .{});
            adapter.adapter_main = true;
            defer adapter.deinit();
            for (requests.requests) |request| {
                try t.expectEqualDeep(try native.request(request), try adapter.request(request));
                try t.expectEqualDeep(native.store.counts, adapter.store.counts);
            }
        }
    }
}

test "agent cache replay: shared system state serves the next different session and budget accounting is exact" {
    var requests = try trace.make(t.allocator, .agent_loops);
    defer requests.deinit();
    var replay: run.Replay = undefined;
    replay.init(t.allocator, .main, 16 << 30, .{});
    defer replay.deinit();
    _ = try replay.request(requests.requests[0]);
    const next = try replay.request(requests.requests[3]);
    try t.expectEqual(@as(u32, 8192), next.from);
    try t.expectEqualSlices(u32, &.{8192}, requests.requests[3].shared);
    try t.expect(next.held <= replay.store.budget);
}
