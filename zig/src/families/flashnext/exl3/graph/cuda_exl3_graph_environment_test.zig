//! The graph settings reach every consumer through platform/environment.zig, which refuses text it does not accept.
const std = @import("std");
const cuda = @import("cuda");
const env = @import("sys").env;
const hub = @import("cuda_exl3_graph.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const memory = @import("cuda_exl3_graph_memory.zig");

const SavedEnv = struct {
    name: [:0]const u8,
    value: ?[:0]u8,

    fn take(name: [:0]const u8) !SavedEnv {
        const old = if (@import("sys").getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, v, 0) else null;
        return .{ .name = name, .value = old };
    }

    fn set(s: SavedEnv, value: ?[:0]const u8) !void {
        if (value) |v| try std.testing.expectEqual(@as(c_int, 0), env.setenv(s.name, v, 1)) else _ = env.unsetenv(s.name);
    }

    fn restore(s: SavedEnv) void {
        if (s.value) |v| {
            _ = env.setenv(s.name, v, 1);
            std.testing.allocator.free(v);
        } else _ = env.unsetenv(s.name);
    }
};

test "exl3: GRAPH accepts its explicit modes and refuses other text" {
    const saved = try SavedEnv.take("TF_FLASHNEXT_EXL3_GRAPH");
    defer saved.restore();
    const previous = hub.force;
    hub.force = null;
    defer hub.force = previous;
    const cases = [_]struct { ?[:0]const u8, hub.Mode }{ .{ null, .off }, .{ "", .off }, .{ " ", .off }, .{ "0", .off }, .{ "eager", .dplan }, .{ "1", .graph } };
    for (cases) |case| {
        try saved.set(case[0]);
        try std.testing.expectEqual(case[1], try hub.mode());
    }
    for ([_][:0]const u8{ "2", "true", "Eager", " eager", "1 ", "graph", "01" }) |text| {
        try saved.set(text);
        try std.testing.expectError(error.InvalidEnvironment, hub.mode());
    }
}

test "exl3: malformed GRAPH_ASYNC refuses before a graph is taken" {
    const saved = try SavedEnv.take("TF_FLASHNEXT_EXL3_GRAPH_ASYNC");
    defer saved.restore();
    var driver: cuda.Driver = undefined;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(1) };
    for ([_][:0]const u8{ "2", "off", "true", " 0", "00" }) |text| {
        try saved.set(text);
        try std.testing.expectError(error.InvalidEnvironment, gpending.startInstantiate(graph, true));
    }
}

test "exl3: malformed graph budgets refuse before workload construction" {
    const streams = try SavedEnv.take("TF_FLASHNEXT_EXL3_STREAMS");
    defer streams.restore();
    const window = try SavedEnv.take("TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK");
    defer window.restore();
    try window.set(null);
    for ([_][:0]const u8{ "x", "-1", "1.5", "2k" }) |text| {
        try streams.set(text);
        try std.testing.expectError(error.InvalidEnvironment, memory.policyFromEnv());
    }
    try streams.set("2");
    try window.set("2");
    try std.testing.expectError(error.InvalidEnvironment, memory.policyFromEnv());
}
