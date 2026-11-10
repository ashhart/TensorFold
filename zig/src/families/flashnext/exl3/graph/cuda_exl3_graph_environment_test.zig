//! The graph settings reach every consumer through platform/environment.zig, which refuses text it does not accept.
const std = @import("std");
const cuda = @import("cuda");
const env = @import("sys").env;
const hub = @import("cuda_exl3_graph.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");
const memory = @import("cuda_exl3_graph_memory.zig");
const dplan = @import("../dplan/cuda_exl3_dplan.zig");
const recycle = @import("../offload/cuda_exl3_recycle.zig");

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

test "exl3: GRAPH selects the planner mode for every consumer and refuses other text" {
    const saved = try SavedEnv.take("TF_FLASHNEXT_EXL3_GRAPH");
    defer saved.restore();
    const previous = hub.force;
    hub.force = null;
    defer hub.force = previous;
    const cases = [_]struct { ?[:0]const u8, hub.Mode }{ .{ null, .off }, .{ "", .off }, .{ " ", .off }, .{ "0", .off }, .{ "eager", .dplan }, .{ "1", .graph } };
    for (cases) |case| {
        try saved.set(case[0]);
        try std.testing.expectEqual(case[1], try hub.mode());
        try std.testing.expectEqual(case[1] != .off, try dplan.enabled());
        try std.testing.expectEqual(case[1] == .graph, try hub.enabled());
        try std.testing.expectEqual(case[1] == .graph, try gprefill.prefillEnabled());
        try std.testing.expectEqual(case[1] == .graph, try recycle.enabled());
    }
    for ([_][:0]const u8{ "2", "true", "Eager", " eager", "1 ", "graph", "01" }) |text| {
        try saved.set(text);
        try std.testing.expectError(error.InvalidEnvironment, hub.mode());
        try std.testing.expectError(error.InvalidEnvironment, hub.enabled());
        try std.testing.expectError(error.InvalidEnvironment, dplan.enabled());
        try std.testing.expectError(error.InvalidEnvironment, gprefill.prefillEnabled());
        try std.testing.expectError(error.InvalidEnvironment, recycle.enabled());
    }
    hub.force = .graph;
    try std.testing.expectEqual(hub.Mode.graph, try hub.mode());
}

test "exl3: GRAPH_ASYNC is on unless 0 and other text is refused before a graph is taken" {
    const saved = try SavedEnv.take("TF_FLASHNEXT_EXL3_GRAPH_ASYNC");
    defer saved.restore();
    for ([_]struct { ?[:0]const u8, bool }{ .{ null, true }, .{ "", true }, .{ "0", false }, .{ "1", true } }) |case| {
        try saved.set(case[0]);
        try std.testing.expectEqual(case[1], try gpending.asyncInstantiate());
    }
    var driver: cuda.Driver = undefined;
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(1) };
    for ([_][:0]const u8{ "2", "off", "true", " 0", "00" }) |text| {
        try saved.set(text);
        try std.testing.expectError(error.InvalidEnvironment, gpending.asyncInstantiate());
        try std.testing.expectError(error.InvalidEnvironment, gpending.startInstantiate(graph, true));
    }
}

test "exl3: the graph budget reads STREAMS and the two-chunk window through their rules" {
    const streams = try SavedEnv.take("TF_FLASHNEXT_EXL3_STREAMS");
    defer streams.restore();
    const window = try SavedEnv.take("TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK");
    defer window.restore();
    try window.set(null);
    try streams.set("2");
    try std.testing.expectEqualDeep(try memory.workloadPolicy(2, 48), try memory.policyFromEnv());
    try streams.set(null);
    try window.set("1");
    try std.testing.expect((try memory.policyFromEnv()).two_chunk_reserved);
    try window.set("0");
    try std.testing.expect(!(try memory.policyFromEnv()).two_chunk_reserved);
    try window.set("2");
    try std.testing.expectError(error.InvalidEnvironment, memory.policyFromEnv());
    try window.set(null);
    for ([_][:0]const u8{ "x", "-1", "1.5", "2k" }) |text| {
        try streams.set(text);
        try std.testing.expectError(error.InvalidEnvironment, memory.policyFromEnv());
    }
}
