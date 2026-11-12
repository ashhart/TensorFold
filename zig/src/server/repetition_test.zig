//! Repetition defaults, overrides and capability admission through the actual server.
const std = @import("std");
const server = @import("root.zig");
const json = server.json;
const errors = @import("errors.zig");
const fixtures = @import("stream_preflight_test.zig");
const GreedyEngine = fixtures.GreedyEngine;
const TestText = fixtures.TestText;

test "unsupported repetition factor is refused before generation" {
    var backend: GreedyEngine = .{};
    var text: TestText = .{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config: server.Config = .{ .served_name = "test-model", .model_ids = &.{"test-model"} };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), config, null);
    defer srv.deinit();
    var cx: errors.Cx = .{ .a = a };
    try std.testing.expectError(error.Refused, srv.resolvePenalties(&cx, (try json.parse(a, "{\"repetition_penalty\":1.1}")).ok, false));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "repetition_penalty") != null);
    _ = try srv.resolvePenalties(&cx, (try json.parse(a, "{\"repetition_penalty\":1}")).ok, false);
}

test "repetition defaults and overrides preserve greedy decoding and additive penalty admission" {
    var backend: GreedyEngine = .{};
    var text: TestText = .{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const config: server.Config = .{ .served_name = "test-model", .model_ids = &.{"test-model"}, .default_sampling = (try json.parse(a, "{\"temperature\":0,\"repetition_penalty\":1.1}")).ok };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), config, null);
    defer srv.deinit();
    srv.info.repetition_penalty = true;
    var cx: errors.Cx = .{ .a = a };
    const unset = (try json.parse(a, "{}")).ok;
    try std.testing.expect(try srv.resolveSampling(&cx, unset, 0, &.{}, false) == null);
    try std.testing.expectEqual(@as(f32, 1.1), (try srv.resolvePenalties(&cx, unset, false)).repetition);
    const override = (try json.parse(a, "{\"repetition_penalty\":1}")).ok;
    const p = try srv.resolvePenalties(&cx, override, false);
    try std.testing.expectEqual(@as(f32, 1), p.repetition);
    try std.testing.expectEqual(@as(f32, 0), p.presence);
    const active = (try json.parse(a, "{\"repetition_penalty\":0.5,\"frequency_penalty\":1}")).ok;
    try std.testing.expectError(error.Refused, srv.resolvePenalties(&cx, active, false));
    srv.info.penalties = true;
    const q = try srv.resolvePenalties(&cx, active, false);
    try std.testing.expectEqual(@as(f32, 0.5), q.repetition);
    try std.testing.expectEqual(@as(f32, 1), q.frequency);
}
