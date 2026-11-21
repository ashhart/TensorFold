//! Native backend allocation lifetime without adding engine imports to source-only operator checks.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("tensorfold").deepseek_v41.Model;
const Backend = @import("tensorfold").deepseek_v41_backend.Backend;

test "V4.1 backend releases all per-slot host state on unload" {
    var model: Model = undefined;
    model.lanes = 4;
    var backend = try Backend.init(std.testing.allocator, &model);
    backend.deinit();
}

test "drafter ID growth preserves pending Markov picks" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const picks = try device.buffer(8 * @sizeOf(u32), mtl.ResourceOptions.shared);
    defer picks.deinit();
    const expected = [_]u32{ 11, 23, 37, 41, 53, 67, 79, 83 };
    @memcpy(picks.slice(u32, expected.len), &expected);
    var model: Model = undefined;
    model.device = device;
    model.residency = try device.residencySet(4);
    model.residency.add(picks);
    model.drafter_ids = null;
    model.markov_ids = picks;
    defer {
        model.residency.deinit();
        if (model.drafter_ids) |ids| ids.deinit();
        if (model.markov_ids) |ids| {
            if (ids.gpuAddress() != picks.gpuAddress()) ids.deinit();
        }
    }
    _ = try model.drafterId(97);
    try std.testing.expectEqualSlices(u32, &expected, model.markov_ids.?.slice(u32, expected.len));
    for ([_]u32{ 5, 17 }) |rows| {
        const ids = try model.drafterRound(97, rows);
        try ids.requireRows(rows, 1, 1, @sizeOf(u32));
        try std.testing.expectEqualSlices(u32, &expected, model.markov_ids.?.slice(u32, expected.len));
    }
}

test "trained-only draft depth stays enabled through acceptance transitions" {
    var model: Model = undefined;
    model.mtp_enabled = true;
    // Only bound-weight presence is read; this host policy performs no GPU operation.
    model.weights.dspark = @as(@TypeOf(model.weights.dspark.?), undefined);
    model.mtp_accept_slot = @splat(0);
    model.mtp_round_slot = @splat(0);
    var previous = [_]u32{0};
    var shallow = [_]bool{false};
    var backend: Backend = undefined;
    backend.model = &model;
    backend.last_depth = &previous;
    backend.shallow = &shallow;
    try std.testing.expectEqual(@as(u32, 1), backend.adaptiveDepth(0));
    model.mtp_accept_slot[0] = 0.9;
    try std.testing.expectEqual(@as(u32, 3), backend.adaptiveDepth(0));
    previous[0] = 3;
    model.mtp_accept_slot[0] = 0.7;
    try std.testing.expectEqual(@as(u32, 3), backend.adaptiveDepth(0));
    model.mtp_accept_slot[0] = 0.69;
    try std.testing.expectEqual(@as(u32, 1), backend.adaptiveDepth(0));
    model.mtp_enabled = false;
    try std.testing.expectEqual(@as(u32, 0), backend.adaptiveDepth(0));
}

test "suffix drafts follow committed context" {
    const gpa = std.testing.allocator;
    var model: Model = undefined;
    model.lanes = 1;
    model.mtp_enabled = false;
    model.weights.dspark = null;
    model.caches = &.{};
    var backend = try Backend.init(gpa, &model);
    defer backend.deinit();
    var stream = try @import("lanes").Stream.init(gpa, .{
        .id = "committed",
        .prompt = &.{ 1, 2, 3, 4, 5, 9 },
        .max_new = 32,
    });
    defer stream.deinit(gpa);
    _ = try stream.commit(gpa, &.{ 1, 2, 3 }, &.{});
    stream.pending = 3;
    try backend.by.put(gpa, &stream, 0);
    try Backend.draft(&backend, &.{.{
        .stream = &stream,
        .follow = &.{},
        .rows = &.{},
        .start = 6,
        .position = stream.context.items.len,
        .depth = 3,
    }});
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, backend.held[0]);
}

test "lookup proposals take precedence over bound trained drafts" {
    const gpa = std.testing.allocator;
    var model: Model = undefined;
    model.lanes = 1;
    model.mtp_enabled = true;
    model.weights.dspark = @as(@TypeOf(model.weights.dspark.?), undefined);
    model.mtp_accept_slot = @splat(1);
    model.mtp_round_slot = @splat(1);
    model.caches = &.{};
    var backend = try Backend.init(gpa, &model);
    defer backend.deinit();
    var stream = try @import("lanes").Stream.init(gpa, .{
        .id = "lookup-preferred",
        .prompt = &.{ 1, 2, 3, 4, 5, 9 },
        .max_new = 32,
    });
    defer stream.deinit(gpa);
    _ = try stream.commit(gpa, &.{ 1, 2, 3 }, &.{});
    stream.pending = 3;
    try backend.by.put(gpa, &stream, 0);
    backend.mtp_cache[0] = .{ .pending = 3, .drafts = .{ 90, 91, 92, 93 }, .count = 3 };
    const held = (try Backend.tree(&backend, &stream, gpa)) orelse return error.MissingLookupProposal;
    const tokens = held.tokens orelse return error.MissingLookupTokens;
    defer gpa.free(tokens);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, tokens);
    try Backend.draft(&backend, &.{.{
        .stream = &stream,
        .follow = &.{},
        .rows = &.{},
        .start = 6,
        .position = stream.context.items.len,
        .depth = 3,
    }});
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, backend.held[0]);
}
