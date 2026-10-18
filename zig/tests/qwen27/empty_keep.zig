//! Physical shared verify and mixed empty/full keep against each slot's exact recurrent, conv and offset reference.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").qwen27;
const graph = @import("synthetic_graph.zig");

fn state(r: *const q.decode_round.Runner, slot: u32) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(std.mem.asBytes(&r.offsets[slot]));
    for (0..r.gdn.budget.layers) |layer| for ([_]bool{ true, false }) |recurrent| {
        h.update(try r.gdn.committed(layer, slot, recurrent));
    };
    return h.finalResult();
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const autorelease = mtl.objc.Pool.push();
    defer autorelease.pop();
    const model = try graph.createLayers(a, 5);
    defer model.deinit();
    var r = try q.decode_round.Runner.init(a, model, 2, 256);
    defer r.deinit();
    var plain = try q.decode_round.Runner.init(a, model, 1, 256);
    defer plain.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    for ([_]bool{ false, true }) |both_empty| {
        for (0..2) |slot| {
            try r.reset(@intCast(slot));
            try (q.session.Session{ .runner = &r, .slot = @intCast(slot) }).prefill(&prompt, 16);
        }
        try plain.reset(0);
        try (q.session.Session{ .runner = &plain }).prefill(&prompt, 16);
        const before = try state(&r, 0);
        var round = try q.round_plan.Round.init(a, &.{
            .{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = &.{6} },
            .{ .slot = 1, .start = r.offsets[1], .capacity = r.capacity, .ids = &.{6} },
        }, @intCast(model.config.conv_kernel), 2, @intCast(model.config.vocab));
        defer round.deinit();
        try r.verifyHead(&round, .all);
        try r.keep(&round, &.{ &.{}, if (both_empty) &.{} else &.{0} });
        if (!std.mem.eql(u8, &before, &try state(&r, 0))) return error.EmptyKeepChangedState;
        if (!both_empty) try (q.session.Session{ .runner = &plain }).step(6);
        if (!std.mem.eql(u8, &try state(&plain, 0), &try state(&r, 1))) return error.MixedKeepChangedState;
    }
    std.debug.print("PASS shared empty/full keeps: recurrent and conv bytes, offsets, every layer\n", .{});
}
