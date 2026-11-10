//! Actual Flash Next guard eligibility for drafting engines with offloaded EXL3 experts.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const load = @import("exl3/pack/cuda_exl3_load.zig");
const Cuda = @import("cuda_launch.zig").native.Cuda;

const Layer = enum { dense, resident, offloaded };
const Guards = struct { plain: bool, rate: bool };

/// The engine's decoder layers as `layers` says; the layer just past n_layers is offloaded but not part of the engine.
fn guards(drafting: bool, layers: []const Layer) Guards {
    var resident: load.Experts = undefined;
    resident.pool = null;
    var offloaded: load.Experts = undefined;
    offloaded.pool = @ptrFromInt(256);
    var engine: prompt.Engine = undefined;
    engine.n_layers = layers.len;
    for (layers, engine.layers[0..layers.len]) |kind, *l| l.exl3 = switch (kind) {
        .dense => null,
        .resident => &resident,
        .offloaded => &offloaded,
    };
    engine.layers[layers.len].exl3 = &offloaded;
    var head: mtp.Head = undefined;
    var self: Cuda = undefined;
    self.engine = &engine;
    self.head = if (drafting) &head else null;
    self.depth = mtp.default_depth;
    self.depth_greedy = mtp.default_depth_greedy;
    const facts = self.facts();
    return .{ .plain = facts.plain_guard, .rate = facts.rate_guard };
}

test "exl3: Flash Next guards plain rounds and rates only when a drafting engine's experts are offloaded" {
    const off: Guards = .{ .plain = false, .rate = false };
    const on: Guards = .{ .plain = true, .rate = true };
    try std.testing.expectEqual(off, guards(true, &.{ .dense, .dense, .dense }));
    try std.testing.expectEqual(off, guards(true, &.{ .resident, .resident, .resident }));
    try std.testing.expectEqual(off, guards(true, &.{ .dense, .resident }));
    try std.testing.expectEqual(on, guards(true, &.{.offloaded}));
    try std.testing.expectEqual(on, guards(true, &.{ .offloaded, .resident, .dense }));
    try std.testing.expectEqual(on, guards(true, &.{ .dense, .offloaded, .dense }));
    try std.testing.expectEqual(on, guards(true, &.{ .dense, .resident, .offloaded }));
    // The offloaded sentinel beyond n_layers is not an engine layer.
    try std.testing.expectEqual(off, guards(true, &.{}));
    try std.testing.expectEqual(off, guards(false, &.{ .offloaded, .resident }));
}
