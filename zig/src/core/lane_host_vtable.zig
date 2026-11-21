//! Which engine seam a family's lane host serves: decisions' label scores only when the backend gives them.
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const recalls = @import("lane_host_recall.zig");
const rows_host = @import("lane_host_rows.zig");
const load = @import("lane_host_load.zig");
const scoring = @import("lane_host_score.zig");
const Engine = api.Engine;

/// The seam every host serves: the engine's own hooks, its learn/recall stores and the hand-off exports.
pub const plain: Engine.VTable = .{
    .info = LaneHost.infoFn,
    .submit = LaneHost.submitFn,
    .cancel = LaneHost.cancelFn,
    .status = status.read,
    .memory = LaneHost.memoryFn,
    .keepalive = LaneHost.keepaliveFn,
    .learn = learn.submit,
    .recall = recalls.attach,
    .imprint = rows_host.ask,
    .drain = rows_host.drain,
    .load = load.attach,
    .handoff = @import("lane_export.zig").serveHook,
};

/// `plain`, with a decision's label scores when the backend prices its choices.
pub const scored: Engine.VTable = blk: {
    var v = plain;
    v.score = scoring.score;
    break :blk v;
};

/// `scored`, with a decision route when the backend makes one.
pub const decided: Engine.VTable = blk: {
    var v = scored;
    v.decide = scoring.decide;
    break :blk v;
};
