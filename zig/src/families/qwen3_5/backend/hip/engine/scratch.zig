//! The scratch each arena needs: the forwards themselves run on the counting stream at their largest shapes.

const std = @import("std");
const hip = @import("hip");
const fwd = @import("../forward/forward.zig");
const win = @import("../forward/window.zig");
const pages = @import("../forward/pages.zig");
const state = @import("../forward/state.zig");
const plan = @import("../forward/plan.zig");
const round = @import("round.zig");
const Engine = @import("engine.zig").Engine;

/// A prompt pass of `rows` rows ending at the last of `capacity` positions, then the head over its last row.
pub fn prompts(e: *Engine, rows: usize, capacity: usize) !usize {
    const m = e.model();
    var pool = try pages.Pool.init(e.gpa, &e.driver, m, 1, false);
    defer pool.deinit();
    var caches = try state.Caches.init(e.gpa, &pool, m, capacity);
    defer caches.deinit(e.gpa);
    var arena = hip.Arena.counting();
    const o = counting(e, &arena);
    // every take grows with the rows and the keys walked, so the last chunk of a full window is the largest
    const hidden = try fwd.span(o, m, &caches, 0, rows, capacity - rows, null);
    const y = try o.project(fwd.at(hidden, (rows - 1) * m.spec.hidden), m.head, 1, false);
    _ = try e.wholeRows(o, y, 1);
    return arena.peak;
}

/// A round of `rows` rows whose attention walks every position of `capacity`: snapshots, forward, whole logits.
pub fn rounds(e: *Engine, rows: usize, capacity: usize) !usize {
    const m = e.model();
    const table = try e.gpa.alloc([2]u64, m.spec.n_layers);
    defer e.gpa.free(table);
    var arena = hip.Arena.counting();
    const o = counting(e, &arena);
    try win.kept(o, m, rows, table);
    const p: hip.plan_ops.Plan = .{ .args = std.mem.zeroes(@FieldType(hip.plan_ops.Plan, "args")), .rows = rows, .slots = rows + 1 };
    const r: win.Round = .{ .plan = p, .tokens = 0, .span = std.mem.alignForward(usize, capacity, plan.min_span), .inputs = table };
    const out = try round.forwardOn(o, m, r, null, 0);
    _ = try e.wholeRows(o, out.y, rows);
    return arena.peak;
}

fn counting(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
    var o = e.ops(arena);
    o.stream = hip.counting;
    return o;
}
