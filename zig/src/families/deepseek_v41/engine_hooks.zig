//! Engine open's forward hooks: every later layer's part of forward_hooks.Hooks, and the forward's prefill state.
const round_graph = @import("round_graph.zig");
const greedy = @import("greedy.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const Forward = @import("forward.zig").Forward;

fn currentFn() anyerror!ft.RoundGraph {
    return round_graph.current();
}

const round_graph_table: hooks.RoundGraph = .{ .current = currentFn, .splitAt = round_graph.splitAt };
const greedy_table: hooks.Greedy = .{ .argmax = greedy.argmax };

/// Every part set (the same table for every forward), then `f`'s prefill state; right after Forward.init.
pub fn open(f: *Forward) !void {
    hooks.install(.{
        .prefill = &@import("forward_prefill.zig").hooks_table,
        .gate = &@import("engram_gate_fwd.zig").table,
        .round_graph = &round_graph_table,
        .greedy = &greedy_table,
        .branches = &@import("branches.zig").hooks_table,
    });
    try @import("forward_prefill.zig").attach(f);
}
