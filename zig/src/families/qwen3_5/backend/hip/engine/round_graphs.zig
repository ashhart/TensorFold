//! Captured lane rounds: a round's forward recorded once per plan shape and replayed on any later round of it.

const hip = @import("hip");
const plan = @import("../forward/plan.zig");
const graph_cache = @import("graph_cache.zig");

/// What a replay hands back: the forward's outputs and where the arena stands after it.
pub const Out = struct { hidden: hip.ops.Tensor, y: hip.ops.Tensor, used: usize };

pub const Graphs = graph_cache.Cache(plan.Shape, Out);

pub const Entry = Graphs.Entry;
