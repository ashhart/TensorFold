//! Captured lane rounds: a round recorded once per plan shape as graphs cut at its collectives, replayed after.

const hip = @import("hip");
const plan = @import("../forward/plan.zig");
const graph_cache = hip.graph_cache;

/// What a replay hands back: the forward's outputs and where the arena stands after it.
pub const Out = struct { hidden: hip.ops.Tensor, y: hip.ops.Tensor, used: usize };

pub const Graphs = graph_cache.CacheOf(plan.Shape, Out, hip.graph_split.Split);

pub const Entry = Graphs.Entry;
