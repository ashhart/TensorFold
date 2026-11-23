//! Which HCA carries each stripe to each peer: on a switch every HCA reaches every peer, on a switchless ring one each.
const guard = @import("roce_multi_guard.zig");
const rule = @import("cuda").roce_pair_rule.Rule(max_hcas, max_peers);

pub const max_hcas = guard.max_hcas;
pub const max_peers = guard.max_peers;

/// The /24 rule of cuda's `roce_pair_rule.zig`, which `roce.zig`'s path pairs by too, at this wire's limits.
pub const Pairs = rule.Pairs;
pub const Plan = rule.Plan;
pub const pairsOf = rule.pairsOf;
pub const plan = rule.plan;
