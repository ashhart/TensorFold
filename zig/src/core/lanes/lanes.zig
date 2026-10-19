//! The lane engine's host core (Python lane_engine.py); backends run forwards and draws behind `backend.Backend`.
const std = @import("std");

pub const proposal_score = @import("proposal_score.zig");
pub const proposal_head = @import("proposal_head.zig");
pub const table = @import("table.zig");
pub const allocate = @import("allocate.zig");
pub const accept = @import("accept.zig");
pub const proposer = @import("proposer.zig");
pub const recall = @import("recall.zig");
pub const recall_store = @import("recall_store.zig");
pub const sampling = @import("sampling.zig");
pub const fill = @import("fill.zig");
pub const shape = @import("shape.zig");
pub const plan_lanes = @import("plan_lanes.zig");
pub const cost_cache = @import("cost_cache.zig");
pub const cost_rule = @import("cost_rule.zig");
pub const gpu_rule = @import("gpu_rule.zig");
pub const gpu_full = @import("gpu_full.zig");
pub const depth = @import("depth.zig");
pub const mtp_policy = @import("mtp_policy.zig");
pub const trim = @import("trim.zig");
pub const config = @import("config.zig");
pub const stream = @import("stream.zig");
pub const backend = @import("backend.zig");
pub const embedding = @import("embedding.zig");
pub const events = @import("events.zig");
pub const windows = @import("windows.zig");
pub const trail = @import("trail.zig");
pub const engine = @import("engine.zig");
pub const fake = @import("fake.zig");
pub const drafter = @import("drafter.zig");
pub const drafted = @import("drafted.zig");
pub const recorder = @import("recorder.zig");
pub const logprob = @import("logprob.zig");
pub const grammar = @import("grammar.zig");

pub const Engine = engine.Engine;
pub const Config = config.Config;
pub const Model = config.Model;
pub const Stream = stream.Stream;
pub const Sampling = sampling.Sampling;
pub const SuffixLookup = proposer.SuffixLookup;
pub const LogprobRow = logprob.Row;
pub const Trim = trim.Trim;

test {
    std.testing.refAllDecls(@This());
    _ = @import("engine_test.zig");
    _ = @import("handover_test.zig");
    _ = @import("open_first_test.zig");
    _ = @import("recorder_test.zig");
    _ = @import("gpu_full_test.zig");
    _ = @import("drafted_test.zig");
    _ = @import("root_copy_test.zig");
    _ = @import("drafted_grammar_test.zig");
    _ = @import("windows_grammar_test.zig");
    _ = @import("pricing_test.zig");
    _ = @import("depth_probe_test.zig");
    _ = @import("proposal_external_test.zig");
    _ = @import("lone_costs_test.zig");
    _ = @import("lone_backoff_test.zig");
    _ = @import("fill_test.zig");
    _ = @import("fill_trim_test.zig");
    _ = @import("fair_turns_test.zig");
}

pub const learned_disk = @import("learned_disk.zig");
pub const learned_dirs = @import("learned_dirs.zig");
