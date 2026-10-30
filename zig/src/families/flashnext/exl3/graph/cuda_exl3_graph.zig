//! CUDA graphs for an EXL3 Flash Next decode step with the expert offload pool (TF_FLASHNEXT_EXL3_GRAPH=1).

const std = @import("std");
const cuda = @import("cuda");
const state_driver = @import("../kv/cuda_exl3_state_binding.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const gpending = @import("cuda_exl3_graph_pending.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gbudget = @import("cuda_exl3_graph_budget.zig");
const ghooks = @import("cuda_exl3_graph_hooks.zig");
const gprefill = @import("cuda_exl3_graph_prefill.zig");

pub const memory = @import("cuda_exl3_graph_memory.zig");

/// TF_FLASHNEXT_EXL3_GRAPH=1: decode rounds and verify windows replay captured graphs.
pub fn enabled() bool {
    if (force) |f| return f == .graph;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// A test's override of the environment: the host cut, the device plan launched eagerly, or graphs.
pub const Mode = enum { off, dplan, graph };
pub var force: ?Mode = null;

pub const Key = gkey.Key;
pub const Pending = gpending.Pending;
pub const fallback = @import("cuda_exl3_graph_fallback.zig");
pub const cache = gcache;
pub const rebind = @import("cuda_exl3_graph_rebind.zig");
pub const asyncInstantiate = gpending.asyncInstantiate;
pub const startInstantiate = gpending.startInstantiate;
pub const takePending = gpending.takePending;
pub const uploadResident = gbudget.uploadResident;
pub const reserveMemory = gbudget.reserveMemory;
pub const clear = gcache.clear;
pub const releaseSeq = gcache.releaseSeq;
pub const releaseDecodeSeq = gcache.releaseDecodeSeq;
pub const releaseFor = gcache.releaseFor;
pub const round = ghooks.round;
pub const forward = ghooks.forward;
pub const Prefill = gprefill.Prefill;
pub const prefillBegin = gprefill.prefillBegin;
pub const prefillEnd = gprefill.prefillEnd;
pub const prefillAbort = gprefill.prefillAbort;
pub const prefillLayer = gprefill.runLayer;
pub const clearPrefill = gprefill.clearPrefill;
pub const clearPrefillRuntime = gprefill.clearRuntime;

/// True while this module captures a step: the pass's pinned uploads become copy nodes (Uploads.copy).
pub var capturing: bool = false;

/// A copy node from pinned host bytes (the pass's staging): the graph reads them at every replay, as `prepare` left
pub fn pinnedCopy(s: cuda.Stream, dst: u64, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
}

pub const Stats = struct {
    captures: u64 = 0,
    replays: u64 = 0,
    rebind_attempts: u64 = 0,
    rebinds: u64 = 0,
    rebind_failures: u64 = 0,
    rebind_declined: u64 = 0,
    /// Steps on the device plan launched one by one (TF_FLASHNEXT_EXL3_DPLAN=1 without graphs).
    eager_dplan: u64 = 0,
    /// Steps that left for the host cut (too many rows, media, no pools).
    declined: u64 = 0,
    capture_ns: u64 = 0,
    /// Captures that updated a cached graph of the same shape in place instead of instantiating one.
    updates: u64 = 0,
    /// Host time of `prepare` (ids, positions, n-gram rows) before a step's launch, and how many.
    prepare_ns: u64 = 0,
    prepares: u64 = 0,
    /// Captures or instantiations that failed; their rounds ran eagerly (cuda_exl3_graph_fallback.zig).
    capture_failures: u64 = 0,
};

pub var stats: Stats = .{};

/// Captures `pass.chunk(ids, pos0, null)` (prepared) under `key`, or replays the graph captured under it.
pub const Outcome = enum { replay, eager, instantiated, updated };
pub var coverage: [2][17][4]u32 = @splat(@splat(@splat(0)));

pub var prefill_stats: struct { captures: u64 = 0, updates: u64 = 0, replays: u64 = 0, sighting_hits: u64 = 0, sighting_misses: u64 = 0, eager: u64 = 0, admission_misses: u64 = 0, capture_ns: u64 = 0 } = .{};

test "exl3: state binding helper contracts" {
    _ = @import("../kv/exl3_state_binding_test.zig");
    _ = state_driver;
    _ = fallback;
    _ = @import("cuda_exl3_graph_fallback_test.zig");
    _ = @import("cuda_exl3_graph_key_test.zig");
    _ = @import("cuda_exl3_graph_rebind_test.zig");
    _ = @import("cuda_exl3_graph_cache_test.zig");
    _ = @import("cuda_exl3_graph_prefill_test.zig");
    _ = @import("cuda_exl3_graph_together_test.zig");
}
