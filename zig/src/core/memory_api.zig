//! Memory engine-API types: engine-thread teach/save/forget/stats ops and LaneHost's MemoryController hook.
const std = @import("std");

pub const SnapHandle = struct {
    saved: *anyopaque,
    at: u32,
};

pub const MemoryOp = union(enum) {
    /// Ensure calibration & shared prefix are compiled using tokenizer-encoded calibration texts and framing.
    init: struct {
        cal_ids: []const []const u32,
        prefix_ids: []const u32,
        suffix_ids: []const u32,
    },
    /// Compile one memory chunk (fact, document section, digest, or standing policy) into pure weights.
    teach_chunk: struct {
        parent_id: u32,
        body_ids: []const u32,
        gate_ids: []const u32,
        title_ids: []const u32 = &.{},
        cue_ids: []const u32 = &.{},
        is_small_fact: bool = false,
        is_standing: bool = false,
        is_digest: bool = false,
    },
    /// Pure-vector (< 1 ms) search for active small-fact parent IDs overlapping `gate_ids` (belief revision).
    find_conflicts: struct {
        gate_ids: []const u32,
    },
    /// Mark all active units of `old_parent_id` as superseded by `new_parent_id` and persist a delta tombstone.
    supersede_parent: struct {
        old_parent_id: u32,
        new_parent_id: u32,
    },
    /// Sleep-time consolidation: prune superseded tombstones, spectrally fuse standing policies, compact delta journal.
    consolidate: struct {
        prune_superseded: bool = true,
        compact_policies: bool = true,
    },
    /// Fast incremental delta save (or atomic compaction when base is absent or >= 16 deltas pending).
    save,
    /// Remove all units belonging to `parent_id`, restore any unit superseded by it, and persist delta/compaction.
    forget_parent: u32,
    /// Remove all units and delete `<model_dir>/living-weights.safetensors` + `.delta`.
    clear_all,
    /// Query current unit count, cluster count, pending deltas, and saved bytes.
    stats,
};

pub const MemoryResult = struct {
    units: u32 = 0,
    affected: u32 = 0,
    tokens: u32 = 0,
    bytes: u64 = 0,
    pruned: u32 = 0,
    clusters: u32 = 0,
    deltas: u32 = 0,
    conflict_ids: [4]u32 = .{ 0, 0, 0, 0 },
    conflict_count: u8 = 0,
    err: ?[]const u8 = null,
};

/// Engine-thread controller owned by a native host (`qwen35`, `nemotron`, `qwen27`).
pub const MemoryController = struct {
    ctx: *anyopaque,
    hasUnits: *const fn (ctx: *anyopaque) bool,
    routeAndCompose: *const fn (ctx: *anyopaque, query_ids: []const u32) anyerror!?SnapHandle,
    dropSnap: *const fn (ctx: *anyopaque, snap: *anyopaque) void,
    exec: *const fn (ctx: *anyopaque, op: *const MemoryOp) MemoryResult,
};
