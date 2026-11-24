//! Living-memory bank: compiles text into exact BF16 KV and F32 low-rank recurrent deltas, with gating and persistence.
const std = @import("std");
const whitener = @import("whitener.zig");
const gate = @import("gate.zig");
const store = @import("store.zig");
const types = @import("types.zig");
const teach = @import("teach.zig");
const compose = @import("compose.zig");
const consolidate_mod = @import("consolidate.zig");
const save_mod = @import("save.zig");
const load = @import("load.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const shared_sys_prefix = types.shared_sys_prefix;
pub const shared_sys_suffix = types.shared_sys_suffix;
pub const Family = types.Family;
pub const ArchSpec = types.ArchSpec;
pub const SnapHandle = types.SnapHandle;
pub const RawState = types.RawState;
pub const Backend = types.Backend;
pub const KvLayerData = types.KvLayerData;
pub const RecLayerData = types.RecLayerData;
pub const Unit = types.Unit;
pub const CalPrompt = types.CalPrompt;
pub const TeachChunkInput = types.TeachChunkInput;
pub const ConsolidateReport = types.ConsolidateReport;

pub const Bank = struct {
    gpa: Allocator,
    io: Io,
    dir: []u8,
    backend: Backend,
    calibrated: bool = false,
    w_lex: ?whitener.LayerWhitener = null,
    w_sem: ?whitener.LayerWhitener = null,
    cal_prompts: std.ArrayList(CalPrompt) = .empty,
    prefix_ids: []u32 = &.{},
    suffix_ids: []u32 = &.{},
    prefix_state: ?RawState = null,
    units: std.ArrayList(Unit) = .empty,
    bridges: []gate.BridgeEdge = &.{},
    cluster_index: gate.ClusterIndex = .{},
    /// Sleep-time spectral-consolidated recurrent steering delta across active standing policies.
    skill_rec: []RecLayerData = &.{},
    skill_policy_count: u32 = 0,
    next_unit_id: u32 = 1,
    next_epoch: u64 = 1,
    base_saved: bool = false,
    pending_deltas: u32 = 0,
    unflushed_from: usize = 0,
    trace: bool = false, // TF_SLIDE_TRACE, set by the native controller: routes and teaches are logged
    journal_gen: u64 = 0, // the base file's generation: only a journal of the same one is replayed onto it
    last_bytes: u64 = 0,
    route_cfg: gate.RouteConfig = .{},

    pub fn init(gpa: Allocator, io: Io, dir: []const u8, backend: Backend) !*Bank {
        const b = try gpa.create(Bank);
        errdefer gpa.destroy(b);
        b.* = .{
            .gpa = gpa,
            .io = io,
            .dir = try gpa.dupe(u8, dir),
            .backend = backend,
        };
        return b;
    }

    pub fn deinit(self: *Bank) void {
        self.clearMemory();
        self.gpa.free(self.dir);
        self.gpa.destroy(self);
    }

    pub fn freeSkillRec(self: *Bank) void {
        for (self.skill_rec) |*r| r.deinit(self.gpa);
        if (self.skill_rec.len > 0) self.gpa.free(self.skill_rec);
        self.skill_rec = &.{};
        self.skill_policy_count = 0;
    }

    pub fn clearMemory(self: *Bank) void {
        self.freeSkillRec();
        self.cluster_index.deinit(self.gpa);
        self.gpa.free(self.bridges);
        self.bridges = &.{};
        for (self.units.items) |*u| u.deinit(self.gpa);
        self.units.deinit(self.gpa);
        self.units = .empty;
        for (self.cal_prompts.items) |*c| c.deinit(self.gpa);
        self.cal_prompts.deinit(self.gpa);
        self.cal_prompts = .empty;
        if (self.w_lex) |*w| w.deinit(self.gpa);
        self.w_lex = null;
        if (self.w_sem) |*w| w.deinit(self.gpa);
        self.w_sem = null;
        if (self.prefix_state) |*p| p.deinit(self.gpa);
        self.prefix_state = null;
        self.gpa.free(self.prefix_ids);
        self.prefix_ids = &.{};
        self.gpa.free(self.suffix_ids);
        self.suffix_ids = &.{};
        self.calibrated = false;
        self.base_saved = false;
        self.pending_deltas = 0;
        self.unflushed_from = 0;
    }

    pub fn hasUnits(self: *const Bank) bool {
        if (!self.calibrated or self.prefix_state == null) return false;
        for (self.units.items) |u| if (u.gate.isActive()) return true;
        return false;
    }

    pub fn activeUnitCount(self: *const Bank) u32 {
        var c: u32 = 0;
        for (self.units.items) |u| if (u.gate.isActive()) {
            c += 1;
        };
        return c;
    }

    pub fn maxParentId(self: *const Bank) u32 {
        var m: u32 = 0;
        for (self.units.items) |u| {
            if (u.parent_id > m) m = u.parent_id;
        }
        return m;
    }

    /// Recomputes both the pure-vector associative bridge graph and the 2-level ClusterIndex (< 2 ms).
    pub fn rebuildBridges(self: *Bank) !void {
        self.gpa.free(self.bridges);
        self.bridges = &.{};
        self.cluster_index.deinit(self.gpa);
        if (self.units.items.len == 0) return;

        const views = try self.gpa.alloc(gate.UnitView, self.units.items.len);
        defer self.gpa.free(views);
        for (self.units.items, views, 0..) |*u, *v, idx| {
            v.* = .{ .index = idx, .parent_id = u.parent_id, .n_tokens = u.n_tokens, .gate = &u.gate };
        }
        if (self.units.items.len >= 2) {
            self.bridges = try gate.computeBridgeEdges(self.gpa, views);
        }
        try self.cluster_index.rebuild(self.gpa, views, self.backend.arch.hidden);
        if (self.trace) {
            std.debug.print("lw6 index: {d} bridges, {d} clusters across {d} units", .{ self.bridges.len, self.cluster_index.n_clusters, self.units.items.len });
            for (self.bridges[0..@min(self.bridges.len, 12)]) |e| {
                std.debug.print(" [u{d}->u{d}:{d:.2}]", .{ self.units.items[e.from].unit_id, self.units.items[e.to].unit_id, e.weight });
            }
            std.debug.print("\n", .{});
        }
    }

    /// Calibrates and compiles the shared prefix once; if loaded calibrated, keeps units and rebuilds `cal_prompts`.
    pub fn ensureCalibrated(
        self: *Bank,
        cal_ids: []const []const u32,
        prefix_ids: []const u32,
        suffix_ids: []const u32,
    ) !void {
        return teach.ensureCalibrated(self, cal_ids, prefix_ids, suffix_ids);
    }

    /// Compiles one chunk (fact, doc section, digest or policy) into KV, recurrent deltas, gate prototypes and bridges.
    pub fn teachChunk(self: *Bank, inp: TeachChunkInput) !u32 {
        return teach.teachChunk(self, inp);
    }

    /// Vector-only entity conflict search: returns up to `max_out` active small-fact parent IDs overlapping `gate_ids`.
    pub fn findConflictParents(self: *Bank, gate_ids: []const u32, max_out: usize) ![]gate.ConflictMatch {
        return compose.findConflictParents(self, gate_ids, max_out);
    }

    /// Marks `old_parent_id`'s active units superseded by `new_parent_id`; they stop routing but stay restorable.
    pub fn supersedeParent(self: *Bank, old_parent_id: u32, new_parent_id: u32) u32 {
        if (old_parent_id == 0 or new_parent_id == 0 or old_parent_id == new_parent_id) return 0;
        var changed: u32 = 0;
        var touched_standing = false;
        for (self.units.items) |*u| {
            if (u.parent_id == old_parent_id and u.gate.superseded_by == 0) {
                u.gate.superseded_by = new_parent_id;
                if (u.gate.is_standing) touched_standing = true;
                changed += 1;
            }
        }
        if (touched_standing) self.freeSkillRec();
        if (changed > 0) self.rebuildBridges() catch {};
        return changed;
    }

    /// Increments the reinforcement counter for all active units of `parent_id`.
    pub fn reinforceParent(self: *Bank, parent_id: u32) u32 {
        var changed: u32 = 0;
        for (self.units.items) |*u| {
            if (u.parent_id == parent_id and u.gate.superseded_by == 0) {
                u.gate.reinforcements +|= 1;
                changed += 1;
            }
        }
        return changed;
    }

    /// Removes `parent_id`'s units and restores any belief that `parent_id` had superseded.
    pub fn forgetParent(self: *Bank, parent_id: u32) u32 {
        var removed: u32 = 0;
        var restored: u32 = 0;
        var touched_standing = false;
        var i: usize = 0;
        while (i < self.units.items.len) {
            if (self.units.items[i].parent_id == parent_id) {
                if (self.units.items[i].gate.is_standing) touched_standing = true;
                self.units.items[i].deinit(self.gpa);
                _ = self.units.orderedRemove(i);
                if (i < self.unflushed_from) self.unflushed_from -= 1;
                removed += 1;
            } else {
                if (self.units.items[i].gate.superseded_by == parent_id) {
                    self.units.items[i].gate.superseded_by = 0;
                    if (self.units.items[i].gate.is_standing) touched_standing = true;
                    restored += 1;
                }
                i += 1;
            }
        }
        if (touched_standing) self.freeSkillRec();
        if (removed > 0 or restored > 0) self.rebuildBridges() catch {};
        return removed;
    }

    /// Runs the gate (ClusterIndex, token guard, coverage, 2-hop bridges, policies) on `query_ids`; null on miss.
    pub fn routeAndCompose(self: *Bank, query_ids: []const u32) !?SnapHandle {
        return compose.routeAndCompose(self, query_ids);
    }

    /// Assembles prefix + active-unit KV (K shifted to contiguous offsets) and SVD recurrent state (prefix if rank 0).
    pub fn composeRawState(self: *const Bank, active_indices: []const usize) !RawState {
        return compose.composeRawState(self, active_indices);
    }

    pub fn filePath(self: *const Bank, gpa: Allocator) ![]u8 {
        return std.fs.path.join(gpa, &.{ self.dir, store.file_name });
    }

    pub fn deltaPath(self: *const Bank, gpa: Allocator) ![]u8 {
        return std.fs.path.join(gpa, &.{ self.dir, store.delta_file_name });
    }

    /// Sleep-time consolidation: can prune tombstones, builds policy delta `skill_rec`, rebuilds indexes, compacts.
    pub fn consolidate(self: *Bank, prune_superseded: bool, compact_policies: bool) !ConsolidateReport {
        return consolidate_mod.consolidate(self, prune_superseded, compact_policies);
    }

    /// Appends only new units to `living-weights.delta` if a base exists and < 16 frames are pending; else `saveFull`.
    pub fn save(self: *Bank) !u64 {
        return save_mod.save(self);
    }

    /// Appends a tombstone frame (forget or supersede), or compacts if no base, threshold reached or bank empty.
    pub fn persistTombstone(self: *Bank, op: store.DeltaOp, pid_a: u32, pid_b: u32) !u64 {
        return save_mod.persistTombstone(self, op, pid_a, pid_b);
    }

    /// Forgets `parent_id` on disk before in memory: on a failed write the bank is exactly as it was, and errs.
    pub fn forgetDurable(self: *Bank, parent_id: u32) !u32 {
        return save_mod.forgetDurable(self, parent_id);
    }

    /// Removes the bank's files, then its units: when a file cannot go, the bank keeps every unit and says why.
    pub fn clearDurable(self: *Bank) !u32 {
        return save_mod.clearDurable(self);
    }

    /// Atomically writes the bank to `living-weights.safetensors` (`tf-living-memory-v6`) and removes the journal.
    pub fn saveFull(self: *Bank) !u64 {
        return save_mod.saveFull(self);
    }

    /// Loads `living-weights.safetensors` (v6, v4 or v3), replays `living-weights.delta`, rebuilds prefix and indexes.
    pub fn loadIfPresent(self: *Bank) !bool {
        return load.loadIfPresent(self);
    }
};
