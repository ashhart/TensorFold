//! Atomic live-state and persistence transactions for staged lessons and sleep-time consolidation.
const std = @import("std");
const compress = @import("compress.zig");
const gate = @import("gate.zig");
const journal = @import("journal.zig");
const save_mod = @import("save.zig");
const store = @import("store.zig");
const types = @import("types.zig");
const unit_store = @import("unit_store.zig");
const Bank = @import("bank.zig").Bank;
const Io = std.Io;
const RecLayerData = types.RecLayerData;
const Unit = types.Unit;

const Removed = struct { at: usize, unit: Unit, added: bool = false };
const GateState = struct { unit_id: u32, superseded_by: u32 };

const SavedIndexes = struct {
    bridges: []gate.BridgeEdge,
    clusters: gate.ClusterIndex,

    fn detach(self: *SavedIndexes, b: *Bank) void {
        self.bridges = b.bridges;
        self.clusters = b.cluster_index;
        b.bridges = &.{};
        b.cluster_index = .{};
    }

    fn discardLiveAndRestore(self: *SavedIndexes, b: *Bank) void {
        b.gpa.free(b.bridges);
        b.cluster_index.deinit(b.gpa);
        b.bridges = self.bridges;
        b.cluster_index = self.clusters;
    }

    fn discardSaved(self: *SavedIndexes, b: *Bank) void {
        b.gpa.free(self.bridges);
        self.clusters.deinit(b.gpa);
    }
};

const LessonRollback = struct {
    added_ids: []const u32,
    removed: []Removed,
    removed_len: usize,
    gates: []GateState,
    gates_len: usize,
    indexes: SavedIndexes,
    skill: []RecLayerData,
    skill_pols: u32,
    skill_changed: bool,
    version: u64,
    base_saved: bool,
    pending_deltas: u32,
    unflushed_from: usize,
    journal_gen: u64,
    last_bytes: u64,

    fn isAdded(self: *const LessonRollback, unit_id: u32) bool {
        return std.mem.indexOfScalar(u32, self.added_ids, unit_id) != null;
    }

    fn restore(self: *LessonRollback, b: *Bank) void {
        var i: usize = 0;
        while (i < b.units.items.len) {
            if (!self.isAdded(b.units.items[i].unit_id)) {
                i += 1;
                continue;
            }
            var u = b.units.orderedRemove(i);
            u.deinit(b.gpa);
        }
        while (self.removed_len > 0) {
            self.removed_len -= 1;
            const x = &self.removed[self.removed_len];
            if (x.added) {
                x.unit.deinit(b.gpa);
            } else {
                b.units.insertAssumeCapacity(x.at, x.unit);
            }
        }
        for (self.gates[0..self.gates_len]) |old| {
            for (b.units.items) |*u| {
                if (u.unit_id == old.unit_id) {
                    u.gate.superseded_by = old.superseded_by;
                    break;
                }
            }
        }
        b.skill_rec = self.skill;
        b.skill_policy_count = self.skill_pols;
        b.version = self.version;
        b.base_saved = self.base_saved;
        b.pending_deltas = self.pending_deltas;
        b.unflushed_from = self.unflushed_from;
        b.journal_gen = self.journal_gen;
        b.last_bytes = self.last_bytes;
        self.indexes.discardLiveAndRestore(b);
    }

    fn finish(self: *LessonRollback, b: *Bank) void {
        for (self.removed[0..self.removed_len]) |*x| x.unit.deinit(b.gpa);
        if (self.skill_changed) freeSkill(b, self.skill);
        self.indexes.discardSaved(b);
    }
};

fn freeSkill(b: *Bank, skill: []RecLayerData) void {
    for (skill) |*r| r.deinit(b.gpa);
    if (skill.len > 0) b.gpa.free(skill);
}

fn rememberGate(rb: *LessonRollback, u: *const Unit) void {
    for (rb.gates[0..rb.gates_len]) |old| if (old.unit_id == u.unit_id) return;
    rb.gates[rb.gates_len] = .{ .unit_id = u.unit_id, .superseded_by = u.gate.superseded_by };
    rb.gates_len += 1;
}

fn replaceLive(b: *Bank, rb: *LessonRollback, candidate_unflushed: *usize, parent_id: u32) !void {
    var changed = false;
    var i: usize = 0;
    while (i < b.units.items.len) {
        if (b.units.items[i].parent_id == parent_id) {
            rb.skill_changed = rb.skill_changed or b.units.items[i].gate.is_standing;
            const u = b.units.orderedRemove(i);
            rb.removed[rb.removed_len] = .{ .at = i, .unit = u, .added = rb.isAdded(u.unit_id) };
            rb.removed_len += 1;
            if (i < candidate_unflushed.*) candidate_unflushed.* -= 1;
            changed = true;
            continue;
        }
        if (b.units.items[i].gate.superseded_by == parent_id) {
            rb.skill_changed = rb.skill_changed or b.units.items[i].gate.is_standing;
            rememberGate(rb, &b.units.items[i]);
            b.units.items[i].gate.superseded_by = 0;
            changed = true;
        }
        i += 1;
    }
    if (changed) b.version = std.math.add(u64, b.version, 1) catch return error.VersionExhausted;
}

fn persistedBaseLength(b: *Bank, path: []const u8) u64 {
    const f = Io.Dir.cwd().openFile(b.io, path, .{}) catch return b.last_bytes;
    defer f.close(b.io);
    return f.length(b.io) catch b.last_bytes;
}

fn prepareDeltaFrames(
    b: *Bank,
    added_ids: []const u32,
    replace_ids: []const u32,
    replace_effective: []const bool,
    superseded_pid: u32,
    supersede_effective: bool,
    new_parent_id: u32,
    frame_count: usize,
) ![]store.DeltaFrame {
    const frames = try b.gpa.alloc(store.DeltaFrame, frame_count);
    var made: usize = 0;
    errdefer {
        for (frames[0..made]) |*fr| fr.deinit(b.gpa);
        b.gpa.free(frames);
    }
    for (added_ids) |uid| {
        var found: ?*Unit = null;
        for (b.units.items) |*u| {
            if (u.unit_id == uid) {
                found = u;
                break;
            }
        }
        const u = found orelse return error.MissingLessonUnit;
        frames[made] = .{ .op = .add_unit, .payload = try unit_store.serializeSingleUnit(b, b.gpa, u) };
        made += 1;
    }
    for (replace_ids, replace_effective) |pid, effective| {
        if (!effective) continue;
        const payload = try b.gpa.alloc(u8, 4);
        std.mem.writeInt(u32, payload[0..4], pid, .little);
        frames[made] = .{ .op = .forget_parent, .payload = payload };
        made += 1;
    }
    if (supersede_effective) {
        const payload = try b.gpa.alloc(u8, 8);
        std.mem.writeInt(u32, payload[0..4], superseded_pid, .little);
        std.mem.writeInt(u32, payload[4..8], new_parent_id, .little);
        frames[made] = .{ .op = .supersede_parent, .payload = payload };
        made += 1;
    }
    std.debug.assert(made == frame_count);
    return frames;
}

pub fn commitLesson(self: *Bank, new_parent_id: u32, replace_ids: []const u32, superseded_id: ?u32) !u32 {
    return commitLessonWithSync(self, new_parent_id, replace_ids, superseded_id, store.syncParent);
}

/// Publishes a staged lesson and every final tombstone in one atomic disk transaction.
pub fn commitLessonWithSync(
    self: *Bank,
    new_parent_id: u32,
    replace_ids: []const u32,
    superseded_id: ?u32,
    sync_dir: *const fn (Io, []const u8) anyerror!void,
) !u32 {
    var added_n: usize = 0;
    var added_standing = false;
    for (self.staged.items) |u| if (u.parent_id == new_parent_id) {
        added_n += 1;
        added_standing = added_standing or u.gate.is_standing;
    };
    if (added_n == 0) return 0;
    const initial_len = self.units.items.len;
    const capacity = std.math.add(usize, initial_len, added_n) catch return error.OutOfMemory;

    const added_ids = try self.gpa.alloc(u32, added_n);
    defer self.gpa.free(added_ids);
    const removed = try self.gpa.alloc(Removed, capacity);
    defer self.gpa.free(removed);
    const gates = try self.gpa.alloc(GateState, capacity);
    defer self.gpa.free(gates);
    const replace_effective = try self.gpa.alloc(bool, replace_ids.len);
    defer self.gpa.free(replace_effective);
    var replacement_frames: usize = 0;
    for (replace_ids, 0..) |pid, ri| {
        var effective = pid == new_parent_id;
        for (self.units.items) |u| {
            if (u.parent_id == pid or u.gate.superseded_by == pid) {
                effective = true;
                break;
            }
        }
        for (replace_ids[0..ri]) |earlier| if (earlier == pid) {
            effective = false;
            break;
        };
        replace_effective[ri] = effective;
        replacement_frames += @intFromBool(effective);
    }
    var supersede_effective = false;
    const superseded_pid = superseded_id orelse 0;
    if (superseded_id != null and superseded_pid != 0 and superseded_pid != new_parent_id) {
        supersede_effective = true;
        for (replace_ids, replace_effective) |pid, effective| {
            if (effective and pid == superseded_pid) {
                supersede_effective = false;
                break;
            }
        }
        var found = false;
        for (self.units.items) |u| {
            if (u.parent_id != superseded_pid) continue;
            var active = u.gate.superseded_by == 0;
            for (replace_ids, replace_effective) |pid, effective| {
                if (effective and u.gate.superseded_by == pid) active = true;
            }
            if (active) {
                found = true;
                break;
            }
        }
        supersede_effective = supersede_effective and found;
    }
    const frame_count = std.math.add(usize, added_n, replacement_frames + @intFromBool(supersede_effective)) catch return error.OutOfMemory;

    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    try self.units.ensureUnusedCapacity(self.gpa, added_n);
    var candidate_unflushed = self.unflushed_from;

    var rb: LessonRollback = .{
        .added_ids = added_ids,
        .removed = removed,
        .removed_len = 0,
        .gates = gates,
        .gates_len = 0,
        .indexes = undefined,
        .skill = self.skill_rec,
        .skill_pols = self.skill_policy_count,
        .skill_changed = added_standing,
        .version = self.version,
        .base_saved = self.base_saved,
        .pending_deltas = self.pending_deltas,
        .unflushed_from = self.unflushed_from,
        .journal_gen = self.journal_gen,
        .last_bytes = self.last_bytes,
    };
    rb.indexes.detach(self);
    var ai: usize = 0;
    var si: usize = 0;
    while (si < self.staged.items.len) {
        if (self.staged.items[si].parent_id != new_parent_id) {
            si += 1;
            continue;
        }
        const u = self.staged.orderedRemove(si);
        added_ids[ai] = u.unit_id;
        ai += 1;
        self.units.appendAssumeCapacity(u);
    }
    std.debug.assert(ai == added_n);
    self.version = std.math.add(u64, self.version, 1) catch {
        rb.restore(self);
        return error.VersionExhausted;
    };
    for (replace_ids, replace_effective) |pid, effective| {
        if (!effective) continue;
        replaceLive(self, &rb, &candidate_unflushed, pid) catch |e| {
            rb.restore(self);
            return e;
        };
    }
    if (supersede_effective) {
        var changed = false;
        for (self.units.items) |*u| {
            if (u.parent_id != superseded_pid or u.gate.superseded_by != 0) continue;
            rb.skill_changed = rb.skill_changed or u.gate.is_standing;
            rememberGate(&rb, u);
            u.gate.superseded_by = new_parent_id;
            changed = true;
        }
        if (changed) self.version = std.math.add(u64, self.version, 1) catch {
            rb.restore(self);
            return error.VersionExhausted;
        };
    }
    if (rb.skill_changed) {
        self.skill_rec = &.{};
        self.skill_policy_count = 0;
    }
    self.rebuildBridges() catch |e| {
        rb.restore(self);
        return e;
    };

    const final_pending = std.math.add(u32, self.pending_deltas, @intCast(frame_count)) catch {
        rb.restore(self);
        return error.TooManyRecords;
    };
    const use_delta = self.calibrated and self.units.items.len > 0 and self.base_saved and
        candidate_unflushed + added_n == self.units.items.len and self.skill_rec.len == 0 and rb.skill.len == 0 and final_pending <= 16;
    if (use_delta) {
        const frames = prepareDeltaFrames(self, added_ids, replace_ids, replace_effective, superseded_pid, supersede_effective, new_parent_id, frame_count) catch |e| {
            rb.restore(self);
            return e;
        };
        defer {
            for (frames) |*fr| fr.deinit(self.gpa);
            self.gpa.free(frames);
        }
        var appended: journal.AppendResult = .{};
        journal.appendDeltaFramesWithSync(self.gpa, self.io, dpath, self.journal_gen, frames, &appended, sync_dir) catch |e| {
            if (!appended.committed) {
                rb.restore(self);
                return e;
            }
        };
        self.unflushed_from = self.units.items.len;
        self.pending_deltas = final_pending;
        self.last_bytes = std.math.add(u64, persistedBaseLength(self, path), appended.end) catch self.last_bytes;
    } else {
        var published = false;
        _ = save_mod.saveFullWithSync(self, sync_dir, &published) catch |e| {
            if (!published) {
                rb.restore(self);
                return e;
            }
        };
    }
    rb.finish(self);
    return @intCast(added_n);
}

pub const ConsolidateOutcome = struct {
    report: types.ConsolidateReport = .{},
    committed: bool = false,
};

const BuiltSkill = struct { skill: []RecLayerData, count: u32 };
fn buildSkill(b: *Bank) !BuiltSkill {
    const arch = b.backend.arch;
    if (arch.num_rec == 0) return .{ .skill = &.{}, .count = 0 };
    var count: usize = 0;
    for (b.units.items) |u| if (u.gate.isActive() and u.gate.is_standing and u.rec.len == arch.num_rec) {
        count += 1;
    };
    if (count == 0) return .{ .skill = &.{}, .count = 0 };
    const w_each = 1.0 / @sqrt(@as(f32, @floatFromInt(count)));
    const rec = try b.gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(b.gpa);
        b.gpa.free(rec);
    }
    for (0..arch.num_rec) |r| {
        const acc = try b.gpa.alloc(f32, arch.rec_heads * arch.rec_rows * arch.rec_cols);
        errdefer b.gpa.free(acc);
        @memset(acc, 0.0);
        for (b.units.items) |u| {
            if (u.gate.isActive() and u.gate.is_standing and u.rec.len == arch.num_rec) {
                compress.addScaled(u.rec[r].delta, w_each, acc);
            }
        }
        rec[r] = .{ .delta = acc, .conv = try b.gpa.alloc(u16, 0) };
        made += 1;
    }
    return .{ .skill = rec, .count = @intCast(count) };
}

/// Consolidates into a candidate, restoring all owned state if its full base cannot be published.
pub fn consolidateWithSync(
    self: *Bank,
    prune_superseded: bool,
    compact_policies: bool,
    sync_dir: *const fn (Io, []const u8) anyerror!void,
    outcome: *ConsolidateOutcome,
) !void {
    outcome.* = .{ .report = .{ .compacted_deltas = self.pending_deltas } };
    var prune_n: usize = 0;
    if (prune_superseded) {
        for (self.units.items) |u| prune_n += @intFromBool(u.gate.superseded_by != 0);
    }
    const removed = try self.gpa.alloc(Removed, prune_n);
    defer self.gpa.free(removed);
    const built: BuiltSkill = if (compact_policies) try buildSkill(self) else .{ .skill = @as([]RecLayerData, &.{}), .count = 0 };
    const new_skill = built.skill;
    var own_new_skill = true;
    defer if (own_new_skill) freeSkill(self, new_skill);

    const old_version = self.version;
    const committed_version = std.math.add(u64, old_version, 1) catch return error.VersionExhausted;
    const old_skill = self.skill_rec;
    const old_skill_pols = self.skill_policy_count;
    const old_base_saved = self.base_saved;
    const old_pending = self.pending_deltas;
    const old_unflushed = self.unflushed_from;
    const old_generation = self.journal_gen;
    const old_bytes = self.last_bytes;
    var indexes: SavedIndexes = undefined;
    indexes.detach(self);
    var removed_len: usize = 0;
    var i: usize = 0;
    while (i < self.units.items.len) {
        if (prune_superseded and self.units.items[i].gate.superseded_by != 0) {
            const u = self.units.orderedRemove(i);
            removed[removed_len] = .{ .at = i, .unit = u };
            removed_len += 1;
            if (i < self.unflushed_from) self.unflushed_from -= 1;
            continue;
        }
        i += 1;
    }
    self.skill_rec = new_skill;
    self.skill_policy_count = built.count;
    own_new_skill = false;

    const Rollback = struct {
        fn run(
            b: *Bank,
            old_indexes: *SavedIndexes,
            taken: []Removed,
            taken_len: *usize,
            skill: []RecLayerData,
            skill_pols: u32,
            version: u64,
            base_saved: bool,
            pending: u32,
            unflushed: usize,
            generation: u64,
            bytes: u64,
        ) void {
            while (taken_len.* > 0) {
                taken_len.* -= 1;
                const x = &taken[taken_len.*];
                b.units.insertAssumeCapacity(x.at, x.unit);
            }
            freeSkill(b, b.skill_rec);
            b.skill_rec = skill;
            b.skill_policy_count = skill_pols;
            b.version = version;
            b.base_saved = base_saved;
            b.pending_deltas = pending;
            b.unflushed_from = unflushed;
            b.journal_gen = generation;
            b.last_bytes = bytes;
            old_indexes.discardLiveAndRestore(b);
        }
    };
    self.rebuildBridges() catch |e| {
        Rollback.run(self, &indexes, removed, &removed_len, old_skill, old_skill_pols, old_version, old_base_saved, old_pending, old_unflushed, old_generation, old_bytes);
        return e;
    };
    var published = false;
    var commit_error: ?anyerror = null;
    _ = save_mod.saveFullWithSync(self, sync_dir, &published) catch |e| {
        if (!published) {
            Rollback.run(self, &indexes, removed, &removed_len, old_skill, old_skill_pols, old_version, old_base_saved, old_pending, old_unflushed, old_generation, old_bytes);
            return e;
        }
        commit_error = e;
    };
    outcome.committed = published;
    for (removed[0..removed_len]) |*x| x.unit.deinit(self.gpa);
    freeSkill(self, old_skill);
    indexes.discardSaved(self);
    self.version = committed_version;
    outcome.report = .{
        .units = self.activeUnitCount(),
        .pruned_superseded = @intCast(removed_len),
        .consolidated_policies = built.count,
        .compacted_deltas = old_pending,
        .clusters = @intCast(self.cluster_index.n_clusters),
        .bytes = self.last_bytes,
    };
    if (commit_error) |e| return e;
}
