//! Living-memory bank persistence writes: delta journal frames, tombstones, durable forget/clear and full saves.
const std = @import("std");
const whitener = @import("whitener.zig");
const store = @import("store.zig");
const types = @import("types.zig");
const unit_store = @import("unit_store.zig");
const Bank = @import("bank.zig").Bank;
const Io = std.Io;
const Unit = types.Unit;
const appendUnitTensors = unit_store.appendUnitTensors;
const serializeSingleUnit = unit_store.serializeSingleUnit;
var quarantine_serial: std.atomic.Value(u64) = .init(0);

/// Appends only new units to `living-weights.delta` if a base exists and < 16 frames are pending; else `saveFull`.
pub fn save(self: *Bank) !u64 {
    return saveWithSync(self, store.syncParent);
}

pub fn saveWithSync(self: *Bank, sync_dir: *const fn (Io, []const u8) anyerror!void) !u64 {
    self.last_persist_committed = false;
    if (self.calibrated and self.units.items.len > 0 and self.base_saved and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) return self.last_bytes;
    if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from < self.units.items.len and self.skill_rec.len == 0) {
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        var delta_total: u64 = 0;
        var commit_error: ?anyerror = null;
        // Once a frame is synced, its bookkeeping advances even if naming durability later reports an error.
        while (self.unflushed_from < self.units.items.len) {
            self.last_persist_committed = false;
            const blob = try serializeSingleUnit(self, self.gpa, &self.units.items[self.unflushed_from]);
            defer self.gpa.free(blob);
            var appended: store.AppendResult = .{};
            store.appendDeltaFrameWithSync(self.gpa, self.io, dpath, self.journal_gen, .add_unit, blob, &appended, sync_dir) catch |e| {
                if (!appended.committed) return e;
                commit_error = e;
            };
            delta_total = appended.end;
            self.unflushed_from += 1;
            self.last_persist_committed = appended.committed;
            self.pending_deltas += 1;
            if (commit_error != null) break;
        }
        const base_len: u64 = blk: {
            const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
            defer f.close(self.io);
            break :blk f.length(self.io) catch self.last_bytes;
        };
        self.last_bytes = base_len + delta_total;
        if (commit_error) |e| return e;
        return self.last_bytes;
    }
    var published = false;
    return saveFullWithSync(self, sync_dir, &published);
}

/// Appends a tombstone frame (forget or supersede), or compacts if no base, threshold reached or bank empty.
pub fn persistTombstone(self: *Bank, op: store.DeltaOp, pid_a: u32, pid_b: u32) !u64 {
    var committed = false;
    return persistTombstoneWithSync(self, op, pid_a, pid_b, store.syncParent, &committed);
}

fn persistTombstoneWithSync(self: *Bank, op: store.DeltaOp, pid_a: u32, pid_b: u32, sync_dir: *const fn (Io, []const u8) anyerror!void, committed: *bool) !u64 {
    self.last_persist_committed = false;
    committed.* = false;
    if (!self.calibrated or self.units.items.len == 0) return saveFullWithSync(self, sync_dir, committed);
    if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], pid_a, .little);
        std.mem.writeInt(u32, buf[4..8], pid_b, .little);
        const payload = if (op == .supersede_parent) buf[0..8] else buf[0..4];
        var appended: store.AppendResult = .{};
        self.last_persist_committed = false;
        var commit_error: ?anyerror = null;
        store.appendDeltaFrameWithSync(self.gpa, self.io, dpath, self.journal_gen, op, payload, &appended, sync_dir) catch |e| {
            if (!appended.committed) return e;
            commit_error = e;
        };
        committed.* = appended.committed;
        self.last_persist_committed = appended.committed;
        self.pending_deltas += 1;
        const base_len: u64 = blk: {
            const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
            defer f.close(self.io);
            break :blk f.length(self.io) catch self.last_bytes;
        };
        self.last_bytes = base_len + appended.end;
        if (commit_error) |e| return e;
        return self.last_bytes;
    }
    return saveFullWithSync(self, sync_dir, committed);
}

pub fn forgetDurable(self: *Bank, parent_id: u32) !u32 {
    return forgetDurableWithSync(self, parent_id, store.syncParent);
}

fn forgetDurableWithSync(self: *Bank, parent_id: u32, sync_dir: *const fn (Io, []const u8) anyerror!void) !u32 {
    var taken_n: usize = 0;
    var revived_n: usize = 0;
    for (self.units.items) |u| {
        if (u.parent_id == parent_id) taken_n += 1 else if (u.gate.superseded_by == parent_id) revived_n += 1;
    }
    if (taken_n == 0 and revived_n == 0) return 0;
    if (self.calibrated and self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
        var committed = false;
        var commit_error: ?anyerror = null;
        _ = persistTombstoneWithSync(self, .forget_parent, parent_id, 0, sync_dir, &committed) catch |e| {
            if (!committed) return e;
            commit_error = e;
        };
        const removed = self.forgetParent(parent_id);
        if (commit_error) |e| return e;
        return removed;
    }
    const Taken = struct { at: usize, unit: Unit };
    const taken = try self.gpa.alloc(Taken, taken_n);
    defer self.gpa.free(taken);
    const revived = try self.gpa.alloc(u32, revived_n);
    defer self.gpa.free(revived);
    const old_version = self.version;
    const old_policy_count = self.skill_policy_count;
    const old_unflushed = self.unflushed_from;
    var t_n: usize = 0;
    var r_n: usize = 0;
    var at: usize = 0;
    var standing = false;
    var i: usize = 0;
    while (i < self.units.items.len) : (at += 1) {
        const u = &self.units.items[i];
        if (u.parent_id == parent_id) {
            standing = standing or u.gate.is_standing;
            taken[t_n] = .{ .at = at, .unit = self.units.orderedRemove(i) };
            t_n += 1;
            if (i < self.unflushed_from) self.unflushed_from -= 1;
            continue;
        }
        if (u.gate.superseded_by == parent_id) {
            standing = standing or u.gate.is_standing;
            u.gate.superseded_by = 0;
            revived[r_n] = u.unit_id;
            r_n += 1;
        }
        i += 1;
    }
    const skill = self.skill_rec;
    if (standing) {
        self.skill_rec = &.{}; // a standing change voids the consolidated steering, as forgetParent does
        self.skill_policy_count = 0;
    }
    if (taken_n > 0 or revived_n > 0) self.version = std.math.add(u64, self.version, 1) catch {
        for (taken) |x| self.units.insertAssumeCapacity(x.at, x.unit);
        for (self.units.items) |*u| if (std.mem.indexOfScalar(u32, revived, u.unit_id) != null) {
            u.gate.superseded_by = parent_id;
        };
        self.skill_rec = skill;
        self.skill_policy_count = old_policy_count;
        self.unflushed_from = old_unflushed;
        return error.VersionExhausted;
    };
    var published = false;
    var commit_error: ?anyerror = null;
    _ = saveFullWithSync(self, sync_dir, &published) catch |e| {
        if (!published) {
            for (taken) |x| self.units.insertAssumeCapacity(x.at, x.unit);
            for (self.units.items) |*u| if (std.mem.indexOfScalar(u32, revived, u.unit_id) != null) {
                u.gate.superseded_by = parent_id;
            };
            self.skill_rec = skill;
            self.skill_policy_count = old_policy_count;
            self.unflushed_from = old_unflushed;
            self.version = old_version;
            return e;
        }
        commit_error = e;
    };
    for (taken) |*x| x.unit.deinit(self.gpa);
    if (standing) {
        for (skill) |*r| r.deinit(self.gpa);
        if (skill.len > 0) self.gpa.free(skill);
    }
    self.rebuildBridges() catch {};
    if (commit_error) |e| return e;
    return @intCast(taken_n);
}

/// Marks a replacement only when its tombstone or compacted base is durable; pre-publication errors restore every flag.
pub fn supersedeDurable(self: *Bank, old_parent_id: u32, new_parent_id: u32) !u32 {
    if (old_parent_id == 0 or new_parent_id == 0 or old_parent_id == new_parent_id) return 0;
    var affected: u32 = 0;
    var standing = false;
    for (self.units.items) |u| if (u.parent_id == old_parent_id and u.gate.superseded_by == 0) {
        affected += 1;
        standing = standing or u.gate.is_standing;
    };
    if (affected == 0) return 0;
    if (self.calibrated and self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
        var committed = false;
        var commit_error: ?anyerror = null;
        _ = persistTombstoneWithSync(self, .supersede_parent, old_parent_id, new_parent_id, store.syncParent, &committed) catch |e| {
            if (!committed) return e;
            commit_error = e;
        };
        const changed = self.supersedeParent(old_parent_id, new_parent_id);
        if (commit_error) |e| return e;
        return changed;
    }
    const touched = try self.gpa.alloc(u32, affected);
    defer self.gpa.free(touched);
    var n: usize = 0;
    for (self.units.items) |u| if (u.parent_id == old_parent_id and u.gate.superseded_by == 0) {
        touched[n] = u.unit_id;
        n += 1;
    };
    const old_version = self.version;
    const old_skill = self.skill_rec;
    const old_policy_count = self.skill_policy_count;
    if (standing) {
        self.skill_rec = &.{};
        self.skill_policy_count = 0;
    }
    const changed = self.supersedeParent(old_parent_id, new_parent_id);
    var published = false;
    var commit_error: ?anyerror = null;
    _ = saveFullWithSync(self, store.syncParent, &published) catch |e| {
        if (!published) {
            for (self.units.items) |*u| if (std.mem.indexOfScalar(u32, touched, u.unit_id) != null) {
                u.gate.superseded_by = 0;
            };
            self.skill_rec = old_skill;
            self.skill_policy_count = old_policy_count;
            self.version = old_version;
            self.rebuildBridges() catch {};
            return e;
        }
        commit_error = e;
    };
    if (standing) {
        for (old_skill) |*r| r.deinit(self.gpa);
        if (old_skill.len > 0) self.gpa.free(old_skill);
    }
    if (commit_error) |e| return e;
    return changed;
}
fn detachForClear(io: Io, path: []const u8, quarantine: []const u8) !bool {
    const st = Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false }) catch |e| switch (e) {
        error.FileNotFound => return false,
        else => return e,
    };
    if (st.kind == .directory) return error.IsDir;
    try Io.Dir.cwd().rename(path, Io.Dir.cwd(), quarantine, io);
    return true;
}

fn clearFilesWithSync(io: Io, path: []const u8, dpath: []const u8, sync_dir: *const fn (Io, []const u8) anyerror!void, committed: *bool) !void {
    committed.* = false;
    const nonce = quarantine_serial.fetchAdd(1, .monotonic);
    const stamp = std.Io.Clock.real.now(io).nanoseconds;
    const qpath = try std.fmt.allocPrint(std.heap.page_allocator, "{s}.clear-{x}-{x}", .{ path, stamp, nonce });
    defer std.heap.page_allocator.free(qpath);
    const qdpath = try std.fmt.allocPrint(std.heap.page_allocator, "{s}.clear-{x}-{x}", .{ dpath, stamp, nonce });
    defer std.heap.page_allocator.free(qdpath);
    const base_detached = try detachForClear(io, path, qpath);
    _ = detachForClear(io, dpath, qdpath) catch |e| {
        if (base_detached) Io.Dir.cwd().rename(qpath, Io.Dir.cwd(), path, io) catch |restore_error| return restore_error;
        return e;
    };
    committed.* = true;
    // Both public names are detached now; stale private quarantine files cannot replay as bank state.
    Io.Dir.cwd().deleteFile(io, qpath) catch {};
    Io.Dir.cwd().deleteFile(io, qdpath) catch {};
    try sync_dir(io, path);
}

/// Removes the bank's files as one detach transaction before changing its live units.
pub fn clearDurable(self: *Bank) !u32 {
    self.last_persist_committed = false;
    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    var committed = false;
    var commit_error: ?anyerror = null;
    clearFilesWithSync(self.io, path, dpath, store.syncParent, &committed) catch |e| {
        if (!committed) return e;
        commit_error = e;
    };
    self.last_persist_committed = committed;
    const removed: u32 = @intCast(self.units.items.len);
    for (self.units.items) |*u| u.deinit(self.gpa);
    self.units.clearRetainingCapacity();
    self.freeSkillRec();
    finishEmptySave(self);
    self.rebuildBridges() catch {};
    self.version += 1;
    if (commit_error) |e| return e;
    return removed;
}

fn finishEmptySave(self: *Bank) void {
    self.last_bytes = 0;
    self.base_saved = false;
    self.pending_deltas = 0;
    self.unflushed_from = 0;
}

fn finishFullSave(self: *Bank, dpath: []const u8, nbytes: u64) void {
    self.journal_gen += 1; // the old journal's frames are in this base: if its delete fails, it is never replayed
    Io.Dir.cwd().deleteFile(self.io, dpath) catch {};
    self.base_saved = true;
    self.pending_deltas = 0;
    self.unflushed_from = self.units.items.len;
    self.last_bytes = nbytes;
}

/// Atomically writes the bank to `living-weights.safetensors` (`tf-living-memory-v6`) and removes the journal.
pub fn saveFull(self: *Bank) !u64 {
    var published = false;
    return saveFullWithSync(self, store.syncParent, &published);
}

pub fn saveFullWithSync(self: *Bank, sync_dir: *const fn (Io, []const u8) anyerror!void, published: *bool) !u64 {
    self.last_persist_committed = false;
    published.* = false;
    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    if (!self.calibrated or self.units.items.len == 0) {
        var committed = false;
        var commit_error: ?anyerror = null;
        clearFilesWithSync(self.io, path, dpath, sync_dir, &committed) catch |e| {
            if (!committed) return e;
            commit_error = e;
        };
        published.* = committed;
        self.last_persist_committed = committed;
        finishEmptySave(self);
        if (commit_error) |e| return e;
        return 0;
    }
    const arch = self.backend.arch;
    const wl = &self.w_lex.?;
    const ws = &self.w_sem.?;
    const pref = &self.prefix_state.?;

    var arena = std.heap.ArenaAllocator.init(self.gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Build structural unit metadata JSON (NO raw text anywhere!)
    var unit_meta: std.ArrayList(u8) = .empty;
    try unit_meta.append(a, '[');
    for (self.units.items, 0..) |*u, i| {
        if (i > 0) try unit_meta.append(a, ',');
        const rank: u16 = @intFromBool(u.rec.len > 0); // 1: whole recurrent changes stored
        const has_conv: bool = u.rec.len > 0 and u.rec[0].conv.len == arch.conv_elems;
        const entry = try std.fmt.allocPrint(
            a,
            "{{\"uid\":{d},\"pid\":{d},\"tok\":{d},\"nl\":{d},\"nbl\":{d},\"ns\":{d},\"rk\":{d},\"cv\":{s},\"sm\":{s},\"st\":{s},\"dg\":{s},\"nsemb\":{d},\"ep\":{d},\"sup\":{d},\"rf\":{d}}}",
            .{
                u.unit_id,
                u.parent_id,
                u.n_tokens,
                u.gate.n_lex,
                u.gate.bodyLexCount(),
                u.gate.n_sem,
                rank,
                if (has_conv) "true" else "false",
                if (u.gate.is_small_fact) "true" else "false",
                if (u.gate.is_standing) "true" else "false",
                if (u.gate.is_digest) "true" else "false",
                @as(u32, @bitCast(u.gate.neg_sem_max)),
                u.gate.epoch,
                u.gate.superseded_by,
                u.gate.reinforcements,
            },
        );
        try unit_meta.appendSlice(a, entry);
    }
    try unit_meta.append(a, ']');

    const skill_rk: u16 = @intFromBool(self.skill_rec.len == arch.num_rec and arch.num_rec > 0); // 1: whole steering
    const meta = [_]store.MetaKV{
        .{ .key = "format", .value = store.format_tag },
        .{ .key = "family", .value = arch.family.name() },
        .{ .key = "arch", .value = try arch.signature(a) },
        .{ .key = "hidden", .value = try std.fmt.allocPrint(a, "{d}", .{arch.hidden}) },
        .{ .key = "num_attn", .value = try std.fmt.allocPrint(a, "{d}", .{arch.num_attn}) },
        .{ .key = "num_rec", .value = try std.fmt.allocPrint(a, "{d}", .{arch.num_rec}) },
        .{ .key = "layer_lex", .value = try std.fmt.allocPrint(a, "{d}", .{arch.layer_lex}) },
        .{ .key = "layer_sem", .value = try std.fmt.allocPrint(a, "{d}", .{arch.layer_sem}) },
        .{ .key = "prefix_len", .value = try std.fmt.allocPrint(a, "{d}", .{pref.len}) },
        .{ .key = "lex_k", .value = try std.fmt.allocPrint(a, "{d}", .{wl.k_pcs}) },
        .{ .key = "lex_cal", .value = try std.fmt.allocPrint(a, "{d}", .{wl.n_cal}) },
        .{ .key = "lex_floor_bits", .value = try std.fmt.allocPrint(a, "{d}", .{@as(u32, @bitCast(wl.norm_floor))}) },
        .{ .key = "sem_k", .value = try std.fmt.allocPrint(a, "{d}", .{ws.k_pcs}) },
        .{ .key = "sem_cal", .value = try std.fmt.allocPrint(a, "{d}", .{ws.n_cal}) },
        .{ .key = "sem_floor_bits", .value = try std.fmt.allocPrint(a, "{d}", .{@as(u32, @bitCast(ws.norm_floor))}) },
        .{ .key = "next_uid", .value = try std.fmt.allocPrint(a, "{d}", .{self.next_unit_id}) },
        .{ .key = "next_epoch", .value = try std.fmt.allocPrint(a, "{d}", .{self.next_epoch}) },
        .{ .key = "journal_gen", .value = try std.fmt.allocPrint(a, "{d}", .{self.journal_gen + 1}) },
        .{ .key = "skill_rk", .value = try std.fmt.allocPrint(a, "{d}", .{skill_rk}) },
        .{ .key = "skill_pols", .value = try std.fmt.allocPrint(a, "{d}", .{self.skill_policy_count}) },
        .{ .key = "units", .value = unit_meta.items },
    };

    var tensors: std.ArrayList(store.TensorSpec) = .empty;
    try tensors.append(a, .{ .name = "frame.prefix_ids", .dtype = .u32, .shape = try a.dupe(usize, &.{self.prefix_ids.len}), .bytes = std.mem.sliceAsBytes(self.prefix_ids) });
    try tensors.append(a, .{ .name = "frame.suffix_ids", .dtype = .u32, .shape = try a.dupe(usize, &.{self.suffix_ids.len}), .bytes = std.mem.sliceAsBytes(self.suffix_ids) });

    // Whiteners: mean, basis and calibration tokens, all F32 as computed
    for ([_]struct { []const u8, *const whitener.LayerWhitener }{ .{ "w.lex", wl }, .{ "w.sem", ws } }) |w| {
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.mean", .{w[0]}), .dtype = .f32, .shape = try a.dupe(usize, &.{arch.hidden}), .bytes = std.mem.sliceAsBytes(w[1].mean) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.basis", .{w[0]}), .dtype = .f32, .shape = try a.dupe(usize, &.{ w[1].k_pcs, arch.hidden }), .bytes = std.mem.sliceAsBytes(w[1].basis) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.cal", .{w[0]}), .dtype = .f32, .shape = try a.dupe(usize, &.{ w[1].n_cal, arch.hidden }), .bytes = std.mem.sliceAsBytes(w[1].cal_tokens) });
    }

    // Sleep-time consolidated spectral skill steering tensors (optional, pure weights!)
    if (skill_rk > 0) {
        for (0..arch.num_rec) |r| {
            const rl = &self.skill_rec[r];
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.d", .{r}), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows, arch.rec_cols }), .bytes = std.mem.sliceAsBytes(rl.delta) });
        }
    }

    // Memory units
    for (self.units.items, 0..) |*u, ui| {
        try appendUnitTensors(a, &tensors, arch, u, ui);
    }

    var written: store.AtomicWriteResult = .{};
    store.writeFileAtomicWithSync(self.gpa, self.io, path, &meta, tensors.items, &written, sync_dir) catch |e| {
        if (!written.published) return e;
        published.* = true;
        self.last_persist_committed = true;
        finishFullSave(self, dpath, written.bytes);
        return e;
    };
    published.* = true;
    self.last_persist_committed = true;
    finishFullSave(self, dpath, written.bytes);
    return written.bytes;
}
