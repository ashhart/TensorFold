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
const removeIfPresent = unit_store.removeIfPresent;
const serializeSingleUnit = unit_store.serializeSingleUnit;

/// Appends only new units to `living-weights.delta` if a base exists and < 16 frames are pending; else `saveFull`.
pub fn save(self: *Bank) !u64 {
    if (!self.calibrated or self.units.items.len == 0) {
        return self.saveFull();
    }
    if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from < self.units.items.len and self.skill_rec.len == 0) {
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        var delta_total: u64 = 0;
        // each unit counts as flushed once its frame is durable: a retry after a failure appends no unit twice
        while (self.unflushed_from < self.units.items.len) : (self.unflushed_from += 1) {
            const blob = try serializeSingleUnit(self, self.gpa, &self.units.items[self.unflushed_from]);
            defer self.gpa.free(blob);
            delta_total = try store.appendDeltaFrame(self.gpa, self.io, dpath, self.journal_gen, .add_unit, blob);
            self.pending_deltas += 1;
        }
        self.unflushed_from = self.units.items.len;
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        const base_len: u64 = blk: {
            const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
            defer f.close(self.io);
            break :blk f.length(self.io) catch self.last_bytes;
        };
        self.last_bytes = base_len + delta_total;
        return self.last_bytes;
    }
    return self.saveFull();
}

/// Appends a tombstone frame (forget or supersede), or compacts if no base, threshold reached or bank empty.
pub fn persistTombstone(self: *Bank, op: store.DeltaOp, pid_a: u32, pid_b: u32) !u64 {
    if (!self.calibrated or self.units.items.len == 0) return self.saveFull();
    if (self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
        const dpath = try self.deltaPath(self.gpa);
        defer self.gpa.free(dpath);
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], pid_a, .little);
        std.mem.writeInt(u32, buf[4..8], pid_b, .little);
        const payload = if (op == .supersede_parent) buf[0..8] else buf[0..4];
        const delta_total = try store.appendDeltaFrame(self.gpa, self.io, dpath, self.journal_gen, op, payload);
        self.pending_deltas += 1;
        const path = try self.filePath(self.gpa);
        defer self.gpa.free(path);
        const base_len: u64 = blk: {
            const f = Io.Dir.cwd().openFile(self.io, path, .{}) catch break :blk self.last_bytes;
            defer f.close(self.io);
            break :blk f.length(self.io) catch self.last_bytes;
        };
        self.last_bytes = base_len + delta_total;
        return self.last_bytes;
    }
    return self.saveFull();
}

/// Forgets `parent_id` on disk before in memory: on a failed write the bank is exactly as it was, and errs.
pub fn forgetDurable(self: *Bank, parent_id: u32) !u32 {
    var taken_n: usize = 0;
    var revived_n: usize = 0;
    for (self.units.items) |u| {
        if (u.parent_id == parent_id) taken_n += 1 else if (u.gate.superseded_by == parent_id) revived_n += 1;
    }
    if (taken_n == 0 and revived_n == 0) return 0;
    // persistTombstone's own journal test: under it the tombstone is appended, never a base of the unchanged bank
    if (self.calibrated and self.base_saved and self.pending_deltas < 16 and self.unflushed_from == self.units.items.len and self.skill_rec.len == 0) {
        _ = try self.persistTombstone(.forget_parent, parent_id, 0); // the journal path: durable, then applied
        return self.forgetParent(parent_id);
    }
    // the compacted path: the units set aside (room taken first), the base written, then kept out or put back
    const Taken = struct { at: usize, unit: Unit };
    const taken = try self.gpa.alloc(Taken, taken_n);
    defer self.gpa.free(taken);
    const revived = try self.gpa.alloc(u32, revived_n);
    defer self.gpa.free(revived);
    const unflushed = self.unflushed_from;
    var t_n: usize = 0;
    var r_n: usize = 0;
    var at: usize = 0;
    var i: usize = 0;
    var standing = false;
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
    if (standing) self.skill_rec = &.{}; // a standing change voids the consolidated steering, as forgetParent does
    _ = self.saveFull() catch |e| {
        for (taken) |x| self.units.insertAssumeCapacity(x.at, x.unit); // ascending: each lands where it was
        for (self.units.items) |*u| if (std.mem.indexOfScalar(u32, revived, u.unit_id) != null) {
            u.gate.superseded_by = parent_id;
        };
        self.skill_rec = skill;
        self.unflushed_from = unflushed;
        return e;
    };
    for (taken) |*x| x.unit.deinit(self.gpa);
    if (standing) {
        for (skill) |*r| r.deinit(self.gpa);
        if (skill.len > 0) self.gpa.free(skill);
    }
    self.rebuildBridges() catch {};
    return @intCast(taken_n);
}

/// Removes the bank's files, then its units: when a file cannot go, the bank keeps every unit and says why.
pub fn clearDurable(self: *Bank) !u32 {
    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    try removeIfPresent(self.io, path);
    try removeIfPresent(self.io, dpath);
    const removed: u32 = @intCast(self.units.items.len);
    for (self.units.items) |*u| u.deinit(self.gpa);
    self.units.clearRetainingCapacity();
    self.freeSkillRec();
    self.last_bytes = 0;
    self.base_saved = false;
    self.pending_deltas = 0;
    self.unflushed_from = 0;
    self.rebuildBridges() catch {};
    self.version += 1;
    return removed;
}

/// Atomically writes the bank to `living-weights.safetensors` (`tf-living-memory-v6`) and removes the journal.
pub fn saveFull(self: *Bank) !u64 {
    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    if (!self.calibrated or self.units.items.len == 0) {
        try removeIfPresent(self.io, path); // emptied on disk too, or an error: never success with files left
        try removeIfPresent(self.io, dpath);
        self.last_bytes = 0;
        self.base_saved = false;
        self.pending_deltas = 0;
        self.unflushed_from = 0;
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
        const rank: u16 = if (u.rec.len > 0) u.rec[0].rank else 0;
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

    const skill_rk: u16 = if (self.skill_rec.len == arch.num_rec and arch.num_rec > 0) self.skill_rec[0].rank else 0;
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
        const rk_u: usize = skill_rk;
        for (0..arch.num_rec) |r| {
            const rl = &self.skill_rec[r];
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.u", .{r}), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, arch.rec_rows, rk_u }), .bytes = std.mem.sliceAsBytes(rl.u) });
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "sk.rc.{d}.v", .{r}), .dtype = .f32, .shape = try a.dupe(usize, &.{ arch.rec_heads, rk_u, arch.rec_cols }), .bytes = std.mem.sliceAsBytes(rl.v) });
        }
    }

    // Memory units
    for (self.units.items, 0..) |*u, ui| {
        try appendUnitTensors(a, &tensors, arch, u, ui);
    }

    const nbytes = try store.writeFileAtomic(self.gpa, self.io, path, &meta, tensors.items);
    self.journal_gen += 1; // the old journal's frames are in this base: if its delete fails, it is never replayed
    Io.Dir.cwd().deleteFile(self.io, dpath) catch {};
    self.base_saved = true;
    self.pending_deltas = 0;
    self.unflushed_from = self.units.items.len;
    self.last_bytes = nbytes;
    return nbytes;
}
