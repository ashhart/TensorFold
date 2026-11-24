//! Living-memory bank loading: reads the base file, recompiles or reads the prefix and replays the delta journal.
const std = @import("std");
const store = @import("store.zig");
const types = @import("types.zig");
const unit_store = @import("unit_store.zig");
const Bank = @import("bank.zig").Bank;
const RawState = types.RawState;
const RecLayerData = types.RecLayerData;
const loadUnitFromObject = unit_store.loadUnitFromObject;
const loadWhitener = unit_store.loadWhitener;
const parseUsizeMeta = unit_store.parseUsizeMeta;
const parseBitsMeta = unit_store.parseBitsMeta;

/// The prefix state a load reads or recompiles; freed here on failure, the caller's once returned.
fn loadPrefix(self: *Bank, lf: *const store.LoadedFile, prefix_len: usize) !RawState {
    const arch = self.backend.arch;
    var pref = try RawState.init(self.gpa, arch, @intCast(prefix_len));
    errdefer pref.deinit(self.gpa);
    var name_buf: [96]u8 = undefined;
    if (lf.getTensor("p.kv.0.k") != null) {
        const p_kv_elems = arch.kvLayerElems(prefix_len);
        for (0..arch.num_attn) |l| {
            const tk = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.k", .{l})) orelse return error.MissingTensor;
            if (tk.bytes.len != p_kv_elems * 2) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.kv_k[l]), tk.bytes);
            const tv = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.v", .{l})) orelse return error.MissingTensor;
            if (tv.bytes.len != p_kv_elems * 2) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.kv_v[l]), tv.bytes);
        }
        const st_elems = arch.recStateElems();
        for (0..arch.num_rec) |r| {
            const ts = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.st", .{r})) orelse return error.MissingTensor;
            if (ts.bytes.len != st_elems * 4) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.rec_state[r]), ts.bytes);
            const tc = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.cv", .{r})) orelse return error.MissingTensor;
            if (tc.bytes.len != arch.conv_elems * 2) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.rec_conv[r]), tc.bytes);
        }
    } else {
        if (self.prefix_ids.len != prefix_len) return error.BadLivingWeights;
        try self.backend.compileState(self.backend.ctx, null, self.prefix_ids, &pref);
    }
    return pref;
}

/// Loads `living-weights.safetensors` (v6, v4 or v3), replays `living-weights.delta`, rebuilds prefix and indexes.
pub fn loadIfPresent(self: *Bank) !bool {
    const path = try self.filePath(self.gpa);
    defer self.gpa.free(path);
    if (!try store.probeFoldedMemoryFile(self.gpa, self.io, path)) return false; // unreadable errs, not empty

    var lf = try store.readFile(self.gpa, self.io, path);
    defer lf.deinit();

    const fmt = lf.getMeta("format") orelse return error.BadLivingWeights;
    // v3, v4 and v6 banks stored Q4/Q8 KV and Q8 prototypes: not as computed, so refused by name
    if (store.inexactFormat(fmt)) return error.LivingWeightsInexact;
    if (!std.mem.eql(u8, fmt, store.format_tag)) return error.BadLivingWeights;

    const arch = self.backend.arch;
    const hidden = try parseUsizeMeta(&lf, "hidden");
    const num_attn = try parseUsizeMeta(&lf, "num_attn");
    const num_rec = try parseUsizeMeta(&lf, "num_rec");
    if (hidden != arch.hidden or num_attn != arch.num_attn or num_rec != arch.num_rec) {
        return error.LivingWeightsArchMismatch;
    }

    self.clearMemory();
    errdefer self.clearMemory();

    const prefix_len = try parseUsizeMeta(&lf, "prefix_len");
    const lex_k = try parseUsizeMeta(&lf, "lex_k");
    const lex_cal = try parseUsizeMeta(&lf, "lex_cal");
    const lex_floor = try parseBitsMeta(&lf, "lex_floor_bits");
    const sem_k = try parseUsizeMeta(&lf, "sem_k");
    const sem_cal = try parseUsizeMeta(&lf, "sem_cal");
    const sem_floor = try parseBitsMeta(&lf, "sem_floor_bits");
    self.next_unit_id = @intCast(try parseUsizeMeta(&lf, "next_uid"));
    self.next_epoch = if (lf.getMeta("next_epoch")) |s| (std.fmt.parseInt(u64, s, 10) catch 1) else 1;
    self.journal_gen = if (lf.getMeta("journal_gen")) |s| std.fmt.parseInt(u64, s, 10) catch return error.BadLivingWeights else 0;

    self.prefix_ids = try lf.copyU32(self.gpa, "frame.prefix_ids");
    self.suffix_ids = try lf.copyU32(self.gpa, "frame.suffix_ids");

    // each read whole before the bank owns it: from then on only clearMemory frees it
    self.w_lex = try loadWhitener(&lf, self.gpa, "w.lex", lex_k, lex_cal, lex_floor, arch.hidden);
    self.w_sem = try loadWhitener(&lf, self.gpa, "w.sem", sem_k, sem_cal, sem_floor, arch.hidden);

    // Recompile static prefix_state on load for v4/v6 (0 bytes on disk!); read from disk for legacy v3
    self.prefix_state = try loadPrefix(self, &lf, prefix_len);
    var name_buf: [96]u8 = undefined;

    // Optional sleep-time consolidated spectral skill steering
    const skill_rk: u16 = if (lf.getMeta("skill_rk")) |s| (std.fmt.parseInt(u16, s, 10) catch 0) else 0;
    const skill_pols: u32 = if (lf.getMeta("skill_pols")) |s| (std.fmt.parseInt(u32, s, 10) catch 0) else 0;
    if (skill_rk > 0 and arch.num_rec > 0) {
        const arr = try self.gpa.alloc(RecLayerData, arch.num_rec);
        var rc_done: usize = 0;
        errdefer {
            for (arr[0..rc_done]) |*r| r.deinit(self.gpa);
            self.gpa.free(arr);
        }
        const rk_u: usize = skill_rk;
        for (0..arch.num_rec) |r| {
            const fu = try lf.copyF32(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.u", .{r}), arch.rec_heads * arch.rec_rows * rk_u);
            errdefer self.gpa.free(fu);
            const fv = try lf.copyF32(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.v", .{r}), arch.rec_heads * rk_u * arch.rec_cols);
            errdefer self.gpa.free(fv);
            const cv = try self.gpa.alloc(u16, 0);
            arr[r] = .{ .rank = skill_rk, .u = fu, .v = fv, .conv = cv };
            rc_done += 1;
        }
        self.skill_rec = arr;
        self.skill_policy_count = skill_pols;
    }

    const units_json = lf.getMeta("units") orelse return error.BadLivingWeights;
    var parse_arena = std.heap.ArenaAllocator.init(self.gpa);
    defer parse_arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, parse_arena.allocator(), units_json, .{});
    if (parsed != .array) return error.BadLivingWeights;

    for (parsed.array.items, 0..) |item, ui| {
        var u = try loadUnitFromObject(self.gpa, &lf, arch, item, ui);
        errdefer u.deinit(self.gpa);
        if (u.gate.epoch >= self.next_epoch) self.next_epoch = u.gate.epoch + 1;
        try self.units.append(self.gpa, u);
    }

    self.calibrated = true;
    self.base_saved = true;
    self.last_bytes = lf.raw.len;

    // Replay any uncompacted crash-safe delta journal frames (`living-weights.delta`)!
    const dpath = try self.deltaPath(self.gpa);
    defer self.gpa.free(dpath);
    const frames = try store.readDeltaFrames(self.gpa, self.io, dpath, self.journal_gen);
    defer {
        for (frames) |*fr| fr.deinit(self.gpa);
        self.gpa.free(frames);
    }
    for (frames) |fr| {
        switch (fr.op) {
            .add_unit => {
                const owned = try self.gpa.dupe(u8, fr.payload);
                var dlf = store.parseOwnedBytes(self.gpa, owned) catch {
                    self.gpa.free(owned);
                    continue;
                };
                defer dlf.deinit();
                const uj = dlf.getMeta("unit") orelse continue;
                var ua = std.heap.ArenaAllocator.init(self.gpa);
                defer ua.deinit();
                const up = std.json.parseFromSliceLeaky(std.json.Value, ua.allocator(), uj, .{}) catch continue;
                var u = loadUnitFromObject(self.gpa, &dlf, arch, up, 0) catch continue;
                if (u.unit_id >= self.next_unit_id) self.next_unit_id = u.unit_id + 1;
                if (u.gate.epoch >= self.next_epoch) self.next_epoch = u.gate.epoch + 1;
                self.units.append(self.gpa, u) catch |e| {
                    u.deinit(self.gpa); // not the bank's yet: freed here, once
                    return e;
                };
            },
            .forget_parent => {
                if (fr.payload.len >= 4) {
                    const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                    _ = forgetParentNoRebuild(self, pid);
                }
            },
            .supersede_parent => {
                if (fr.payload.len >= 8) {
                    const old_pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                    const new_pid = std.mem.readInt(u32, fr.payload[4..8], .little);
                    for (self.units.items) |*u| {
                        if (u.parent_id == old_pid and u.gate.superseded_by == 0) {
                            u.gate.superseded_by = new_pid;
                        }
                    }
                }
            },
            .reinforce_parent => {
                if (fr.payload.len >= 4) {
                    const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                    _ = self.reinforceParent(pid);
                }
            },
            .generation => {}, // only a journal's first frame, which the read checks and does not return
        }
        self.pending_deltas += 1;
        self.last_bytes += fr.payload.len + 24;
    }
    self.unflushed_from = self.units.items.len;
    try self.rebuildBridges();
    return true;
}

fn forgetParentNoRebuild(self: *Bank, parent_id: u32) u32 {
    var removed: u32 = 0;
    var i: usize = 0;
    while (i < self.units.items.len) {
        if (self.units.items[i].parent_id == parent_id) {
            self.units.items[i].deinit(self.gpa);
            _ = self.units.orderedRemove(i);
            removed += 1;
        } else {
            if (self.units.items[i].gate.superseded_by == parent_id) {
                self.units.items[i].gate.superseded_by = 0;
            }
            i += 1;
        }
    }
    return removed;
}
