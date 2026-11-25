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

fn checkedProduct(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.BadLivingWeights;
}

fn advanceUnitId(next: *u32, uid: u32) !void {
    if (uid < next.*) return;
    next.* = std.math.add(u32, uid, 1) catch return error.BadLivingWeights;
}

fn advanceEpoch(next: *u64, epoch: u64) !void {
    if (epoch < next.*) return;
    next.* = std.math.add(u64, epoch, 1) catch return error.BadLivingWeights;
}

/// The prefix state a load reads or recompiles; freed here on failure, the caller's once returned.
fn loadPrefix(self: *Bank, lf: *const store.LoadedFile, prefix_len: usize) !RawState {
    const arch = self.backend.arch;
    const plen_u32 = std.math.cast(u32, prefix_len) orelse return error.BadLivingWeights;
    var pref = try RawState.init(self.gpa, arch, plen_u32);
    errdefer pref.deinit(self.gpa);
    var name_buf: [96]u8 = undefined;
    if (lf.getTensor("p.kv.0.k") != null) {
        const p_kv_elems = arch.kvLayerElems(prefix_len);
        for (0..arch.num_attn) |l| {
            const tk = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.k", .{l})) orelse return error.MissingTensor;
            if (tk.bytes.len != try checkedProduct(p_kv_elems, 2)) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.kv_k[l]), tk.bytes);
            const tv = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.kv.{d}.v", .{l})) orelse return error.MissingTensor;
            if (tv.bytes.len != try checkedProduct(p_kv_elems, 2)) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.kv_v[l]), tv.bytes);
        }
        const st_elems = arch.recStateElems();
        for (0..arch.num_rec) |r| {
            const ts = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.st", .{r})) orelse return error.MissingTensor;
            if (ts.bytes.len != try checkedProduct(st_elems, 4)) return error.BadTensorShape;
            @memcpy(std.mem.sliceAsBytes(pref.rec_state[r]), ts.bytes);
            const tc = lf.getTensor(try std.fmt.bufPrint(&name_buf, "p.rec.{d}.cv", .{r})) orelse return error.MissingTensor;
            if (tc.bytes.len != try checkedProduct(arch.conv_elems, 2)) return error.BadTensorShape;
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
    // the whole geometry, not only three counts: a bank from another model or revision of its shape is refused
    const saved_arch = lf.getMeta("arch") orelse return error.LivingWeightsArchMismatch;
    var arch_buf: [512]u8 = undefined;
    var arch_fba = std.heap.FixedBufferAllocator.init(&arch_buf);
    if (!std.mem.eql(u8, saved_arch, try arch.signature(arch_fba.allocator()))) return error.LivingWeightsArchMismatch;
    if (lf.getMeta("data_crc32") == null) return error.LivingWeightsUnchecked; // every v7 base carries its data's CRC
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
    // These are the maxima used by ensureCalibrated; larger values cannot come from this writer.
    if (lex_k > 16 or lex_cal > 560 or sem_k > 32 or sem_cal > 480) return error.BadLivingWeights;
    const lex_basis = try checkedProduct(lex_k, arch.hidden);
    const lex_calibration = try checkedProduct(lex_cal, arch.hidden);
    const sem_basis = try checkedProduct(sem_k, arch.hidden);
    const sem_calibration = try checkedProduct(sem_cal, arch.hidden);
    for ([_]usize{ lex_basis, lex_calibration, sem_basis, sem_calibration }) |elems| {
        _ = try checkedProduct(elems, @sizeOf(f32));
    }
    self.next_unit_id = std.math.cast(u32, try parseUsizeMeta(&lf, "next_uid")) orelse return error.BadLivingWeights;
    self.journal_gen = if (lf.getMeta("journal_gen")) |s| std.fmt.parseInt(u64, s, 10) catch return error.BadLivingWeights else 0;
    self.next_epoch = if (lf.getMeta("next_epoch")) |s| std.fmt.parseInt(u64, s, 10) catch return error.BadLivingWeights else 1;

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
    if (skill_rk > 1) return error.BadLivingWeights; // 1: the whole steering is stored
    if (skill_rk == 1 and arch.num_rec > 0) {
        const rec_rows = try checkedProduct(arch.rec_heads, arch.rec_rows);
        const rec_elems = try checkedProduct(rec_rows, arch.rec_cols);
        _ = try checkedProduct(rec_elems, @sizeOf(f32));
        const arr = try self.gpa.alloc(RecLayerData, arch.num_rec);
        var rc_done: usize = 0;
        errdefer {
            for (arr[0..rc_done]) |*r| r.deinit(self.gpa);
            self.gpa.free(arr);
        }
        for (0..arch.num_rec) |r| {
            const d = try lf.copyF32(self.gpa, try std.fmt.bufPrint(&name_buf, "sk.rc.{d}.d", .{r}), rec_elems);
            errdefer self.gpa.free(d);
            arr[r] = .{ .delta = d, .conv = try self.gpa.alloc(u16, 0) };
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
    const kv_per_token = try checkedProduct(arch.num_attn, arch.kv_heads);
    const kv_elems = try checkedProduct(kv_per_token, arch.head_dim);
    const min_unit_kv_bytes = try checkedProduct(kv_elems, 4); // K and V, two bytes each, at least one token
    if (min_unit_kv_bytes == 0 or parsed.array.items.len > lf.raw.len / min_unit_kv_bytes) return error.BadLivingWeights;

    for (parsed.array.items, 0..) |item, ui| {
        var u = try loadUnitFromObject(self.gpa, &lf, arch, item, ui);
        errdefer u.deinit(self.gpa);
        try advanceUnitId(&self.next_unit_id, u.unit_id);
        try advanceEpoch(&self.next_epoch, u.gate.epoch);
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
                var dlf = store.parseOwnedBytes(self.gpa, owned) catch |e| {
                    self.gpa.free(owned);
                    return e;
                };
                defer dlf.deinit();
                const uj = dlf.getMeta("unit") orelse return error.MissingMetadata;
                var ua = std.heap.ArenaAllocator.init(self.gpa);
                defer ua.deinit();
                const up = try std.json.parseFromSliceLeaky(std.json.Value, ua.allocator(), uj, .{});
                var u = try loadUnitFromObject(self.gpa, &dlf, arch, up, 0);
                errdefer u.deinit(self.gpa);
                try advanceUnitId(&self.next_unit_id, u.unit_id);
                try advanceEpoch(&self.next_epoch, u.gate.epoch);
                try self.units.append(self.gpa, u);
            },
            .forget_parent => {
                if (fr.payload.len != 4) return error.BadDeltaFrame;
                const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                _ = forgetParentNoRebuild(self, pid);
            },
            .supersede_parent => {
                if (fr.payload.len != 8) return error.BadDeltaFrame;
                const old_pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                const new_pid = std.mem.readInt(u32, fr.payload[4..8], .little);
                for (self.units.items) |*u| {
                    if (u.parent_id == old_pid and u.gate.superseded_by == 0) {
                        u.gate.superseded_by = new_pid;
                    }
                }
            },
            .reinforce_parent => {
                if (fr.payload.len != 4) return error.BadDeltaFrame;
                const pid = std.mem.readInt(u32, fr.payload[0..4], .little);
                _ = self.reinforceParent(pid);
            },
            .generation => {}, // only a journal's first frame, which the read checks and does not return
        }
        self.pending_deltas = std.math.add(u32, self.pending_deltas, 1) catch return error.BadDeltaFrame;
        self.last_bytes = std.math.add(u64, self.last_bytes, std.math.add(u64, fr.payload.len, 24) catch return error.BadDeltaFrame) catch return error.BadDeltaFrame;
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
