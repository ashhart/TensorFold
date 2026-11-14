//! The backend's heavy vtable entries: verify, keep, draft, tree and release.
const std = @import("std");
const ds = @import("model.zig");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const greedy = @import("greedy.zig");
const Backend = @import("backend.zig").Backend;
const self = Backend.self;
const slotOf = Backend.slotOf;
const put = Backend.put;
const free = Backend.free;
const draw = Backend.draw;
const noBatchInstrument = Backend.noBatchInstrument;
const batchOn = Backend.batchOn;
const batchMax = Backend.batchMax;
const penBump = Backend.penBump;
const sampleRow = Backend.sampleRow;
const policy = @import("backend_policy.zig");
const fastArgmax = policy.fastArgmax;
const lookupTokens = policy.lookupTokens;
const noRefeedOn = policy.noRefeedOn;
const draftRowsFollowWindow = policy.draftRowsFollowWindow;
const roundSignalMode = policy.roundSignalMode;
const mtpConfirmRounds = policy.mtpConfirmRounds;
const mtpRatchetDown = policy.mtpRatchetDown;
const mtpDepthMax = policy.mtpDepthMax;
const draftGovernorOff = policy.draftGovernorOff;

pub fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
    const b = self(ptr);
    for (windows) |w| {
        if (w.parents != null or w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
    }
    // A round wider than one batched forward (TF_DSV41_BATCH_MAX, four by default) runs as several.
    const wide = batchMax();
    if (windows.len > wide and batchOn() and noBatchInstrument()) {
        var all_one = true;
        for (windows) |w| if (w.rows() != 1) {
            all_one = false;
            break;
        };
        if (all_one) {
            var done: usize = 0;
            while (done < windows.len) {
                const n = @min(windows.len - done, wide);
                try verify(ptr, windows[done..][0..n], out[done..][0..n]);
                done += n;
            }
            return;
        }
    }
    // A shared round's plain windows are one row each.
    const batch = batchOn() and windows.len > 1 and windows.len <= wide and noBatchInstrument() and blk: {
        for (windows) |w| if (w.rows() != 1) break :blk false;
        break :blk true;
    };
    var batch_logits: ?[]const f32 = null;
    var batch_feeds: [8][]u32 = undefined;
    var batch_off: [8]u32 = undefined;
    if (batch) {
        var segs_buf: [8]ds.Seg = undefined;
        var off: u32 = 0;
        var made: usize = 0;
        var begun: usize = 0;
        errdefer for (batch_feeds[0..made]) |f| b.gpa.free(f);
        errdefer for (segs_buf[0..begun]) |sg| b.model.commitSlot(sg.slot) catch {};
        for (windows, 0..) |w, wi| {
            const slot = try b.slotOf(w.stream);
            const st = &b.model.caches[slot];
            if (w.positions.len != 1 or w.positions[0] != st.position + 1) return error.PositionMismatch;
            b.model.commitSlot(slot) catch {};
            try b.model.beginSlot(slot);
            begun += 1;
            batch_feeds[wi] = try b.gpa.alloc(u32, 1);
            made += 1;
            batch_feeds[wi][0] = w.pending;
            batch_off[wi] = off;
            segs_buf[wi] = .{ .slot = slot, .ids = batch_feeds[wi], .p0 = st.position };
            off += 1;
        }
        batch_logits = try b.model.forwardSegs(segs_buf[0..windows.len], .all);
        if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("batch fwd: windows={d} gpu={d:.3} ms", .{ windows.len, b.model.last_gpu_seconds * 1e3 });
    }
    for (windows, out, 0..) |w, o, wi| {
        const i = try b.slotOf(w.stream);
        const state = &b.model.caches[i];
        const rows = w.rows();
        var feeds: []u32 = undefined;
        var opened = false;
        if (batch) {
            feeds = batch_feeds[wi];
        } else {
            if (w.positions.len != rows) return error.WindowOutOfStep;
            for (w.positions, 0..) |p, r| if (p != state.position + r + 1) return error.PositionMismatch;
            b.model.commitSlot(i) catch {}; // a window the engine accepted whole leaves its transaction to close here
            try b.model.beginSlot(i);
            opened = true;
            feeds = try b.gpa.alloc(u32, rows);
            feeds[0] = w.pending;
            const held0 = b.held[i];
            if (w.held > held0.len) return error.WindowOutOfStep;
            @memcpy(feeds[1 .. 1 + w.held], held0[0..w.held]);
            @memcpy(feeds[1 + w.held ..], w.tokens);
            // A row that is not a token (a stale host-held draft) is a wrong guess, never an error.
            for (feeds[1..]) |*t| {
                if (t.* >= 129280) t.* = feeds[0];
            }
        }
        defer b.gpa.free(feeds);
        errdefer if (opened) b.model.commitSlot(i) catch {};
        const held = b.held[i];
        if (w.held > held.len) return error.WindowOutOfStep;
        var row: []const f32 = undefined;
        if (batch_logits) |lg| {
            row = lg[@as(usize, batch_off[wi]) * 129280 ..];
        } else {
            row = try b.model.forwardOn(i, feeds, .all);
            if (std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("single fwd: slot={d} rows={d} gpu={d:.3} ms", .{ i, rows, b.model.last_gpu_seconds * 1e3 });
        }
        // TF_DSV41_ROUTE_DUMP=N: the last layer's routes of the first N windows.
        if (std.c.getenv("TF_DSV41_ROUTE_DUMP")) |raw| {
            const S = struct {
                var n: u32 = 0;
            };
            const max = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 0;
            const slots = rows * 6;
            if (S.n < max and slots > 0 and slots <= 96) {
                S.n += 1;
                const v = b.model.workspace.view(.route_ids);
                const ids: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                var sizes: [96]u32 = @splat(0);
                var distinct: u32 = 0;
                for (0..slots) |q| {
                    var first = true;
                    for (0..q) |j| if (ids[j] == ids[q]) {
                        first = false;
                        break;
                    };
                    if (!first) continue;
                    var c: u32 = 0;
                    for (0..slots) |j| if (ids[j] == ids[q]) {
                        c += 1;
                    };
                    sizes[distinct] = c;
                    distinct += 1;
                }
                std.log.info("routes: rows={d} distinct={d} sizes={any}", .{ rows, distinct, sizes[0..distinct] });
            }
        }
        // A stream with a positive temperature samples every row; the greedy path below is untouched.
        const sampled_stream: ?lanes.sampling.Sampling = if (w.stream.sampling) |sp|
            (if (sp.temperature > 0) sp else null)
        else
            null;
        // Presence/frequency penalties, upstream's semantics (lanes.Penalties, the CUDA lane's tf_penalize_rows form).
        const pen_on = w.stream.penalties.active();
        if (pen_on) {
            b.pen_counts.clearRetainingCapacity();
            const pen_start = @min(w.stream.prompt_len, w.stream.context.items.len);
            for (w.stream.context.items[pen_start..]) |t| try penBump(b.gpa, &b.pen_counts, t);
            if (std.c.getenv("TF_DSV41_PEN_DEBUG") != null) std.log.info(
                "pen: context={d} prompt={d} pending={d} last={d} rows={d} distinct={d}",
                .{ w.stream.context.items.len, w.stream.prompt_len, w.pending, if (w.stream.context.items.len > 0) w.stream.context.items[w.stream.context.items.len - 1] else 0, rows, b.pen_counts.count() },
            );
        }
        for (0..rows) |r| {
            const logits_row = row[r * 129280 ..][0..129280];
            // With penalties on, row r draws from a penalized copy of its logits.
            const target: []const f32 = if (pen_on) blk: {
                if (r > 0) try penBump(b.gpa, &b.pen_counts, feeds[r]);
                if (b.pen_row.len < 129280) {
                    if (b.pen_row.len > 0) b.gpa.free(b.pen_row);
                    b.pen_row = try b.gpa.alloc(f32, 129280);
                }
                @memcpy(b.pen_row, logits_row);
                var pen_it = b.pen_counts.iterator();
                while (pen_it.next()) |e| b.pen_row[e.key_ptr.*] -= w.stream.penalties.of(e.value_ptr.*);
                break :blk b.pen_row;
            } else logits_row;
            // The row's own logits and its top-2, for the long-prompt equivalence gate.
            if (std.c.getenv("TF_DSV41_LOGIT_DUMP")) |dv| {
                const D = struct {
                    var n: u64 = 0;
                };
                const dir = std.mem.span(dv);
                var nb: [300]u8 = undefined;
                const name = std.fmt.bufPrint(&nb, "{s}/logits-{d}-{d}.bin", .{ dir, @as(u64, @intCast(w.positions[r])), D.n }) catch "logits.bin";
                D.n += 1;
                std.Io.Dir.cwd().writeFile(b.model.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(logits_row) }) catch {};
                var m1: f32 = -std.math.inf(f32);
                var m2: f32 = -std.math.inf(f32);
                var best: u32 = 0;
                for (target, 0..) |v, id| {
                    if (!std.math.isFinite(v)) continue;
                    if (v > m1) {
                        m2 = m1;
                        m1 = v;
                        best = @intCast(id);
                    } else if (v > m2) m2 = v;
                }
                std.log.info("logitdump {s} top1={d} m1={d:.7} m2={d:.7} margin={d:.7}", .{ name, best, m1, m2, m1 - m2 });
            }
            if (sampled_stream) |sp| {
                o.sampled[r] = try sampleRow(target, sp, w.positions[r]);
            } else if (fastArgmax()) {
                const best = greedy.argmax(target);
                if (!std.math.isFinite(target[best])) return error.NonfiniteLogits;
                o.sampled[r] = @intCast(best);
            } else {
                var best: u32 = 0;
                var maximum = -std.math.inf(f32);
                for (target, 0..) |value, id| {
                    if (!std.math.isFinite(value)) return error.NonfiniteLogits;
                    if (value > maximum) {
                        maximum = value;
                        best = @intCast(id);
                    }
                }
                o.sampled[r] = best;
            }
        }
        // The target's pick per row, joined to the proposal's top-2 by the round counter.
        if (std.c.getenv("TF_DSV41_TREE_PRICE") != null) {
            std.log.info("treeprice verify round {d} rows {d} held {d} pending {d} sampled {any}", .{ ds.price_round, rows, w.held, w.pending, o.sampled[0..rows] });
            ds.price_round += 1;
        }
        // THE ADAPTIVE DEPTH POLICY'S SIGNAL, FED FROM THE VERIFY: row 1 is the first draft's acceptance.
        const first_draft: ?u32 = if (w.held > 0)
            (if (held.len > 0) held[0] else null)
        else if (w.tokens.len > 0)
            w.tokens[0]
        else
            null;
        // THE ACCEPTANCE TEST IS THE ENGINE'S OWN (core/lanes/accept.zig's acceptPath).
        const hit_dbg: bool = rows > 1 and first_draft != null and o.sampled[0] == first_draft.?;
        // The signal: the previous window's outcome, one round lagged; a whole window landing reads 1.0.
        if (i < b.gov_rounds.len) {
            if (b.gov_rows[i] > 0) {
                b.gov_rounds[i] += 1;
                b.gov_committed[i] += if (b.keep_pending[i]) b.gov_kept[i] else b.gov_rows[i];
            }
            b.gov_rows[i] = @intCast(rows);
        }
        const ratio: f64 = if (b.keep_pending[i]) b.keep_ratio[i] else 1.0;
        b.keep_pending[i] = false;
        // The ratchet: only a deep round can ratchet, and two in a row below the crossover go shallow.
        const ratchet_down = mtpRatchetDown();
        if (ratchet_down > 0 and i < b.shallow.len and b.last_depth[i] > 1) {
            if (ratio < ratchet_down) b.low_streak[i] += 1 else b.low_streak[i] = 0;
            if (b.low_streak[i] >= 2) b.shallow[i] = true;
        }
        b.model.noteAcceptRatioSlot(i, ratio);
        if (roundSignalMode() != 1) b.model.noteAcceptSlot(i, hit_dbg);
        if (std.c.getenv("TF_DSV41_MTP_ADAPT_DEBUG") != null) std.log.info(
            "mtp policy: slot {d} held {d} rows {d} sampled0 {d} first {d} hit {} signal {d:.4}",
            .{ i, w.held, rows, if (rows > 1) o.sampled[0] else 0, if (first_draft) |f| f else 0, hit_dbg, b.model.mtpAcceptSlot(i) },
        );
        // The window's fed tokens after the pending one landed: the verified drafts, GLM's contract.
        if (o.drafts.len > 0) @memcpy(o.drafts, feeds[1 .. 1 + o.drafts.len]);
        @memcpy(b.last_sampled[0..rows], o.sampled[0..rows]);
    }
}

pub fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
    const b = self(ptr);
    for (windows, paths) |w, path| {
        const i = try b.slotOf(w.stream);
        const rows = w.rows();
        const kept = path.len;
        if (i < b.gov_kept.len) b.gov_kept[i] = @intCast(kept);
        b.keep_pending[i] = true;
        b.keep_ratio[i] = if (rows > 1) @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(rows - 1)) else 1.0;
        // THE POLICY'S SIGNAL IS TAKEN WHERE THE KEPT COUNT IS KNOWN: portKeep missed the whole-window case.
        if (kept >= rows) continue; // the whole window landed
        try b.model.rollbackSlot(i);
        // The ring rows the window's rejected rows published are aliases the next forward can still read.
        try b.model.restoreRing(i, @intCast(kept), @intCast(rows));
        if (kept == 0) continue; // the rollback closed the transaction
        var feeds = try b.gpa.alloc(u32, kept);
        defer b.gpa.free(feeds);
        feeds[0] = w.pending;
        const held = b.held[i];
        const held_kept = @min(w.held, kept - 1);
        if (held_kept > held.len) return error.WindowOutOfStep;
        @memcpy(feeds[1 .. 1 + held_kept], held[0..held_kept]);
        @memcpy(feeds[1 + held_kept ..], w.tokens[0 .. kept - 1 - held_kept]);
        for (feeds[1..]) |*t| {
            if (t.* >= 129280) t.* = feeds[0];
        }
        if (noRefeedOn()) {
            // The no-re-feed port.
            try b.model.portKeep(i, @intCast(kept));
            b.model.commitSlot(i) catch {};
            continue;
        }
        // Diagnostic (TF_DSV41_CLEAR_STALE=1): the rejected rows the window wrote stay in the ring; zero them.
        if (std.c.getenv("TF_DSV41_CLEAR_PARTIALS") != null) b.model.caches[i].clearPartials();
        if (std.c.getenv("TF_DSV41_CLEAR_STALE") != null) {
            const start = b.model.caches[i].position;
            b.model.caches[i].clearRows(start + kept, start + rows);
        }
        _ = b.model.forwardOn(i, feeds, .last) catch |e| {
            std.log.err("keep: refeed failed: {s}", .{@errorName(e)});
            return e;
        };
        // The refeed's own forward closed the transaction; commit only if one is still open.
        b.model.commitSlot(i) catch {};
    }
}

/// Nothing to absorb: the lookup drafter reads the stream's own context each round.
pub fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
    const b = self(ptr);
    for (requests) |r| {
        const i = b.by.get(r.stream) orelse continue;
        if (b.held[i].len > 0) b.gpa.free(b.held[i]);
        b.held[i] = &.{};
        if (r.depth == 0) continue;
        if (draftGovernorOff(b, i)) {
            // The switch must reach the lane planner, not only the held list.
            r.stream.drafts = false;
            continue;
        }
        const ctx = r.stream.context.items;
        // Prefer a matching CPU suffix arm before paying for a trained forward.
        var copied: [policy.max_lookup_drafts]u32 = undefined;
        const lookup_cap = @min(r.depth, policy.lookupDepth());
        const got = if (std.c.getenv("TF_DSV41_DRAFT_COPY") == null)
            lookupTokens(ctx, copied[0..lookup_cap])
        else
            0;
        if (got > 0) {
            const out = try b.gpa.alloc(u32, r.depth);
            @memcpy(out[0..got], copied[0..got]);
            for (out[got..]) |*t| t.* = copied[got - 1];
            b.held[i] = out;
            continue;
        }
        // The trained proposer (TF_DSV41_MTP_DRAFTS) runs HERE, between rounds.
        const last_tok: ?u32 = r.stream.pending orelse
            (if (ctx.len > 0) ctx[ctx.len - 1] else null);
        if (b.model.mtp_enabled and b.model.weights.dspark != null and
            last_tok != null and r.depth <= mtpDepthMax())
        mtp: {
            // Both hooks reuse this pending token's proposal, bounded by the trained head's capacity.
            const count = @min(mtpDepthMax(), if (draftRowsFollowWindow()) b.adaptiveDepth(@intCast(i)) else 3);
            if (count == 0) break :mtp;
            const drafts = b.model.proposeTokens(@intCast(i), last_tok.?, count) catch |err| {
                std.log.warn("mtp: drafts unavailable ({s})", .{@errorName(err)});
                break :mtp;
            };
            // The engine plans its window from r.depth, so the hand-off must always carry that many rows.
            const out = try b.gpa.alloc(u32, r.depth);
            const n = @min(r.depth, count);
            @memcpy(out[0..n], drafts[0..n]);
            for (out[n..]) |*t| t.* = drafts[n - 1];
            b.held[i] = out;
            // The round's other hook (tree) asks for the same drafts; keep them here.
            if (i < b.mtp_cache.len) b.mtp_cache[i] = .{ .pending = last_tok.?, .drafts = drafts, .count = count };
            continue;
        }
        // Diagnostic (TF_DSV41_DRAFT_COPY=1): every draft repeats the pending token, so a two-row window is [pending.
        const want: usize = r.depth;
        if (std.c.getenv("TF_DSV41_DRAFT_COPY") != null) {
            const out0 = try b.gpa.alloc(u32, want);
            const pend = r.stream.pending orelse 0;
            for (out0) |*t| t.* = pend;
            b.held[i] = out0;
            continue;
        }
        const out = try b.gpa.alloc(u32, want);
        for (out) |*t| t.* = r.stream.pending orelse 0;
        b.held[i] = out;
    }
}

/// A suffix lookup drafter (TF_DSV41_LOOKUP_DRAFTS=<depth>).
pub fn tree(ptr: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
    const b = self(ptr);
    // The proposer's forward needs the lane's slot (the trunk's key ring lives there).
    const slot_i = b.by.get(s) orelse return null;
    if (draftGovernorOff(b, slot_i)) {
        s.drafts = false;
        return null;
    }
    var depth = policy.maxDraftDepth(b);
    // THE ADAPTIVE DEPTH POLICY: yield from the accepted drafts selects the cheap depth-1 or the deep window.
    const adapt_raw = std.c.getenv("TF_DSV41_MTP_ADAPT");
    const adapt_off = adapt_raw != null and std.mem.eql(u8, std.mem.span(adapt_raw.?), "0");
    if (!adapt_off) {
        const prev: u32 = if (slot_i < b.last_depth.len) b.last_depth[slot_i] else 0;
        depth = b.adaptiveDepth(slot_i);
        // THE CONFIRMED SWITCH-UP: one sample is not a change of text, so the up-transition must hold `confirm` rounds.
        const confirm = mtpConfirmRounds();
        if (confirm > 1 and slot_i < b.up_streak.len) {
            if (depth > 1 and prev < 2) {
                b.up_streak[slot_i] += 1;
                if (b.up_streak[slot_i] < confirm) depth = 1;
            } else if (depth == 1) {
                b.up_streak[slot_i] = 0;
            }
        }
        if (slot_i < b.last_depth.len) b.last_depth[slot_i] = depth;
        if (std.c.getenv("TF_DSV41_MTP_ADAPT_DEBUG") != null) std.log.info(
            "mtp policy: slot {d} rate {d:.4} prev {d} depth {d} streak {d}",
            .{ slot_i, b.model.mtpAcceptSlot(slot_i), prev, depth, if (slot_i < b.up_streak.len) b.up_streak[slot_i] else 0 },
        );
    }
    if (depth == 0) return null;
    var copied: [policy.max_lookup_drafts]u32 = undefined;
    const got = lookupTokens(s.context.items, copied[0..@min(depth, policy.lookupDepth())]);
    if (got > 0) return .{ .count = @intCast(got), .tokens = try gpa.dupe(u32, copied[0..got]) };
    // The trained proposer (TF_DSV41_MTP_DRAFTS) is the drafter when it is bound and has a pending token.
    if (depth > 0 and b.model.mtp_enabled and b.model.weights.dspark != null and s.pending != null) {
        // The round's other hook (draft) already ran the proposer's forward for this pending token.
        const n: usize = @min(depth, mtpDepthMax());
        var drafts: [4]u32 = undefined;
        var cached = false;
        if (slot_i < b.mtp_cache.len) if (b.mtp_cache[slot_i]) |c| if (c.pending == s.pending.? and c.count >= n) {
            drafts = c.drafts;
            cached = true;
        };
        if (!cached) {
            drafts = b.model.proposeTokens(@intCast(slot_i), s.pending.?, @intCast(n)) catch |err| {
                std.log.warn("mtp: proposer unavailable ({s})", .{@errorName(err)});
                return null;
            };
        }
        // The depth this round runs at, per slot: the adaptive policy's state.
        if (slot_i < b.last_depth.len) b.last_depth[slot_i] = @intCast(n);
        const out = try gpa.alloc(u32, n);
        @memcpy(out, drafts[0..n]);
        return .{ .count = @intCast(n), .tokens = out };
    }
    return null;
}

pub fn release(ptr: *anyopaque, s: *Stream) void {
    const b = self(ptr);
    if (b.by.get(s)) |i| {
        if (b.held[i].len > 0) b.gpa.free(b.held[i]);
        b.held[i] = &.{};
    }
    const entry = b.by.fetchRemove(s) orelse return;
    b.model.resetAcceptSlot(entry.value);
    if (entry.value < b.last_depth.len) b.last_depth[entry.value] = 0;
    if (entry.value < b.shallow.len) {
        b.shallow[entry.value] = false;
        b.low_streak[entry.value] = 0;
    }
    if (entry.value < b.gov_off.len) {
        b.gov_off[entry.value] = false;
        b.gov_rounds[entry.value] = 0;
        b.gov_committed[entry.value] = 0;
        b.gov_rows[entry.value] = 0;
        b.gov_kept[entry.value] = 0;
    }
    b.model.resetSlot(entry.value) catch {};
}
