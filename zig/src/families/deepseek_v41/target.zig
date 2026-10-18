//! The GPU target: draft/iface.zig's `Target` over forward.zig, so the lanes backend drives the real model.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const sg = @import("sessions_gpu.zig");
const batch = @import("batch.zig");
const samp = @import("sampling_gpu.zig");
const branches = @import("draft/branches.zig");
const pk = @import("serve_knobs.zig");
const ced = @import("ced.zig");
const pf = @import("forward_prefill.zig");

pub const Error = error{ TreeWindow, Sampling, Slots, NotAChain, WindowRows };

pub const GpuTarget = struct {
    f: *fwd.Forward,
    /// rows a prefill window takes (the decode path's buckets; the forward plans buffers for them)
    prefill_rows: u32 = 16,
    /// a slot that already holds the prompt: prefill returns this choice without running the prompt again
    resume_first: ?u32 = null,
    /// the most rows a window may take: the forward's buffers are planned for its buckets (run.zig has no bounds)
    max_rows: u32 = 16,
    /// the prompt through the forward's own prefill (2,048-row segments) instead of `prefill_rows`-row decode windows
    own_prefill: bool = false,
    /// with `own_prefill`: where the prompt's last token runs (`.verify`: first decode window; `.prefill`: last row)
    prompt_tail: pk.PromptTail = .prefill,
    /// the session store over the paged pool (null: every prompt from an empty slot, nothing kept)
    sessions: ?*sg.Sessions = null,
    /// several live slots (TF_DSV41_SLOTS > 1): windows in row mode over every slot, the prompt on the activated slot
    rows: ?*batch.Batch = null,
    /// the keyed sampler (model.zig makes it on every rank): sampled choices; null refuses them (error.Sampling)
    sampler: ?*samp.GpuSampler = null,
    /// draft trees (`parents`): resolved as chain windows on the slot (draft/branches.zig), the last one kept by path
    tree: branches.Resolver = .{},
    /// the DSpark pass the lanes drive: with `.verify`, the prefilled tail's taps go to it before the first window
    pass: ?iface.Pass = null,
    ingest_window: u32 = 0,
    /// each slot's next request's reply budget (`admit`, before its prefill): its pool admission's price
    max_new: [16]?u64 = @splat(null),
    /// the next `begin`'s admission as the round planner decided it (kv/sched.zig); null: priced at `begin`
    admission: ?Planned = null,
    /// releases wait for `settleReleases` (the rounds: after the next round's admissions are decided)
    defer_release: bool = false,
    /// TF_DSV41_ENGRAM_WARM=1 (`warm`): commit and draft-end Engram warms (every rank alike)
    engram_warm: bool = false,
    releasing: [16]bool = @splat(false),
    abandoning: [16]bool = @splat(false),
    /// TF_DSV41_PIECE_RUNS: rounds whose pieces ran as one multi-segment run
    piece_runs: u64 = 0,
    /// TF_DSV41_REPLAY_RUNS: slots whose decoder replay a batched run did (`finish` / `tail` skip it), and the runs
    replayed: [16]bool = @splat(false),
    replay_runs: u64 = 0,

    /// A request's admission (kv/sched.zig Admission + the round's spills, with its first admission).
    pub const Planned = struct { need: u32, hit: ?@import("kv").sessions.store.Hit, spills: []const u32 = &.{} };

    pub fn target(t: *GpuTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .release_checked = releaseChecked, .available = available, .admit = admit, .begin = begin, .piece = piece, .finish = finish, .tail = tail, .pieces = pieces, .replays = replays, .warm = warm };

    fn admit(p: *anyopaque, slot: u32, max_new: u64) void {
        const t = self_(p);
        if (slot < t.max_new.len) t.max_new[slot] = max_new;
    }

    fn self_(p: *anyopaque) *GpuTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self_(p);
        if (!available(p, slot)) return error.SlotUnavailable;
        if (t.rows) |b| try b.beginPrefill(slot) else if (slot != 0) return error.Slots;
        _ = t.takeReplayed(slot);
        const keyed = try t.keyedOf(sampling);
        t.tree.clear();
        if (t.resume_first) |first| if (t.f.slot.pos == ids.len and t.f.slot.pending == null) return first;
        // the request's pool pages, then a saved entry that prefixes the prompt: the slot continues, the rest prefills
        const max_new: ?u64 = if (slot < t.max_new.len) t.max_new[slot] else null;
        if (slot < t.max_new.len) t.max_new[slot] = null;
        // a failed prompt leaves the slot empty on every rank, its pages and reservation back
        errdefer if (t.sessions != null) t.abandon(slot);
        const at: usize = if (t.sessions) |ss| @intCast(try ss.resume_(ids, max_new)) else 0;
        const rest = ids[at..];
        if (t.own_prefill and t.prompt_tail == .verify) {
            // the rows before the last token prefill; the last token is the pending row of the first verify window
            const sp = pk.promptSplit(rest.len, .verify);
            pf.state(t.f).taps_rows = 0; // a one-token prompt prefills nothing: no stale taps handed over
            // sessions: the prompt's snapshot at its replay point: its rows prefilled, an entry, then the rest
            const save_at: ?u64 = if (t.sessions != null) ced.savePoint(ids.len, at) else null;
            if (save_at) |s_at| {
                const cut: usize = @intCast(s_at);
                try t.f.prefill(ids[at..cut]);
                // full mode: the piece's last rows to the drafter first, so the snapshot's drafter rings hold them
                if (pf.state(t.f).mode == .full) try t.handTaps(slot);
                _ = try t.sessions.?.savePrompt();
                try t.f.prefill(ids[cut .. at + sp.prefill]);
            } else try t.f.prefill(rest[0..sp.prefill]);
            // the decoder replay over the prompt's tail (full mode: nothing)
            try t.f.finishPrompt();
            try t.handTaps(slot);
            var one: [1]u32 = undefined;
            var out = [_][]u32{&one};
            try window(p, &.{.{ .slot = slot, .start = ids.len - sp.window, .tokens = ids[ids.len - sp.window ..], .parents = null, .draws = &.{draw}, .sampling = sampling }}, &out);
            return one[0];
        }
        if (t.own_prefill) {
            // the keyed choice after the prompt: the last segment's last row
            try t.f.prefill(rest);
            const seg: usize = @import("forward_prefill.zig").segmentRows(t.f);
            const last = if (rest.len % seg == 0) seg else rest.len % seg;
            if (keyed) |s| return t.pickOne(@intCast(last - 1), s, draw);
            const picks = try t.f.gpa.alloc(u32, last);
            defer t.f.gpa.free(picks);
            try t.f.greedy(picks);
            return picks[last - 1];
        }
        var at_ = at;
        var choice: [1]u32 = .{0};
        while (at_ < ids.len) {
            const n = @min(@min(t.prefill_rows, t.max_rows), ids.len - at_);
            try t.f.window(ids[at_ .. at_ + n]);
            if (at_ + n == ids.len) {
                // the keyed choice after the prompt's last row: the last row's greedy pick, or its sampled one
                if (keyed) |s| choice[0] = try t.pickOne(@intCast(n - 1), s, draw) else {
                    var rows: [64]u32 = undefined;
                    try t.f.greedy(rows[0..n]);
                    choice[0] = rows[n - 1];
                }
            }
            try t.f.keep(@intCast(n - 1));
            at_ += n;
        }
        return choice[0];
    }

    /// The pieces path's slot for an operation: activated (row mode), or the one slot.
    fn onSlot(t: *GpuTarget, slot: u32) !void {
        if (t.rows) |b| try b.beginPrefill(slot) else if (slot != 0) return error.Slots;
    }

    fn begin(p: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const t = self_(p);
        if (!available(p, slot)) return error.SlotUnavailable;
        if (!(t.own_prefill and t.prompt_tail == .verify)) return error.NoPieces;
        _ = t.takeReplayed(slot); // a batched replay of a request that never finished is not this one's
        const planned = t.admission;
        t.admission = null;
        try t.onSlot(slot);
        t.tree.clear();
        const max_new: ?u64 = if (slot < t.max_new.len) t.max_new[slot] else null;
        if (slot < t.max_new.len) t.max_new[slot] = null;
        errdefer if (t.sessions != null) t.abandon(slot);
        pf.state(t.f).taps_rows = 0;
        const ss = t.sessions orelse return .{};
        if (planned) |a| {
            const r = try ss.resumePlanned(ids, a.need, a.hit, a.spills);
            return .{ .at = r.at, .damaged = r.damaged };
        }
        const before = ss.stats.damaged;
        const at = try ss.resume_(ids, max_new);
        return .{ .at = at, .damaged = ss.stats.damaged != before };
    }

    fn piece(p: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void {
        const t = self_(p);
        try t.onSlot(slot);
        if (start == end and save) {
            // a leaving prompt's finished rows kept as they stand (the lane host: its client left mid-prompt)
            if (t.f.slot.pos != start or start < ced.session_min or end + 1 > ids.len) return error.NotAChain;
            if (t.sessions) |ss| _ = try ss.savePrompt();
            return;
        }
        if (t.f.slot.pos != start or start >= end or end + 1 > ids.len) return error.NotAChain;
        // the next round's piece of this prompt read ahead behind this one, up to the prompt's last token
        const last = ids.len - 1;
        if (end < last) try t.f.prefillAhead(ids[end..@min(last, end + (end - start))]);
        try t.f.prefill(ids[start..end]);
        const End = struct {
            target: *GpuTarget,
            slot: u32,
            pub fn snapshot(e: @This()) !void {
                if (e.target.sessions) |ss| _ = try ss.savePrompt();
            }
            pub fn handoff(e: @This()) !void {
                try e.target.handTaps(e.slot);
            }
        };
        try @import("piece_handoff.zig").finish(End{ .target = t, .slot = slot }, pf.state(t.f).mode == .full, save and end >= ced.session_min);
    }

    fn finish(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self_(p);
        try t.onSlot(slot);
        if (ids.len == 0 or t.f.slot.pos != ids.len - 1) return error.NotAChain;
        if (pf.state(t.f).mode == .replay and !t.takeReplayed(slot)) {
            // the decoder replay over the prompt's tail, its taps the drafter's context
            pf.state(t.f).taps_rows = 0;
            try t.f.finishPrompt();
            try t.handTaps(slot);
        }
        var one: [1]u32 = undefined;
        var out = [_][]u32{&one};
        try window(p, &.{.{ .slot = slot, .start = ids.len - 1, .tokens = ids[ids.len - 1 ..], .parents = null, .draws = &.{draw}, .sampling = sampling }}, &out);
        return one[0];
    }

    /// A round's pieces of several slots (TF_DSV41_PIECE_RUNS): their encoder passes in one run, then each snapshot.
    fn pieces(p: *anyopaque, list: []const iface.Target.Piece) anyerror!void {
        const t = self_(p);
        if (list.len > 1 and list.len <= 16 and pf.state(t.f).mode == .replay and t.f.pf_switch != null) {
            var segs: [16]pf.MultiSeg = undefined;
            for (list, segs[0..list.len]) |q, *seg| {
                if (q.start >= q.end or q.end + 1 > q.ids.len) return error.NotAChain;
                seg.* = .{ .slot = q.slot, .start = q.start, .ids = q.ids[q.start..q.end] };
            }
            if (try pf.runnable(t.f, segs[0..list.len])) {
                try t.onSlot(list[0].slot); // the slots' pending windows kept (every rank), as each `piece` does
                try t.f.prefillMulti(segs[0..list.len]);
                t.piece_runs += 1;
                if (t.piece_runs == 1 or t.piece_runs % 256 == 0) std.log.scoped(.dsv41).info("piece runs: {d} (this one {d} segments in one run)", .{ t.piece_runs, list.len });
                // the session save after the round's prefill, a piece at a time
                for (list) |q| if (q.save and q.end >= ced.session_min) if (t.sessions) |ss| {
                    try t.onSlot(q.slot);
                    _ = try ss.savePrompt();
                };
                return;
            }
        }
        for (list) |q| try piece(p, q.slot, q.ids, q.start, q.end, q.save);
    }

    /// Iface.Target.warm: row mode warms on every rank; one slot, this rank's reads. Off unless TF_DSV41_ENGRAM_WARM=1.
    fn warm(p: *anyopaque, asks: []const iface.Warm) void {
        const t = self_(p);
        if (!t.engram_warm or asks.len == 0) return;
        if (t.rows) |b| {
            b.warmEngram(asks) catch |e| if (!quiet) std.log.scoped(.dsv41).warn("engram warm: {t} (the windows read their own rows)", .{e});
            return;
        }
        for (asks) |a| if (a.slot == 0) t.f.prefetchEngram(a.start, a.ids);
    }

    /// TF_DSV41_ENGRAM_WARM: 0 (default) | 1.
    pub fn engramWarmEnv() !bool {
        const v = std.mem.trim(u8, std.mem.span(std.c.getenv("TF_DSV41_ENGRAM_WARM") orelse return false), " ");
        if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
        if (std.mem.eql(u8, v, "1")) return true;
        return error.BadEngramWarm;
    }

    /// `finish` up to the last token (TF_DSV41_TAIL_JOIN): the decoder replay and its taps, no window pending.
    fn tail(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 {
        _ = draw;
        const t = self_(p);
        try t.onSlot(slot);
        if (ids.len == 0 or t.f.slot.pos != ids.len - 1) return error.NotAChain;
        _ = try t.keyedOf(sampling); // refused here as `finish` would, not in the round
        if (pf.state(t.f).mode == .replay and !t.takeReplayed(slot)) {
            pf.state(t.f).taps_rows = 0;
            try t.f.finishPrompt();
            try t.handTaps(slot);
        }
        return null;
    }

    /// The releases held since the last round (`defer_release`) run now, slot by slot.
    pub fn settleReleases(t: *GpuTarget) void {
        for (&t.releasing, 0..) |*x, s| if (x.*) {
            t.completeRelease(@intCast(s)) catch |e| {
                if (!quiet) std.log.err("dsv41 deferred release {d}: {t}", .{ s, e });
            };
        };
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.targetWindow(segments, choices, t.sampler);
        if (segments.len != 1) return error.Slots;
        const s = segments[0];
        if (s.slot != 0) return error.Slots;
        if (s.parents != null) return t.tree.window(.{ .ptr = t, .vtable = &chains }, s, choices[0]);
        t.tree.clear();
        const keyed = try t.keyedOf(s.sampling);
        // a chain window never kept keeps every row: the lanes' undrafted step sends the next row without a keep
        if (t.f.slot.pending) |w| {
            if (s.start != w.start + w.n) return error.NotAChain;
            try t.f.keep(w.n - 1);
        }
        if (s.start != t.f.slot.pos) return error.NotAChain;
        if (s.tokens.len > t.max_rows) return error.WindowRows;
        try t.f.window(s.tokens);
        if (keyed) |smp| return t.sampler.?.choose(0, @intCast(s.tokens.len), smp, s.draws, choices[0][0..s.tokens.len]);
        try t.f.greedy(choices[0][0..s.tokens.len]);
    }

    /// The slot's pending window (a chain window never kept), if any.
    pub fn hasPending(t: *const GpuTarget, slot: u32) bool {
        if (t.rows) |b| return slot < b.ss.n and b.pending(slot) != null;
        return slot == 0 and t.f.slot.pending != null;
    }

    /// The slot's pending window dropped on every rank, nothing committed (calibration timing).
    pub fn dropWindow(t: *GpuTarget, slot: u32) !void {
        if (t.rows) |b| return b.drop(slot);
        if (slot != 0) return error.Slots;
        t.tree.clear();
        if (t.f.slot.pending != null) try t.f.drop();
    }

    /// The prompt's prefilled tail's taps into the DSpark pass: its last `ingest_window` rows, one ingest.
    fn handTaps(t: *GpuTarget, slot: u32) !void {
        const st = pf.state(t.f);
        return t.handTapsAt(slot, st.taps_at, st.taps_rows, 0);
    }

    /// The drafter's context from "w.taps": positions [at, at + rows) at rows [base, base + rows), the last window.
    fn handTapsAt(t: *GpuTarget, slot: u32, at: u64, n: u32, base: u32) !void {
        const ps = t.pass orelse return;
        const h = ced.handoff(at, n, t.ingest_window) orelse return;
        var rows: [256]u32 = undefined;
        if (h.n > rows.len) return error.WindowRows;
        for (rows[0..h.n], 0..) |*r, i| r.* = base + h.row0 + @as(u32, @intCast(i));
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        const tp: iface.Taps = .{ .address = t.f.runner.addressOf("w.taps") orelse return error.Unbound, .rows = base + n, .stride = k * t.f.cfg.hidden };
        try ps.ingest(slot, h.start, tp, rows[0..h.n]);
    }

    /// A batched run replayed `slot`'s tail (TF_DSV41_REPLAY_RUNS): true once, and the flag cleared.
    fn takeReplayed(t: *GpuTarget, slot: u32) bool {
        if (slot >= t.replayed.len or !t.replayed[slot]) return false;
        t.replayed[slot] = false;
        return true;
    }

    /// TF_DSV41_REPLAY_RUNS: the round's finished prompts' decoder replays in one run, before their `finish` / `tail`.
    fn replays(p: *anyopaque, list: []const iface.Target.Final) anyerror!void {
        const t = self_(p);
        if (list.len < 2 or pf.state(t.f).mode != .replay or t.f.pf_switch == null) return;
        var segs: [16]pf.MultiSeg = undefined;
        var n: usize = 0;
        for (list) |q| {
            if (q.ids.len < 2 or n == segs.len) continue;
            const end: u64 = q.ids.len - 1;
            const tl = ced.tailOf(end);
            if (tl.m <= 32) continue;
            segs[n] = .{ .slot = q.slot, .start = tl.r0, .ids = q.ids[tl.r0..end] };
            n += 1;
        }
        if (n < 2 or !try pf.replayRunnable(t.f, segs[0..n])) return;
        try t.onSlot(segs[0].slot); // the slots' pending windows kept (every rank), as each `finish` does
        var slots: [16]u32 = undefined;
        for (segs[0..n], slots[0..n]) |seg, *x| x.* = seg.slot;
        var out: [16]pf.Replayed = undefined;
        try t.f.replayMulti(slots[0..n], &out);
        for (out[0..n]) |o| {
            t.replayed[o.slot] = true;
            try t.handTapsAt(o.slot, o.at, o.rows, o.row0);
        }
        t.replay_runs += 1;
        if (t.replay_runs == 1 or t.replay_runs % 256 == 0) std.log.scoped(.dsv41).info("replay runs: {d} (this one {d} slots' tails in one run)", .{ t.replay_runs, n });
    }

    /// T > 0: the sampling the sampler keys (null: greedy); refused without a sampler.
    fn keyedOf(t: *const GpuTarget, sampling: ?lanes.Sampling) !?lanes.Sampling {
        if (!samp.sampled(sampling)) return null;
        if (t.sampler == null) return error.Sampling;
        return sampling;
    }

    /// The keyed choice of "w.logits" row `row`, drawn at `draw`.
    fn pickOne(t: *GpuTarget, row: u32, s: lanes.Sampling, draw: u64) !u32 {
        var one: [1]u32 = undefined;
        try t.sampler.?.choose(row, 1, s, &.{draw}, &one);
        return one[0];
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self_(p);
        if (t.rows) |b| return b.keepPath(slot, path);
        if (slot != 0) return error.Slots;
        if (t.tree.live) return t.f.keep(try t.tree.keep(path));
        // a chain keeps rows 0..k: the forward commits row 0 + k accepted drafts
        for (path, 0..) |row, i| if (row != i) return error.NotAChain;
        if (path.len == 0) return error.NotAChain;
        try t.f.keep(@intCast(path.len - 1));
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        const t = self_(p);
        if (t.rows) |b| if (b.tapsOf(slot)) |x| return .{ .address = x.address, .rows = x.rows, .stride = @intCast(t.f.cfg.dspark_targets.items().len * t.f.cfg.hidden) };
        const n: u32 = if (t.f.slot.pending) |w| w.n else @intCast(t.f.ids.len);
        const D: u32 = t.f.cfg.hidden;
        const k: u32 = @intCast(t.f.cfg.dspark_targets.items().len);
        return .{ .address = t.f.runner.addressOf("w.taps") orelse 0, .rows = n, .stride = k * D, .map = t.tree.map() };
    }

    /// The tree resolver's chains: a chain window through `window` (its rules and refusals), dropped on every rank.
    const chains: branches.Chains.VTable = .{ .run = chainRun, .drop = chainDrop };

    fn chainRun(p: *anyopaque, slot: u32, start: u64, tokens: []const u32, draws: []const u64, sampling: ?lanes.Sampling, out: []u32) anyerror!void {
        var one = [_][]u32{out};
        return window(p, &.{.{ .slot = slot, .start = start, .tokens = tokens, .parents = null, .draws = draws, .sampling = sampling }}, &one);
    }

    fn chainDrop(p: *anyopaque, slot: u32) anyerror!void {
        if (slot != 0) return error.Slots;
        return self_(p).f.drop();
    }

    fn empty(t: *GpuTarget, slot: u32) !void {
        t.tree.clear();
        if (t.rows) |b| return b.release(slot);
        if (slot != 0) return error.Slots;
        try t.f.release();
    }
    fn abandon(t: *GpuTarget, slot: u32) void {
        if (slot >= t.releasing.len) return;
        t.releasing[slot] = true;
        t.abandoning[slot] = true;
        t.completeRelease(slot) catch |e| {
            if (!quiet) std.log.err("dsv41 abandon {d}: {t}", .{ slot, e });
        };
    }
    fn available(p: *anyopaque, slot: u32) bool {
        const t = self_(p);
        return slot < t.releasing.len and !t.releasing[slot];
    }
    fn completeRelease(t: *GpuTarget, slot: u32) !void {
        if (t.abandoning[slot]) try t.empty(slot) else try t.releaseNow(slot);
        t.releasing[slot] = false;
        t.abandoning[slot] = false;
    }
    fn releaseChecked(p: *anyopaque, slot: u32) !void {
        const t = self_(p);
        if (slot >= t.releasing.len) return error.Slots;
        t.releasing[slot] = true;
        if (!t.defer_release) try t.completeRelease(slot);
    }
    fn release(p: *anyopaque, slot: u32) void {
        releaseChecked(p, slot) catch |e| {
            if (!quiet) std.log.err("dsv41 release {d}: {t}", .{ slot, e });
        };
    }
    fn releaseNow(t: *GpuTarget, slot: u32) !void {
        if (t.rows) |b| {
            if (t.sessions) |ss| if (!(t.own_prefill and t.prompt_tail == .verify)) {
                b.saveTurn(slot, ss) catch {
                    try b.release(slot);
                };
                return;
            };
            return b.release(slot);
        }
        if (slot != 0) return error.Slots;
        t.tree.clear();
        if (t.sessions) |ss| {
            if (t.own_prefill and t.prompt_tail == .verify) return t.f.release();
            if (t.f.slot.pending) |w| try t.f.keep(w.n - 1);
            _ = ss.save() catch {
                try t.f.release();
                return;
            };
            return;
        }
        try t.f.release();
    }
};
