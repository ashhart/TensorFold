//! GLM-5.3 behind the lane core on a pair: a stream a slot, every operation sent to rank 1 before rank 0 runs it.
const std = @import("std");
const lanes = @import("lanes");
const eng = @import("engine.zig");
const fl = @import("follow.zig");
const snap = @import("snap.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const gpu = @import("gpu.zig");
const bt = @import("backend_timing.zig");
const R = gpu.max_rows;

const RING = 1024;
const P_BINS = bt.P_BINS;
const ProbStats = bt.ProbStats;
const SlotCal = bt.SlotCal;

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *eng.Engine,
    by: std.AutoHashMapUnmanaged(*Stream, u32) = .empty,
    /// Drawn tokens by handle (every read waits for its forward anyway: a pair runs no rounds ahead).
    ring: [RING]u32 = undefined,
    puts: u64 = 0,
    first: []u32,
    pending_keep: []u32,
    pending_path: [][8]u32,
    last_draft_depth: []u32,
    cal: []SlotCal,
    fused_wd: bool = true,
    use_draft_probs: bool = true,
    use_tree: bool = true,
    prob_stats: bool = false,
    pstats: ProbStats = .{},
    wins: std.ArrayList(eng.Win) = .empty,
    reqs: std.ArrayList(eng.Draft) = .empty,
    toks: std.ArrayList(u32) = .empty,
    /// Window ms by rows and one chained head step, timed at load (both Macs in step).
    costs: [R]lanes.config.Cost = undefined,
    head_ms: f64 = 0,
    timed: bool = false,
    next_snap: u32 = 1,

    pub fn init(gpa: std.mem.Allocator, e: *eng.Engine) !Backend {
        const n = e.slots.len;
        const first = try gpa.alloc(u32, n);
        errdefer gpa.free(first);
        const pk = try gpa.alloc(u32, n);
        errdefer gpa.free(pk);
        @memset(pk, 0);
        const pp = try gpa.alloc([8]u32, n);
        errdefer gpa.free(pp);
        @memset(pp, .{ 0, 1, 2, 3, 4, 5, 6, 7 });
        const ldd = try gpa.alloc(u32, n);
        errdefer gpa.free(ldd);
        @memset(ldd, 0);
        const cal = try gpa.alloc(SlotCal, n);
        @memset(cal, .{});
        const use_probs = @import("sys").environment.read(bool, "TF_GLM_DRAFT_PROBS", true) catch true;
        const pstats = @import("sys").environment.read(bool, "TF_GLM_PROF", false) catch false;
        return .{ .gpa = gpa, .e = e, .first = first, .pending_keep = pk, .pending_path = pp, .last_draft_depth = ldd, .cal = cal, .fused_wd = true, .use_draft_probs = use_probs, .use_tree = e.use_tree, .prob_stats = pstats };
    }

    pub fn deinit(b: *Backend) void {
        b.by.deinit(b.gpa);
        b.wins.deinit(b.gpa);
        b.reqs.deinit(b.gpa);
        b.toks.deinit(b.gpa);
        b.gpa.free(b.first);
        b.gpa.free(b.pending_keep);
        b.gpa.free(b.pending_path);
        b.gpa.free(b.last_draft_depth);
        b.gpa.free(b.cal);
    }

    fn flushKeeps(b: *Backend) !void {
        for (b.pending_keep, 0..) |*k, slot| if (k.* > 0) {
            const kept = k.*;
            k.* = 0;
            const path = b.pending_path[slot][0..kept];
            if (eng.tr.needsCompact(path)) try fl.keepPath(b.e, @intCast(slot), path) else try fl.keep(b.e, @intCast(slot), kept);
        };
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = firstFn, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .probabilities = probabilitiesFn, .tree = treeFn, .release = release, .reports_fill = true } };
    }

    /// Windows of up to 16 rows (each row's bits whatever shares its forward), MTP chains, streams in one forward.
    pub fn facts(b: *const Backend) lanes.Model {
        const df2 = b.e.hasDFlash2();
        return .{
            .exact_width = 8,
            .first_copy_rows = 8,
            .root_copies = b.e.hasDraft(),
            .copy_match = 4,
            .own_copy_rates = true,
            .copy_guard = true,
            .mtp = b.e.hasDraft(),
            .speculate = b.e.hasDraft(),
            .speculate_early = false,
            .gpu_tokens = false,
            .plain_guard = true,
            .fixed_draft_block = df2 and !b.e.hasMtp(),
            .draft_prior = if (df2) &.{ 0.88, 0.84, 0.80, 0.76, 0.72, 0.68, 0.64 } else &.{ 0.80, 0.72, 0.65, 0.58, 0.52, 0.46, 0.40 },
            .drafts = eng.max_depth,
            .draft_probabilities = b.e.hasDraft() and b.use_draft_probs,
            .lone_cost_tile = 0,
            .window_costs = if (b.timed) &b.costs else &.{},
            .mtp_step_ms = b.head_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = R,
            .max_streams = @intCast(b.e.slots.len),
            .draft_streams = true,
        };
    }

    fn probabilitiesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const b = self(ptr);
        if (!b.use_draft_probs) return false;
        const slot = try b.slotOf(s);
        const st = &b.e.slots[slot];
        const is_df2 = if (b.e.dflash2) |d| !d.slots[slot].use_mtp else false;
        const cal = &b.cal[slot];
        const sc0: f64 = if (cal.was_mtp == !is_df2 and !is_df2) std.math.clamp((cal.hit0 + 8.0) / (cal.pred0 + 8.0), 0.82, 1.02) else 1.0;
        const scale: f64 = if (cal.was_mtp == !is_df2) std.math.clamp((cal.hit + 4.5) / (cal.pred + 4.5), 0.58, 1.02) else 1.0;
        const long_c = st.len >= 2048;
        if (b.use_tree and st.tree.use_tree and out.len <= st.tree.n) {
            for (0..out.len) |j| out[j] = if (long_c and j > 0) st.tree.probs[j] * 0.92 else if (!is_df2) st.tree.probs[j] else if (st.tree.parents[j] >= 0) st.tree.probs[j] * scale else st.tree.probs[j];
            return true;
        }
        if (!st.held_probs_valid or out.len > st.held_n) return false;
        var reach: f64 = 1.0;
        for (0..out.len) |j| {
            const p: f64 = std.math.clamp(@as(f64, st.held_probs[j]), 0.0, 1.0);
            const p_base: f64 = if (p <= 1e-6) 0.0 else if (is_df2) p else if (j == 0) eng.tr.calD0(p) else eng.tr.calNext(p, long_c);
            const p_cal: f64 = std.math.clamp(p_base * (if (j == 0) sc0 else scale), 0.0, 0.99);
            reach *= p_cal;
            out[j] = reach;
        }
        return true;
    }

    fn treeFn(ptr: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        if (!b.use_tree) return null;
        const slot = try b.slotOf(s);
        const st = &b.e.slots[slot];
        if (!st.tree.use_tree or st.tree.n == 0) return null;
        const n = st.tree.n;
        const toks = try gpa.dupe(u32, st.tree.tokens[0..n]);
        errdefer gpa.free(toks);
        const pars = try gpa.dupe(i32, st.tree.parents[0..n]);
        return .{ .count = n, .tokens = toks, .parents = pars };
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn slotOf(b: *Backend, s: *Stream) !u32 {
        return b.by.get(s) orelse error.UnknownStream;
    }

    fn put(b: *Backend, v: u32) u64 {
        const h = b.puts;
        b.ring[h % RING] = v;
        b.puts += 1;
        return h;
    }

    fn free(b: *const Backend) ?u32 {
        for (b.e.slots, 0..) |s, i| if (!s.used) return @intCast(i);
        return null;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const e = b.e;
        if (e.rank() != 0) return error.FollowsPeer;
        if (s.sampling) |p| if (p.temperature > 0) return error.GreedyOnly;
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + s.max_new + eng.max_depth + R + 1 > e.context) return error.ContextFull;
        const i = b.free() orelse return error.NoFreeSlot;
        b.pending_keep[i] = 0;
        b.last_draft_depth[i] = 0;
        b.cal[i] = .{};
        try b.by.put(b.gpa, s, i);
        try fl.begin(e, i, prompt);
        s.cached = 0;
        var at: u32 = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const k: *snap.Snap = @ptrCast(@alignCast(saved));
            if (k.at < prompt.len) {
                try fl.restore(e, i, k.id);
                at = k.at;
                s.cached = at;
            } else s.reuse_failed = true;
        }
        var next: usize = 0;
        var mark: usize = 0;
        const use_cuts = s.reuse.marks.len > 0 or e.slots.len > 1;
        const max_chunk: u32 = if (e.pair.pp > 1) blk: {
            const rem: u32 = @intCast(prompt.len - at);
            const mb: u32 = if (rem >= 1024) e.pair.pp * 2 else e.pair.pp;
            break :blk @min(@as(u32, 768), @max(@as(u32, 256), (rem + mb - 1) / mb));
        } else gpu.prompt_rows;
        while (at < prompt.len) {
            if (s.isCancelled()) return error.Cancelled;
            while (next < s.chunks.len and s.chunks[next] <= at) next += 1;
            while (mark < s.reuse.marks.len and s.reuse.marks[mark] <= at) mark += 1;
            var end: u32 = if (use_cuts and next < s.chunks.len) s.chunks[next] else @intCast(prompt.len);
            if (mark < s.reuse.marks.len) end = @min(end, s.reuse.marks[mark]); // a mark ends a chunk: its state is kept there
            const n = @min(end - at, max_chunk);
            if (try fl.chunk(e, i, at, n, false)) |t| b.first[i] = t;
            lanes.prefill_widths.completed(b.gpa, s, n);
            at += n;
            s.fill_at.store(at, .monotonic); // fl.chunk waits on rank 0: `at` is landed progress
            if (std.mem.indexOfScalar(u32, s.reuse.marks, at) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, at);
        }
    }

    /// The prompt cache's functions: a stream's state at a mark, kept on both Macs by id.
    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return snap.bytes(self(ptr).e, at);
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const b = self(ptr);
        try b.flushKeeps();
        const i = try b.slotOf(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        const id = b.next_snap;
        b.next_snap +%= 1;
        return try fl.save(b.e, id, i, at);
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        const k: *snap.Snap = @ptrCast(@alignCast(saved));
        fl.drop(self(ptr).e, k.id);
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        if (position != b.e.cacheLen(i)) return error.PositionMismatch;
        return b.put(b.first[i]);
    }

    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unsupported; // a pair's ranks run windows in step
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = self(ptr);
        if (handle >= b.puts or b.puts - handle > RING) return error.StaleHandle;
        return b.ring[handle % RING];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        try b.flushKeeps();
        const e = b.e;
        b.wins.clearRetainingCapacity();
        b.toks.clearRetainingCapacity();
        var total: usize = 0;
        for (windows) |w| {
            if (w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
            const i = try b.slotOf(w.stream);
            const held = e.held(i);
            if (w.held > held.len) return error.WindowOutOfStep;
            if (w.parents) |ps| {
                if (windows.len != 1 or w.rows() > 8 or ps.len != w.rows()) return error.TreeWindowsUnsupported;
                var ds: [8]u32 = undefined;
                _ = try eng.tr.packAnc(e.cacheLen(i), ps, &ds);
                for (w.positions, 0..) |p, r| if (p != e.cacheLen(i) + 1 + ds[r]) return error.PositionMismatch;
            } else {
                for (w.positions, 0..) |p, r| if (p != e.cacheLen(i) + r + 1) return error.PositionMismatch;
            }
            total += w.rows();
        }
        if (total > R) return error.WindowOutOfStep;
        try b.toks.ensureTotalCapacity(b.gpa, total);
        for (windows) |w| {
            const i = b.by.get(w.stream).?;
            const at = b.toks.items.len;
            b.toks.appendAssumeCapacity(w.pending);
            b.toks.appendSliceAssumeCapacity(e.held(i)[0..w.held]);
            b.toks.appendSliceAssumeCapacity(w.tokens);
            try b.wins.append(b.gpa, .{ .slot = i, .tokens = b.toks.items[at..] });
        }
        const is_tree = windows.len == 1 and if (windows[0].parents) |ps| !lanes.accept.isChain(ps) else false;
        const from_tree = windows.len == 1 and e.slots[b.wins.items[0].slot].tree.use_tree;
        var prev_probs: [eng.max_depth]f32 = @splat(0);
        var prev_top4_ids: [4]u32 = @splat(0);
        var prev_top4_p: [4]f32 = @splat(0);
        var had_probs = false;
        var was_mtp = false;
        var p_slot: u32 = 0;
        if ((b.use_draft_probs or b.prob_stats) and windows.len == 1 and (windows[0].tokens.len == 0 or from_tree)) {
            p_slot = b.wins.items[0].slot;
            const st0 = &e.slots[p_slot];
            prev_probs = st0.held_probs;
            prev_top4_ids = st0.held_top4_ids;
            prev_top4_p = st0.held_top4_p;
            had_probs = st0.held_probs_valid;
            was_mtp = if (e.dflash2) |d| d.slots[p_slot].use_mtp else true;
        }
        var picks: [R]u32 = undefined;
        var pred_depth: u32 = 0;
        const can_fuse = b.fused_wd and e.hasDraft() and e.m.c.pp == 1 and windows.len == 1 and
            windows[0].stream.drafts and windows[0].stream.sampling == null and windows[0].stream.grammar == null and
            b.wins.items[0].tokens.len <= 8 and blk: {
            const w0_slot = b.wins.items[0].slot;
            const w0_len: u32 = @intCast(b.wins.items[0].tokens.len);
            if (e.dflash2) |d| if (!d.slots[w0_slot].use_mtp) {
                if (d.slots[w0_slot].end != e.cacheLen(w0_slot)) break :blk false;
                if (e.cacheLen(w0_slot) + w0_len + eng.max_depth + 2 > e.context) break :blk false;
                pred_depth = eng.max_depth;
                break :blk true;
            };
            const room = e.context -| (e.cacheLen(w0_slot) + w0_len + 3);
            if (!is_tree and windows[0].tokens.len > 0 and !from_tree) {
                pred_depth = 0;
                break :blk e.cacheLen(w0_slot) + w0_len + 1 <= e.context;
            }
            const cal0 = &b.cal[w0_slot];
            const sc1 = if (cal0.was_mtp) std.math.clamp((cal0.hit + 4.5) / (cal0.pred + 4.5), 0.58, 1.02) else 1.0;
            e.slots[w0_slot].cal0_scale = if (cal0.was_mtp) std.math.clamp((cal0.hit0 + 8.0) / (cal0.pred0 + 8.0), 0.82, 1.02) else 1.0;
            e.slots[w0_slot].cal_scale = sc1;
            const long_c = e.cacheLen(w0_slot) >= 2048;
            const cap_d: u32 = if (!long_c and cal0.was_mtp and cal0.pred >= 2.5 and sc1 < 0.82) 2 else if (long_c or (cal0.was_mtp and sc1 < 0.94)) 4 else 5;
            pred_depth = @min(cap_d, room);
            break :blk pred_depth > 0;
        };
        if (is_tree) {
            if (b.prob_stats) b.pstats.tree_rounds += 1;
            if (can_fuse) try fl.windowTreeDraft(e, b.wins.items[0].slot, b.wins.items[0].tokens, windows[0].parents.?, pred_depth, &picks) else try fl.windowTree(e, b.wins.items[0].slot, b.wins.items[0].tokens, windows[0].parents.?, &picks);
        } else if (can_fuse) {
            try fl.windowDraft(e, b.wins.items[0].slot, b.wins.items[0].tokens, pred_depth, &picks);
        } else {
            try fl.window(e, b.wins.items, &picks);
        }
        if (had_probs) {
            const h = @min(windows[0].rows() - 1, eng.max_depth);
            const wt = b.wins.items[0].tokens;
            if (b.prob_stats) {
                b.pstats.hist[h] += 1;
                if (was_mtp) {
                    b.pstats.top4_rounds += 1;
                    for (0..4) |k| if (picks[0] == prev_top4_ids[k]) {
                        b.pstats.top4_hits[k] += 1;
                        break;
                    };
                    const p2v: f64 = std.math.clamp(@as(f64, prev_top4_p[1]), 0.0, 0.999999);
                    const p2b: usize = @min(P_BINS - 1, @as(usize, @intFromFloat(p2v * @as(f64, P_BINS))));
                    b.pstats.p2_bins_o[p2b] += 1;
                    if (picks[0] == prev_top4_ids[1]) b.pstats.p2_bins_h[p2b] += 1;
                }
            }
            const cal = &b.cal[p_slot];
            const long_cal = e.cacheLen(p_slot) >= 2048;
            if (cal.was_mtp != was_mtp) cal.* = .{ .was_mtp = was_mtp };
            if (!is_tree) {
                for (0..h) |j| {
                    const pv: f64 = std.math.clamp(@as(f64, prev_probs[j]), 0.0, 0.999999);
                    const p_base: f64 = if (pv <= 1e-6) 0.0 else if (!was_mtp) pv else if (j == 0) eng.tr.calD0(pv) else eng.tr.calNext(pv, long_cal);
                    const hit = picks[j] == wt[j + 1];
                    if (j == 0 and was_mtp) {
                        cal.pred0 += p_base;
                        if (hit) cal.hit0 += 1.0;
                    } else if (j >= 1) {
                        cal.pred += p_base;
                        if (hit) cal.hit += 1.0;
                    }
                    if (b.prob_stats) {
                        const bin: usize = @min(P_BINS - 1, @as(usize, @intFromFloat(pv * @as(f64, P_BINS))));
                        b.pstats.offered[j][bin] += 1;
                        b.pstats.sum_p[j][bin] += pv;
                        if (hit) b.pstats.landed[j][bin] += 1;
                    }
                    if (!hit) break;
                }
            } else if (was_mtp and windows[0].parents != null and wt.len >= 2) {
                const hit0 = picks[0] == wt[1];
                cal.pred0 += eng.tr.calD0(std.math.clamp(@as(f64, prev_probs[0]), 0.0, 0.999999));
                if (hit0) {
                    cal.hit0 += 1.0;
                    const ps = windows[0].parents.?;
                    var cur: usize = 1;
                    var dep: usize = 1;
                    while (dep < eng.max_depth) : (dep += 1) {
                        const pv: f64 = std.math.clamp(@as(f64, prev_probs[dep]), 0.0, 0.999999);
                        var child: ?usize = null;
                        for (cur + 1..wt.len) |r| if (ps[r] == @as(i32, @intCast(cur))) {
                            child = r;
                            break;
                        };
                        const c = child orelse break;
                        cal.pred += eng.tr.calNext(pv, long_cal);
                        if (picks[cur] == wt[c]) {
                            cal.hit += 1.0;
                            cur = c;
                        } else break;
                    }
                }
            }
        }
        var row: usize = 0;
        for (windows, out, b.wins.items) |w, o, win| {
            @memcpy(o.sampled, picks[row..][0..w.rows()]);
            @memcpy(o.drafts, win.tokens[1..]);
            row += w.rows();
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            const slot = try b.slotOf(w.stream);
            const s = &b.e.slots[slot];
            const non_contig = eng.tr.needsCompact(path);
            if (non_contig and !eng.tr.validPath(path, R)) return error.TreeWindowsUnsupported;
            if (b.prob_stats and non_contig) {
                b.pstats.tree_branch_wins += 1;
                b.pstats.tree_branch_toks += path.len - 1;
            }
            if (s.fused_valid) {
                if (path.len > s.kept or !std.mem.eql(u32, path, s.fused_path[0..path.len])) return error.TreeWindowsUnsupported;
                if (path.len != s.kept) try fl.keep(b.e, slot, @intCast(path.len));
                continue;
            }
            b.pending_keep[slot] = @intCast(path.len);
            if (path.len <= 8) @memcpy(b.pending_path[slot][0..path.len], path);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (requests.len > R) return error.DraftOutOfStep;
        if (requests.len == 1 and requests[0].rows != null and requests[0].first == null and !requests[0].early and requests[0].lanes == null and !requests[0].ranks) {
            const r = requests[0];
            const slot = try b.slotOf(r.stream);
            const s = &b.e.slots[slot];
            const is_mtp = if (b.e.dflash2) |d| d.slots[slot].use_mtp else true;
            const want = @min(r.depth, if (is_mtp) @as(u32, 6) else eng.max_depth);
            const pad_to = @min(r.depth, eng.max_depth);
            if (s.fused_valid and b.pending_keep[slot] == 0 and r.follow.len == s.kept and
                std.mem.eql(u32, r.rows.?, s.fused_path[0..s.kept]) and std.mem.eql(u32, r.follow, s.fused_picks[0..s.kept]))
            {
                const no_ext = (s.fused_depth == 2 and s.cal_scale < 0.82) or
                    (s.fused_depth > 0 and s.held_probs_valid and !eng.tr.shouldExtend(s.held_probs[0..s.fused_depth], s.cal0_scale, s.cal_scale, s.len >= 2048));
                if (want <= s.fused_depth or no_ext) {
                    const base_n = @min(want, s.fused_depth);
                    if (base_n > 0 and base_n < pad_to) {
                        for (base_n..pad_to) |j| {
                            s.held[j] = s.held[base_n - 1];
                            s.held_probs[j] = 0;
                        }
                    }
                    s.held_n = pad_to;
                    s.fused_valid = false;
                    b.last_draft_depth[slot] = want;
                    return;
                }
                if (want <= eng.max_depth and b.e.cacheLen(slot) + want + 1 <= b.e.context) {
                    try fl.extendDraft(b.e, slot, want);
                    if (s.held_n > 0 and s.held_n < pad_to) {
                        for (s.held_n..pad_to) |j| {
                            s.held[j] = s.held[s.held_n - 1];
                            s.held_probs[j] = 0;
                        }
                        s.held_n = pad_to;
                    }
                    b.last_draft_depth[slot] = want;
                    return;
                }
            }
        }
        var ones: [R]u32 = undefined;
        b.reqs.clearRetainingCapacity();
        var pad_slot: ?u32 = null;
        var pad_want: u32 = 0;
        for (requests, 0..) |r, k| {
            if (r.early or r.lanes != null or r.ranks) return error.TreeDraftsUnsupported;
            const path: []const u32 = r.rows orelse &.{};
            if (path.len > 0 and eng.tr.needsCompact(path) and !eng.tr.validPath(path, R)) return error.TreeDraftsUnsupported;
            const follow: []const u32 = if (r.first) |f| blk: {
                ones[k] = switch (f) {
                    .handle => |h| try read(b, h),
                    .value => |v| v,
                };
                break :blk ones[k .. k + 1];
            } else r.follow;
            const slot = try b.slotOf(r.stream);
            const is_mtp = if (b.e.dflash2) |d| d.slots[slot].use_mtp else true;
            const depth = @min(r.depth, if (is_mtp) @as(u32, 6) else eng.max_depth);
            const pad_to = @min(r.depth, eng.max_depth);
            const run_d = if (is_mtp and b.use_tree and requests.len == 1 and r.rows != null and follow.len <= 2) @min(depth, @as(u32, 4)) else depth;
            if (run_d < pad_to) {
                pad_slot = slot;
                pad_want = pad_to;
            }
            b.last_draft_depth[slot] = depth;
            if (b.pending_keep[slot] != 0) {
                if (r.rows == null or b.pending_keep[slot] != follow.len) {
                    const kept = b.pending_keep[slot];
                    b.pending_keep[slot] = 0;
                    const pp = b.pending_path[slot][0..kept];
                    if (eng.tr.needsCompact(pp)) try fl.keepPath(b.e, slot, pp) else try fl.keep(b.e, slot, kept);
                } else {
                    b.pending_keep[slot] = 0;
                }
            }
            try b.reqs.append(b.gpa, .{ .slot = slot, .prompt = r.rows == null, .follow = follow, .depth = run_d, .path = path });
        }
        try fl.draft(b.e, b.reqs.items);
        if (pad_slot) |ps| {
            const st = &b.e.slots[ps];
            if (st.held_n > 0 and st.held_n < pad_want) {
                for (st.held_n..pad_want) |j| {
                    st.held[j] = st.held[st.held_n - 1];
                    st.held_probs[j] = 0;
                }
                st.held_n = pad_want;
            }
        }
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const b = self(ptr);
        const kv = b.by.fetchRemove(s) orelse return;
        if (b.prob_stats) b.pstats.logAndReset();
        b.pending_keep[kv.value] = 0;
        b.last_draft_depth[kv.value] = 0;
        fl.release(b.e, kv.value);
    }

    /// Rank 0 at load: windows of 1-16 distinct-token rows and head chains timed on a scratch stream.
    pub fn time(b: *Backend, io: std.Io) !void {
        b.head_ms = try bt.time(b.e, io, &b.costs);
        b.timed = true;
    }
};
