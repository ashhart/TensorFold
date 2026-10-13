//! The 27B's CUDA engine behind the lane core: every stream's window in one verify, DFlash2 drafts in one batch.

const std = @import("std");
const lanes = @import("lanes");
const c = @import("shape.zig");
const st = @import("state.zig");
const Engine = @import("engine.zig").Engine;
const Part = @import("forward.zig").Part;
const Forward = @import("forward.zig").Forward;
const eng = @import("engine.zig");
const prompt_rows = eng.prompt_rows;
const reuse = @import("reuse.zig");
const exact = @import("exact_prompt_cache.zig");
const exact_reuse = @import("exact_seq_reuse.zig");
const draft_reuse = @import("exact_draft_reuse.zig");
const dl = @import("draft_load.zig");
const dr = @import("draft.zig");
const drafts = @import("drafts.zig");
const lone = @import("lone.zig");
const trees = @import("tree.zig");
const vision = @import("vision.zig");
const be = lanes.backend;

/// The row counts calibrate times (calibration_rows without the tile steps a GB10 skips), to 16 a stream.
const calibration = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256 };

/// The last verified round, committed when the next call needs the streams' states (keep may cut it first).
const Pending = struct {
    streams: [st.max_streams]*lanes.Stream = undefined,
    parts: [st.max_streams]Part = undefined,
    paths: [st.max_streams][]const u32 = undefined,
    path_rows: [st.round_rows]u32 = undefined, // the round's kept rows, each stream's path in its own slice
    tokens: [st.round_rows]u32 = undefined,
    parents: [st.round_rows]i32 = undefined,
    starts: [st.max_streams]usize = undefined, // each window's first row in the round
    n: usize = 0,
};

pub const Cuda = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    streams: u32,
    seqs: std.AutoHashMapUnmanaged(*lanes.Stream, *st.Seq) = .empty,
    logits: []u16 = &.{}, // a whole row on the host, for a draw with top_k off
    cand_values: []f32 = &.{}, // the rows' top candidates on the host
    cand_ids: []i64 = &.{},
    values: []f64 = &.{},
    pending: Pending = .{},
    drafter: ?*dr.DFlash2 = null, // DFlash2, when its checkpoint is pulled and drafts are on
    tower: ?*vision.Tower = null, // the image tower, with --vision
    drafting: std.AutoHashMapUnmanaged(*lanes.Stream, *drafts.Stream) = .empty,
    costs: [calibration.len + 1]lanes.config.Cost = undefined, // the forward's ms by a round's rows (calibrate)
    cost_count: usize = 0,
    exact_cache: exact.Cache,
    seq_pool: exact_reuse.Storage,
    restored: exact_reuse.Proofs = .{},
    draft_pool: draft_reuse.Storage,
    draft_initialized: draft_reuse.policy.Proofs = .{},
    reset_skip: exact_reuse.reset_skip.Stats,

    pub const openPrefill = exact_reuse.openPrefill;
    pub const lookupPrefill = exact_reuse.lookupPrefill;

    pub fn init(gpa: std.mem.Allocator, e: *Engine, streams: u32) !Cuda {
        return @import("exact_init.zig").init(gpa, e, streams);
    }

    pub fn deinit(x: *Cuda) void {
        draft_reuse.shutdown(x) catch |err| {
            std.log.err("fatal unresolved draft Context fence; terminating before device frees: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        exact_reuse.shutdown(x);
        x.exact_cache.deinit(x.e.ops()) catch |err| std.log.err("exact cache shutdown: {s}", .{@errorName(err)});
        var it = x.seqs.valueIterator();
        while (it.next()) |seq| {
            seq.*.deinit();
            x.gpa.destroy(seq.*);
        }
        x.seqs.deinit(x.gpa);
        var dit = x.drafting.valueIterator();
        while (dit.next()) |ds| {
            ds.*.deinit();
            x.gpa.destroy(ds.*);
        }
        x.drafting.deinit(x.gpa);
        x.gpa.free(x.logits);
        x.gpa.free(x.cand_values);
        x.gpa.free(x.cand_ids);
        x.gpa.free(x.values);
    }

    /// The facts the round loop reads: 16-row windows in one forward; with DFlash2, host trees with node chances.
    pub fn facts(x: *const Cuda) lanes.Model {
        var m: lanes.Model = .{ .exact_width = st.max_rows, .hidden_rows = true, .batch_rows = st.round_rows, .max_streams = @min(x.streams, st.max_streams), .streams_exact = true };
        if (x.drafter != null) {
            m.mtp = true;
            m.speculate = true;
            m.speculate_early = false;
            m.drafts = drafts.max_nodes;
            m.draft_probabilities = true;
            m.draft_streams = true;
        }
        const timed = x.costs[0..x.cost_count];
        var narrow: usize = 0;
        while (narrow < timed.len and timed[narrow].width <= st.max_rows) narrow += 1;
        m.window_costs = timed[0..narrow];
        m.shared_costs = timed;
        return m;
    }

    /// multi.calibrate: the forward's ms (median of three) by the rows `streams` windows bring, on a fresh state.
    pub fn calibrate(x: *Cuda, io: std.Io, streams: usize) !void {
        var seq = x.e.sequence(st.max_rows + 2) catch |err| return if (err == error.OutOfMemory) err else error.OutOfDeviceMemory;
        defer seq.deinit();
        const zeros: [st.max_rows]u32 = @splat(0);
        const most: usize = st.max_rows * @as(usize, @min(streams, st.max_streams));
        var rows: [calibration.len + 1]usize = undefined;
        var n: usize = 0;
        for (calibration) |r| if (r < most) {
            rows[n] = r;
            n += 1;
        };
        rows[n] = most;
        n += 1;
        var f = x.e.forward();
        const clock = std.Io.Clock.awake;
        for (rows[0..n], x.costs[0..n]) |r, *cost| {
            const parts_n = (r + st.max_rows - 1) / st.max_rows;
            var parts: [st.max_streams]Part = undefined;
            for (parts[0..parts_n], 0..) |*part, i| part.* = .{ .seq = &seq, .tokens = zeros[0 .. r / parts_n + @intFromBool(i < r % parts_n)] };
            var times: [3]f64 = undefined;
            for (0..4) |rep| {
                try f.ops.s.synchronize();
                const t0 = clock.now(io).toNanoseconds();
                try f.round(parts[0..parts_n]);
                try f.ops.s.synchronize();
                if (rep > 0) times[rep - 1] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e6;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            cost.* = .{ .width = @intCast(r), .ms = times[1] };
        }
        x.cost_count = n;
    }

    const ordinary_vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .probabilities = probabilities, .tree = tree, .release = release, .prefill_many = prefillMany };
    const post_first_vtable: be.Backend.VTable = blk: {
        var table = ordinary_vtable;
        table.after_first = @import("exact_post_first.zig").after;
        break :blk table;
    };

    pub fn backend(x: *Cuda) be.Backend {
        return .{ .ptr = x, .first_before_draft = true, .vtable = if (x.exact_cache.post_first.enabled) &post_first_vtable else &ordinary_vtable };
    }

    /// The stream's drafting (its DFlash2 context and copy index), made on first use.
    pub fn draftingOf(x: *Cuda, s: *lanes.Stream) !*drafts.Stream {
        return draft_reuse.obtain(x, s);
    }

    fn cast(p: *anyopaque) *Cuda {
        return @ptrCast(@alignCast(p));
    }

    fn seqOf(x: *Cuda, s: *lanes.Stream) !*st.Seq {
        return x.seqs.get(s) orelse error.UnknownQwenStream;
    }

    /// Commits the last verified round: every stream its kept rows (its whole window unless keep cut it).
    pub fn settle(x: *Cuda) !void {
        const p = &x.pending;
        if (p.n == 0) return;
        const n = p.n;
        p.n = 0;
        var f = x.e.forward();
        try f.commit(p.parts[0..n], p.paths[0..n]);
    }

    /// The stream's state, the last round committed first.
    pub fn settled(x: *Cuda, s: *lanes.Stream) !*st.Seq {
        try x.settle();
        return x.seqOf(s);
    }

    /// The round's logits rows drawn: greedy by one argmax, sampled from top (top_k + 8) GPU candidates or whole rows.
    pub fn drawRows(x: *Cuda, rules: []const ?lanes.Sampling, positions: []const u64, out: []u32) !void {
        const rows = positions.len;
        var f = x.e.forward();
        const s = f.s;
        try f.picks(rows, out[0..rows]);
        var want: usize = 0;
        for (rules) |r| if (r) |rule| if (rule.temperature > 0 and rule.top_k != 0) {
            want = @max(want, @min(c.vocab, @as(usize, rule.top_k) + 8));
        };
        if (want > st.max_candidates) want = 0; // past the GPU's candidates: whole rows below
        if (want > 0) {
            const t = f.ops.torch();
            try t.toF32(s.logits, s.logits32, rows * c.vocab);
            try t.topk(s.logits32, c.vocab, rows, want, s.cand_values, s.cand_ids, s.topk_scratch);
            try f.ops.download(std.mem.sliceAsBytes(x.cand_values[0 .. rows * want]), s.cand_values);
            try f.ops.download(std.mem.sliceAsBytes(x.cand_ids[0 .. rows * want]), s.cand_ids);
            try f.ops.s.synchronize();
        }
        for (rules, positions, out[0..rows], 0..) |rule_, pos, *o, r| {
            const rule = rule_ orelse continue;
            if (rule.temperature <= 0) continue;
            if (rule.top_k != 0 and want > 0) {
                const vals = x.values[0..want];
                for (vals, x.cand_values[r * want ..][0..want]) |*v, w| v.* = w;
                const ids = x.cand_ids[r * want ..][0..want];
                o.* = @intCast(try lanes.sampling.choose(x.gpa, vals, @ptrCast(ids), pos, rule));
                continue;
            }
            try f.ops.download(std.mem.sliceAsBytes(x.logits), s.logits + r * c.vocab * 2);
            try f.ops.s.synchronize();
            o.* = try draw(x.gpa, x.logits, rule, pos);
        }
    }

    pub const open = exact_reuse.open;

    fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        const x = cast(p);
        var opened = try x.openPrefill(s);
        errdefer x.dropSeq(s);
        try x.prefillOpened(s, &opened);
    }

    /// prefill_state into an opened stream: the prompt in chunks, a drafting stream's last window of taps absorbed. A
    /// kept state the lane host found is restored first, and the pass stops at its marks for the cache to copy.
    fn prefillOpened(x: *Cuda, s: *lanes.Stream, opened: *exact_reuse.Opened) !void {
        errdefer exact_reuse.fail(x);
        const seq = opened.seq;
        if (x.exact_cache.eligible(s) and !x.e.kv8) {
            const ctx = if (x.drafter != null and s.drafts) &(try x.draftingOf(s)).ctx else null;
            if (try exact_reuse.lookupPrefill(x, s, opened, ctx)) return;
        }
        // an image prompt: the tower's rows and the prompt's three positions for its pass (EncodedVision)
        const media: ?*vision.Media = if (s.images.len == 0) null else blk: {
            const tower = x.tower orelse return error.VisionDisabled;
            const gpu: vision.Gpu = .{ .d = x.e.ops().k.d, .s = x.e.ops().s, .discrete = x.e.ops().k.discrete };
            break :blk try vision.Media.init(x.gpa, tower, gpu, s.prompt(), s.images, prompt_rows);
        };
        defer if (media) |m| m.deinit();
        const from = x.restoreKept(s, seq, media != null);
        // an image prompt keeps nothing: its rows' positions are the tower's, not the tokens'
        const marks: ?eng.Keep = if (s.reuse.marks.len == 0 or media != null) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
        if (x.drafter) |d| if (s.drafts) {
            const ds = try x.draftingOf(s);
            try lone.prefillFrom(x.e, seq, &ds.ctx, d, s.prompt(), from, media, marks);
            try x.exact_cache.captureBeforeFirst(s, seq, &ds.ctx, x.e.ops(), x.e.w.g);
            return draft_reuse.initialized(x, s);
        };
        try x.e.prefillFrom(seq, s.prompt(), from, media, marks);
        try x.exact_cache.captureBeforeFirst(s, seq, null, x.e.ops(), x.e.w.g);
    }

    /// The stream's kept state restored on its sequence: where its prompt pass starts (0: none, off the grid, failed).
    fn restoreKept(x: *Cuda, s: *lanes.Stream, seq: *st.Seq, image: bool) usize {
        s.cached = 0;
        const saved = s.reuse.saved orelse return 0;
        if (image or !x.e.grid or !eng.resumable(s.reuse.at, s.prompt().len)) {
            s.reuse_failed = true;
            return 0;
        }
        reuse.restoreSeq(x.e, seq, saved) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = s.reuse.at;
        return s.reuse.at;
    }

    /// The prompt pass stands at a mark: the lane host copies the stream's state there.
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    /// prefill_batch: prompts in shared passes of at most a chunk's rows (a longer one alone, in its chunks).
    fn prefillMany(p: *anyopaque, streams: []const *lanes.Stream, errs: []?anyerror) !void {
        const x = cast(p);
        var pieces: [st.max_streams]Forward.Piece = undefined;
        var which: [st.max_streams]usize = undefined;
        var n: usize = 0;
        var rows: usize = 0;
        for (streams, errs, 0..) |s, *err, i| {
            var opened = x.openPrefill(s) catch |e| {
                err.* = e;
                continue;
            };
            const seq = opened.seq;
            const ids = s.prompt();
            if (x.exact_cache.eligible(s) and !x.e.kv8) {
                const ctx = if (x.drafter != null and s.drafts) &(x.draftingOf(s) catch |e| {
                    x.dropSeq(s);
                    err.* = e;
                    continue;
                }).ctx else null;
                const hit = exact_reuse.lookupPrefill(x, s, &opened, ctx) catch |e| {
                    x.dropSeq(s);
                    err.* = e;
                    continue;
                };
                if (hit) continue;
            }
            if (ids.len > prompt_rows) {
                x.prefillOpened(s, &opened) catch |e| {
                    x.dropSeq(s);
                    err.* = e;
                };
                continue;
            }
            if (n == st.max_streams or rows + ids.len > prompt_rows) {
                x.pass(streams, errs, pieces[0..n], which[0..n]);
                n = 0;
                rows = 0;
            }
            pieces[n] = .{ .ids = ids, .seq = seq, .last = true };
            which[n] = i;
            n += 1;
            rows += ids.len;
        }
        if (n > 0) x.pass(streams, errs, pieces[0..n], which[0..n]);
    }

    /// One shared prompt pass; on a failure every stream in it fails.
    fn pass(x: *Cuda, streams: []const *lanes.Stream, errs: []?anyerror, pieces: []const Forward.Piece, which: []const usize) void {
        x.passPieces(streams, pieces, which) catch |e| for (which) |i| {
            x.dropSeq(streams[i]);
            errs[i] = e;
        };
    }

    fn passPieces(x: *Cuda, streams: []const *lanes.Stream, pieces: []const Forward.Piece, which: []const usize) !void {
        var f = x.e.forward();
        const d = x.drafter orelse {
            try f.chunks(pieces);
            if (x.exact_cache.enabled and !x.e.kv8) for (pieces, which) |pc, i| {
                try x.exact_cache.captureBeforeFirst(streams[i], pc.seq, null, x.e.ops(), x.e.w.g);
            };
            return;
        };
        f.taps = x.e.scratch.taps;
        try f.chunks(pieces);
        // each drafting stream's last window of taps, consecutive whole prompts absorbed together
        const row = drafts.taps_width * 2;
        var group: [st.max_streams]dr.Absorb = undefined;
        var g: usize = 0;
        var g_rows: usize = 0;
        var g_from: usize = 0; // the group's first tap row
        var at: usize = 0; // this piece's first row in the pass
        for (pieces, which) |pc, i| {
            defer at += pc.ids.len;
            const s = streams[i];
            if (!s.drafts) continue;
            const ds = try x.draftingOf(s);
            ds.ctx.len = 0;
            ds.ctx.end = 0;
            const tap_from = pc.ids.len -| dl.window;
            ds.ctx.skip(tap_from);
            const taken = pc.ids.len - tap_from;
            if (g > 0 and (tap_from > 0 or g_from + g_rows != at or g_rows + taken > dl.window + 1)) {
                try d.absorbMany(group[0..g], x.e.scratch.taps + g_from * row);
                g = 0;
            }
            if (g == 0) {
                g_from = at + tap_from;
                g_rows = 0;
            }
            group[g] = .{ .ctx = &ds.ctx, .rows = taken };
            g += 1;
            g_rows += taken;
        }
        if (g > 0) try d.absorbMany(group[0..g], x.e.scratch.taps + g_from * row);
        for (pieces, which) |pc, i| {
            const s = streams[i];
            const ctx = if (s.drafts) &(try x.draftingOf(s)).ctx else null;
            if (!x.e.kv8) try x.exact_cache.captureBeforeFirst(s, pc.seq, ctx, x.e.ops(), x.e.w.g);
            if (ctx != null) try draft_reuse.initialized(x, s);
        }
    }

    fn dropSeq(x: *Cuda, s: *lanes.Stream) void {
        if (x.draft_pool.stats.enabled) draft_reuse.drop(x, s);
        _ = x.restored.remove(s);
        exact_reuse.fail(x);
        x.exact_cache.abandon(s, x.e.ops()) catch |err| std.log.err("exact cache abandon: {s}", .{@errorName(err)});
        if (x.seqs.fetchRemove(s)) |entry| {
            entry.value.deinit();
            x.gpa.destroy(entry.value);
        }
    }

    const first = @import("exact_first_logits.zig").first;

    fn queue(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const x = cast(p);
        const seq = try x.settled(s);
        if (position != seq.pos + 1) return error.InvalidQwenPosition;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        var f = x.e.forward();
        const part = [_]Part{.{ .seq = seq, .tokens = &.{token} }};
        try f.round(&part);
        var out: [1]u32 = undefined;
        try x.drawRows(&.{s.sampling}, &.{position}, &out);
        try f.commit(&part, &.{&.{0}});
        return out[0];
    }

    fn read(_: *anyopaque, handle: u64) !u32 {
        if (handle >= c.vocab) return error.InvalidQwenToken;
        return @intCast(handle);
    }

    /// One forward over every window (one stream: tree_forward's, several: multi_tree_forward's), each row drawn.
    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const x = cast(p);
        if (windows.len != out.len or windows.len == 0 or windows.len > st.max_streams) return error.InvalidQwenWindows;
        try x.settle();
        const pend = &x.pending;
        var total: usize = 0;
        for (windows, out, 0..) |w, o, k| {
            const rows = w.rows();
            if (w.held != 0 or w.positions.len != rows or rows > st.max_rows or total + rows > st.round_rows) return error.UnsupportedQwenWindow;
            if (o.sampled.len != rows or o.drafts.len != w.tokens.len) return error.InvalidQwenOutputs;
            const seq = try x.seqOf(w.stream);
            pend.tokens[total] = w.pending;
            @memcpy(pend.tokens[total + 1 ..][0..w.tokens.len], w.tokens);
            const parents: ?[]const i32 = if (w.parents) |given| blk: {
                if (given.len != rows) return error.InvalidQwenWindows;
                @memcpy(pend.parents[total..][0..rows], given);
                break :blk pend.parents[total..][0..rows];
            } else null;
            // each row's draw is keyed at its depth past the committed position
            var depth: [st.max_rows]usize = undefined;
            if (parents) |ps| {
                try trees.depths(ps, depth[0..rows]);
            } else for (depth[0..rows], 0..) |*d, r| {
                d.* = r;
            }
            for (w.positions, depth[0..rows]) |pos, d| if (pos != seq.pos + d + 1) return error.InvalidQwenPosition;
            pend.parts[k] = .{ .seq = seq, .tokens = pend.tokens[total..][0..rows], .parents = parents };
            pend.starts[k] = total;
            pend.streams[k] = w.stream;
            // a window kept whole: its rows in order (keep replaces them with its path)
            for (pend.path_rows[total..][0..rows], 0..) |*row, r| row.* = @intCast(r);
            pend.paths[k] = pend.path_rows[total..][0..rows];
            total += rows;
        }
        var f = x.e.forward();
        if (x.drafter != null) f.taps = x.e.scratch.taps;
        try f.round(pend.parts[0..windows.len]);
        pend.n = windows.len;
        var rules: [st.round_rows]?lanes.Sampling = undefined;
        var positions: [st.round_rows]u64 = undefined;
        var drawn: [st.round_rows]u32 = undefined;
        var row: usize = 0;
        for (windows) |w| for (w.positions) |pos| {
            rules[row] = w.stream.sampling;
            positions[row] = pos;
            row += 1;
        };
        try x.drawRows(rules[0..total], positions[0..total], &drawn);
        row = 0;
        for (windows, out) |w, o| {
            @memcpy(o.sampled, drawn[row..][0..o.sampled.len]);
            row += o.sampled.len;
            @memcpy(o.drafts, w.tokens);
        }
    }

    /// A window keeps its accepted path (a chain's prefix, or a tree's root-to-node rows); verify keeps all of it.
    fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const x = cast(p);
        const pend = &x.pending;
        for (windows, paths) |w, path| {
            const k = for (pend.streams[0..pend.n], 0..) |s, k| {
                if (s == w.stream) break k;
            } else return error.UnknownQwenStream;
            const rows = pend.parts[k].tokens.len;
            if (path.len == 0 or path.len > rows) return error.InvalidQwenCommit;
            const slot: [*]u32 = @constCast(pend.paths[k].ptr);
            for (path, 0..) |r, i| {
                if (r >= rows) return error.InvalidQwenCommit;
                slot[i] = r;
            }
            pend.paths[k] = slot[0..path.len];
        }
    }

    /// The kept rows' taps into each stream's drafter context, then each stream's next drafts.
    fn draft(p: *anyopaque, requests: []const be.DraftRequest) !void {
        const x = cast(p);
        const d = x.drafter orelse {
            for (requests) |r| if (r.depth != 0) return error.QwenCheckpointHasNoDraftHead;
            return;
        };
        const pend = &x.pending;
        var kept: [st.max_streams]drafts.Kept = undefined;
        var rows: [st.round_rows]u32 = undefined;
        var asks: [st.max_streams]drafts.Ask = undefined;
        var firsts: [st.max_streams]std.ArrayList(u32) = @splat(.empty);
        defer for (&firsts) |*l| l.deinit(x.gpa);
        var nk: usize = 0;
        var nr: usize = 0;
        var na: usize = 0;
        for (requests) |r| {
            if (na == st.max_streams) return error.InvalidQwenWindows;
            const ds = x.drafting.get(r.stream) orelse continue;
            var context = r.stream.context.items;
            if (r.rows) |path| {
                const k = for (pend.streams[0..pend.n], 0..) |s, k| {
                    if (s == r.stream) break k;
                } else return error.UnknownQwenStream;
                for (path, rows[nr..][0..path.len]) |row, *g| g.* = @intCast(pend.starts[k] + row);
                kept[nk] = .{ .x = ds, .rows = rows[nr..][0..path.len] };
                nk += 1;
                nr += path.len;
            } else if (r.first) |feed| {
                // the prompt's taps went in with its prefill; the first token ends the context the copies read
                const token: u32 = switch (feed) {
                    .value => |v| v,
                    .handle => |h| try read(p, h),
                };
                try firsts[na].appendSlice(x.gpa, r.stream.prompt());
                try firsts[na].append(x.gpa, token);
                context = firsts[na].items;
            }
            asks[na] = .{ .x = ds, .context = context, .sampling = r.stream.sampling };
            na += 1;
        }
        const f = x.e.forward();
        try drafts.absorb(d, f.ops, x.e.scratch.taps, x.e.scratch.copy_items, kept[0..nk]);
        try drafts.propose(d, asks[0..na]);
    }

    /// The drafts a stream holds as host tokens and parents (a copy's as a chain).
    fn tree(p: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) !?lanes.stream.Held {
        const x = cast(p);
        const ds = x.drafting.get(s) orelse return null;
        const tokens = try gpa.dupe(u32, ds.tokens.items);
        errdefer gpa.free(tokens);
        const parents: ?[]i32 = if (ds.kind == .tree) try gpa.dupe(i32, ds.parents.items) else null;
        return .{ .count = @intCast(tokens.len), .tokens = tokens, .parents = parents };
    }

    /// Each held draft's chance of landing: a tree node's exp(-path score) (allocate's), a copied token's 1.
    fn probabilities(p: *anyopaque, s: *lanes.Stream, out: []f64) !bool {
        const x = cast(p);
        const ds = x.drafting.get(s) orelse return false;
        if (out.len > ds.scores.items.len) return false;
        for (out, ds.scores.items[0..out.len]) |*o, score| o.* = @exp(-score);
        return true;
    }

    const release = exact_reuse.release;
};

fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

/// The Metal backend's draw: the top-k candidates by value then id, then the shared fp64 keyed sampler.
pub fn draw(gpa: std.mem.Allocator, logits: []const u16, settings: ?lanes.Sampling, position: u64) !u32 {
    var best: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteQwenLogits;
        if (x > value(logits[best])) best = id;
    }
    const sampling = settings orelse return @intCast(best);
    if (sampling.temperature <= 0) return @intCast(best);
    const k = if (sampling.top_k == 0) logits.len else @min(logits.len, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (logits, 0..) |word, id| {
        const x = value(word);
        if (k == logits.len) {
            values[id] = x;
            ids[id] = id;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}

test {
    _ = @import("exact_cache_test.zig");
    _ = @import("exact_eligible.zig");
    _ = @import("backend_vtable_test.zig");
}
