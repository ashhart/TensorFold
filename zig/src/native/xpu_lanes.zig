//! The XPU 27B on the lane host: a stream per runner slot, shared rounds across slots, a lone drafted stream on trees.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const be = lanes.backend;
const Drafter = xq.drafter.Drafter;
const Allocator = std.mem.Allocator;

const ring = 64;
const TAP_ROW = 5 * xq.D * 2; // bf16 [5 D]
const TREE_MIN = 5; // drafts from which a stream's window is a DFlash2 tree (below: its chain)

/// A kept prompt state (prompt cache): the slot's states after `at` tokens and its DFlash2 context.
pub const Kept = struct { snap: xq.Runner.Snap, ctx: ?*anyopaque = null, range: [2]u32 = .{ 0, 0 } };

pub const Lanes = struct {
    gpa: Allocator,
    runner: *xq.Runner,
    drafter: ?*Drafter,
    pick: *const fn (ctx: *anyopaque, row: []const f32, position: u64, s: api.Sampling) anyerror!u32,
    pick_ctx: *anyopaque,
    slots: u32,
    block: u32 = 8,
    nodes: u32 = 15,
    owners: [xq.MAX_SLOTS]?*lanes.Stream = @splat(null),
    first_logits: [xq.MAX_SLOTS]?[]f32 = @splat(null), // a sampled or structured stream's prompt-row logits, for `first`
    mask_rows: [16]u32 = undefined, // a verify's greedy structured rows (argmaxMasked on the device)
    mask_bits: ?[]u32 = null, // and their grammar bits, gathered
    masked_vals: ?[]f64 = null, // a structured sampled row's allowed candidates
    masked_ids: ?[]u64 = null,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    ids: [16]u32 = undefined, // the last verify's rows, window after window
    widths: [16]u32 = undefined, // and each window's rows
    count: usize = 0,
    wslots: [16]u32 = undefined, // and each window's slot
    wfirst: [16]u32 = undefined, // and its first row
    absorbed: [16]bool = @splat(false), // the window's kept rows already in its slot's drafter context
    chain: [xq.MAX_SLOTS][16]u32 = undefined, // each slot's DFlash2 drafts for its next window (a tree)
    chain_par: [xq.MAX_SLOTS][16]i32 = undefined, // and each one's parent (−1: the pending token)
    chain_n: [xq.MAX_SLOTS]u32 = @splat(0),
    prompt_skip: u32 = 0, // a prefill's first rows the drafter never reads (further than WIN from the prompt's end)
    costs: [16]lanes.config.Cost = undefined, // a window's ms by rows (verify and keep), timed by measure()
    timed: usize = 0,

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn backend(b: *Lanes) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .tree = treeFn, .release = releaseFn } };
    }

    /// What the round loop reads: chains of up to 16 rows, all streams' windows in one shared forward, DFlash2 drafts
    pub fn facts(b: *const Lanes) lanes.Model {
        const head = b.drafter != null;
        return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = if (head) b.block - 1 else 0, .mtp_step_ms = if (head) 0.6 else 0, .draft_streams = head, .window_costs = b.costs[0..b.timed], .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = b.slots, .lone_copies = false };
    }

    /// Each window width's cost on a scratch context in the last slot, for the round loop's depth rule
    pub fn measure(b: *Lanes) !void {
        const r = b.runner;
        const slot = b.slots - 1;
        try r.select(slot);
        try r.reset();
        var ids: [128]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % xq.VOCAB);
        try r.prompt(&ids, false);
        const one = [_]u32{0};
        const time = struct {
            fn f(rr: *xq.Runner, sl: u32, row: []const u32, k: []const u32) !f64 {
                const t0 = xq.now();
                try rr.verifyMulti(&.{.{ .slot = sl, .ids = row }});
                try rr.keepPaths(&.{k});
                try rr.d.sync();
                return xq.now() - t0;
            }
        }.f;
        for (0..3) |_| _ = try time(r, slot, ids[0..16], &one); // the widest first, while the clocks ramp up
        for (&b.costs, 1..) |*c, w| {
            var best: f64 = 1e9;
            for (0..3) |_| best = @min(best, try time(r, slot, ids[0..w], &one));
            c.* = .{ .width = @intCast(w), .ms = best * 1e3 };
        }
        b.timed = b.costs.len;
        try r.reset();
    }

    /// Every forward's taps reach the drafter context of the slot it ran (with a drafter attached).
    pub fn initDrafts(b: *Lanes) void {
        b.runner.on_taps = .{ .ctx = b, .f = onTaps };
    }

    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) anyerror!void {
        const b = self(ctx);
        const d = b.drafter.?;
        try d.use(b.runner.slot);
        const k = @min(n, b.prompt_skip);
        b.prompt_skip -= k;
        d.skip(k);
        if (k < n) try d.addTaps(xq.off(taps, @as(usize, k) * TAP_ROW), n - k);
    }

    /// The last verify's windows `which`: each kept path's taps go into its slot's drafter context in one pass.
    fn absorb(b: *Lanes, which: []const usize, paths: []const []const u32) !void {
        const d = b.drafter orelse return;
        var items: [16]xq.drafter.Absorb = undefined;
        for (which, paths, items[0..which.len]) |i, path, *it| {
            var straight = true;
            for (path, 0..) |row, j| straight = straight and row == j;
            it.* = .{ .slot = b.wslots[i], .taps = xq.off(b.runner.acts.taps, @as(usize, b.wfirst[i]) * TAP_ROW), .n = @intCast(path.len), .rows = if (straight) null else path };
            b.absorbed[i] = true;
        }
        try d.absorbMulti(items[0..which.len]);
    }

    /// Keeps the root path `paths[i]` of the last verify's window i (its taps to the drafter, unless taken already).
    fn keepRows(b: *Lanes, paths: []const []const u32) !void {
        var which: [16]usize = undefined;
        var ps: [16][]const u32 = undefined;
        var n: usize = 0;
        for (paths, 0..) |p, i| if (!b.absorbed[i]) {
            which[n] = i;
            ps[n] = p;
            n += 1;
        };
        try b.absorb(which[0..n], ps[0..n]);
        @memset(&b.absorbed, false);
        try b.runner.keepPaths(paths);
    }

    /// A verify whose rows all stayed: the core calls keep only to drop rows, so the next call keeps them all first.
    fn settle(b: *Lanes) !void {
        if (b.count == 0) return;
        const n = b.count;
        b.count = 0;
        const all = [16]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        var paths: [16][]const u32 = undefined;
        for (paths[0..n], b.widths[0..n]) |*p, w| p.* = all[0..w];
        try b.keepRows(paths[0..n]);
    }

    fn slotOf(b: *const Lanes, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn push(b: *Lanes, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn value(b: *const Lanes, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| if (h < b.next and b.next - h <= ring) b.drawn[h % ring] else error.NoSuchToken,
        };
    }

    /// `slot`'s next greedy token (the runner keeps it per slot).
    fn lastOf(b: *const Lanes, slot: u32) u32 {
        const r = b.runner;
        return if (r.slot == slot) r.last else r.seqs[slot].last;
    }

    /// A stream's prompt into its slot (a free one, slot 0 last); its last row's logits are kept when it samples.
    fn prefillSlot(b: *Lanes, s: *lanes.Stream, slot: u32) !void {
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 17 > b.runner.context) return error.PromptTooLong;
        const r = b.runner;
        try r.select(slot);
        try r.reset();
        b.chain_n[slot] = 0;
        if (b.drafter) |d| {
            try d.use(slot);
            d.reset();
            r.taps_on = true;
            b.prompt_skip = @intCast(ids.len -| xq.drafter.WIN);
        }
        defer {
            r.taps_on = false;
            b.prompt_skip = 0;
        }
        s.cached = 0;
        var at: usize = 0;
        if (s.reuse.saved) |saved| { // a kept state of this prompt's prefix: the pass starts there
            const k: *const Kept = @ptrCast(@alignCast(saved));
            if (k.snap.at < ids.len) {
                if (b.restoreKept(slot, k)) {
                    at = k.snap.at;
                    s.cached = k.snap.at;
                } else |_| {
                    s.reuse_failed = true;
                    try r.select(slot);
                    try r.reset();
                    if (b.drafter) |d| {
                        try d.use(slot);
                        d.reset();
                    }
                }
            } else s.reuse_failed = true;
        }
        if (b.drafter != null) b.prompt_skip = @intCast((ids.len -| xq.drafter.WIN) -| at);
        // chunks cut where the request's chunks and the cache's marks are, as a fresh pass cuts them
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            var end = @min(ids.len, at + xq.CHUNK);
            for (s.chunks) |cut| if (cut > at and cut < end) {
                end = cut;
            };
            for (s.reuse.marks) |mark| if (mark > at and mark < end) {
                end = mark;
            };
            try r.prompt(ids[at..end], end == ids.len);
            at = end;
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(at)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
        if (s.sampling != null or s.grammar != null) {
            const row = try r.logits();
            const keep = b.first_logits[slot] orelse try b.gpa.alloc(f32, row.len);
            @memcpy(keep, row);
            b.first_logits[slot] = keep;
        }
    }

    /// The bytes keepSave takes at `at` tokens.
    pub fn keepBytes(b: *const Lanes, at: u32) u64 {
        return b.runner.snapBytes(at) + if (b.drafter != null) Drafter.ctxBytes() else 0;
    }

    /// `s`'s slot standing at `at` prompt tokens, copied (its DFlash2 context too).
    pub fn keepSave(b: *Lanes, s: *const lanes.Stream, at: u32) !*Kept {
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const k = try b.gpa.create(Kept);
        errdefer b.gpa.destroy(k);
        k.* = .{ .snap = try b.runner.saveSlot(slot, at) };
        errdefer b.runner.dropSnap(k.snap);
        if (b.drafter) |d| {
            const buf = try b.runner.d.alloc_(Drafter.ctxBytes());
            errdefer b.runner.d.g.free(buf);
            k.range = try d.saveCtx(slot, buf);
            k.ctx = buf;
        }
        return k;
    }

    pub fn keepDrop(b: *Lanes, k: *Kept) void {
        b.runner.dropSnap(k.snap);
        if (k.ctx) |x| b.runner.d.g.free(x);
        b.gpa.destroy(k);
    }

    fn restoreKept(b: *Lanes, slot: u32, k: *const Kept) !void {
        try b.runner.restoreSlot(slot, k.snap);
        if (b.drafter) |d| try d.restoreCtx(slot, k.ctx orelse return error.NoContext, k.range);
    }

    fn freeSlot(b: *const Lanes) ?u32 {
        var k: u32 = b.slots;
        while (k > 0) {
            k -= 1;
            if (b.owners[k] == null) return k;
        }
        return null;
    }

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        try b.settle();
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        try b.prefillSlot(s, slot);
    }

    fn draw(b: *Lanes, s: *const lanes.Stream, slot: u32, row: ?[]const f32, position: u64) !u32 {
        const sp = s.sampling orelse return b.lastOf(slot);
        return b.pick(b.pick_ctx, row orelse return error.NoLogits, position, sp);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return b.push(try b.draw(s, slot, b.first_logits[slot], position));
    }

    /// Structured output: the stream's draw among the tokens `mask` allows, greedy (first argmax) or sampled (keyed)
    fn drawMasked(b: *Lanes, s: *const lanes.Stream, row: []const f32, position: u64, mask: []const u32) !u32 {
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const vals = b.masked_vals orelse try b.gpa.alloc(f64, row.len);
            b.masked_vals = vals;
            const ids = b.masked_ids orelse try b.gpa.alloc(u64, row.len);
            b.masked_ids = ids;
            const k: usize = if (sp.top_k == 0) row.len else sp.top_k;
            var n: usize = 0;
            for (row, 0..) |v32, i| {
                if (!lanes.grammar.allows(mask, i)) continue;
                const v: f64 = v32;
                if (!std.math.isFinite(v)) return error.NonfiniteLogits;
                if (n == k and !(v > vals[k - 1])) continue; // equal values keep the lower id
                var j = if (n < k) n else k - 1;
                if (n < k) n += 1;
                if (k < row.len) while (j > 0 and v > vals[j - 1]) : (j -= 1) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                };
                vals[j] = v;
                ids[j] = i;
            }
            if (n == 0) return error.NoAllowedToken;
            return @intCast(try lanes.sampling.choose(b.gpa, vals[0..n], ids[0..n], position, sp));
        };
        var best: ?usize = null;
        for (row, 0..) |v, id| if (lanes.grammar.allows(mask, id) and (best == null or v > row[best.?])) {
            best = id;
        };
        return @intCast(best orelse return error.NoAllowedToken);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return b.push(try b.drawMasked(s, b.first_logits[slot] orelse return error.NoLogits, position, mask));
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        try b.settle();
        const r = b.runner;
        try r.select(slot);
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        b.chain_n[slot] = 0;
        try r.step(try b.value(feed));
        return b.push(try b.draw(s, slot, if (s.sampling != null) try r.logits() else null, position));
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).value(.{ .handle = handle });
    }

    /// Every stream's window (pending token, then its copied drafts as a chain) in one forward, drawn with its sampling
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > 16 or windows.len != out.len) return error.BadWindows;
        try b.settle();
        var wins: [16]xq.Runner.Win = undefined;
        var total: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.held > 0) return error.HeldDrafts;
            const rows = w.rows();
            if (total + rows > 16) return error.WindowTooWide;
            const ids = b.ids[total..][0..rows];
            ids[0] = w.pending;
            @memcpy(ids[1..], w.tokens);
            wins[i] = .{ .slot = b.slotOf(w.stream) orelse return error.UnknownStream, .ids = ids, .parents = w.parents };
            b.wslots[i] = wins[i].slot;
            b.wfirst[i] = @intCast(total);
            b.widths[i] = @intCast(rows);
            total += rows;
        }
        const r = b.runner;
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        @memset(&b.absorbed, false);
        try r.verifyMulti(wins[0..windows.len]);
        // structured greedy rows: their tokens among their grammars' on the device, all in one pass
        var nm: usize = 0;
        var mwords: usize = 0;
        for (windows, 0..) |w, i| if (w.masks.len > 0 and w.stream.sampling == null) {
            const words = w.masks.len / b.widths[i];
            if (mwords != 0 and words != mwords) return error.BadMasks;
            mwords = words;
            const bits = b.mask_bits orelse try b.gpa.alloc(u32, 16 * xq.MASK_WORDS);
            b.mask_bits = bits;
            if (words > xq.MASK_WORDS) return error.BadMasks;
            for (0..b.widths[i]) |j| {
                b.mask_rows[nm] = b.wfirst[i] + @as(u32, @intCast(j));
                @memcpy(bits[nm * words ..][0..words], w.masks[j * words ..][0..words]);
                nm += 1;
            }
        };
        var mtok: [16]u32 = undefined;
        if (nm > 0) try r.argmaxMasked(b.mask_rows[0..nm], b.mask_bits.?[0 .. nm * mwords], mwords, &mtok);
        var mi: usize = 0;
        var row: u32 = 0;
        for (windows, out, 0..) |w, o, i| {
            const words = if (w.masks.len > 0) w.masks.len / b.widths[i] else 0;
            for (0..b.widths[i]) |j| {
                if (words > 0 and w.stream.sampling == null) {
                    o.sampled[j] = mtok[mi];
                    mi += 1;
                    row += 1;
                    continue;
                }
                if (words > 0) { // structured output, sampled: the row's grammar bits on the host
                    o.sampled[j] = try b.drawMasked(w.stream, try r.rowLogits(row), w.positions[j], w.masks[j * words ..][0..words]);
                    row += 1;
                    continue;
                }
                const logits = if (w.stream.sampling != null) try r.rowLogits(row) else null;
                o.sampled[j] = if (w.stream.sampling) |sp| try b.pick(b.pick_ctx, logits.?, w.positions[j], sp) else r.block[row];
                row += 1;
            }
            @memcpy(o.drafts, w.tokens);
        }
        b.count = windows.len;
    }

    /// Each window keeps a root path (a prefix of its chain, or a path down its tree).
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != b.count or paths.len != b.count) return error.NothingToKeep;
        for (paths) |p| if (p.len == 0) return error.NothingToKeep;
        b.count = 0;
        try b.keepRows(paths);
    }

    /// Each asked stream's DFlash2 chain for its next window in one drafter pass, after the kept rows' taps.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        const d = b.drafter orelse return;
        var which: [16]usize = undefined;
        var ps: [16][]const u32 = undefined;
        var na: usize = 0;
        var items: [16]xq.drafter.Block = undefined;
        var asked: [16]u32 = undefined;
        var reqs: [16]*const be.DraftRequest = undefined;
        var ni: usize = 0;
        var depth: u32 = 0;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.chain_n[slot] = 0;
            if (q.rows) |rows| if (b.count > 0) {
                const i = std.mem.indexOfScalar(u32, b.wslots[0..b.count], slot) orelse return error.UnknownStream;
                if (!b.absorbed[i]) {
                    which[na] = i;
                    ps[na] = rows;
                    na += 1;
                }
            };
            if (q.depth == 0 or ni == items.len) continue;
            items[ni] = .{ .slot = slot, .pending = if (q.rows != null) q.follow[q.follow.len - 1] else try b.value(q.first orelse return error.NoFirstToken) };
            asked[ni] = @min(q.depth, b.nodes);
            reqs[ni] = &q;
            depth = @max(depth, @min(asked[ni], b.block - 1));
            ni += 1;
        }
        try b.absorb(which[0..na], ps[0..na]);
        if (ni == 0) return;
        for (requests) |q| if (q.depth > 0) {
            try d.use(b.slotOf(q.stream).?);
            if (d.end() + 1 != q.position) return error.BadDraftPosition;
        };
        try d.latticeMulti(items[0..ni], depth);
        // each stream's best-first tree of its asked drafts over its lattice, with the target's keyed noise if sampled
        for (items[0..ni], asked[0..ni], reqs[0..ni], 0..) |it, n, q, i| {
            d.take(i);
            var noise: [16][xq.drafter.TOPK]f64 = undefined;
            const sp = q.stream.sampling;
            if (sp) |x| for (0..depth) |dd| for (0..xq.drafter.TOPK) |j| {
                noise[dd][j] = -@log(-@log(lanes.sampling.uniform(x.seed, q.position + dd, d.cand[dd][j])));
            };
            if (n < TREE_MIN) { // few drafts: a chain goes deeper than a tree of siblings
                d.chain(it.pending, n, b.chain[it.slot][0..n]);
                for (b.chain_par[it.slot][0..n], 0..) |*x, j| x.* = @as(i32, @intCast(j)) - 1;
                b.chain_n[it.slot] = n;
                continue;
            }
            b.chain_n[it.slot] = d.tree(it.pending, n, if (sp != null) &noise else null, if (sp) |x| x.temperature else 1.0, b.chain[it.slot][0..n], b.chain_par[it.slot][0..n]);
        }
    }

    /// A slot's held chain as host tokens (taken: the next window spends it).
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return null;
        const n = b.chain_n[slot];
        b.chain_n[slot] = 0;
        const tokens = try gpa.dupe(u32, b.chain[slot][0..n]);
        errdefer gpa.free(tokens);
        const parents = try gpa.dupe(i32, b.chain_par[slot][0..n]); // by draft, −1 the pending token
        return .{ .count = n, .tokens = tokens, .parents = parents };
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        b.settle() catch {};
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.chain_n[slot] = 0;
    }

    /// A lone drafted stream in slot 0 on DFlash2 trees until it ends (false) or another stream arrives (true).
    pub fn lone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        if (hooks.resumed) return resumeLone(ctx, s, hooks);
        const b = self(ctx);
        if (b.drafter == null) return error.DraftNotAttached;
        try b.settle();
        if (b.owners[0] != null) return error.StreamBusy;
        b.owners[0] = s;
        var handed = false;
        defer if (!handed) {
            b.owners[0] = null;
        };
        try b.prefillSlot(s, 0);
        const pending = try b.draw(s, 0, b.first_logits[0], s.prompt_len);
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        handed = try b.alone(s, 0, pending, hooks);
        return handed;
    }

    /// The stream the lane core was left with, alone again in its slot, on DFlash2 trees as `lone`.
    fn resumeLone(ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = self(ctx);
        const d = b.drafter orelse return error.DraftNotAttached;
        try b.settle();
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const r = b.runner;
        try r.select(slot);
        var handed = false;
        defer if (!handed) {
            b.owners[slot] = null;
        };
        try d.use(slot);
        if (d.end() != r.pos) d.restart(r.pos); // its context should end there: a cold one otherwise
        handed = try b.alone(s, slot, s.pending orelse return error.NoPending, hooks);
        return handed;
    }

    /// DFlash2 trees for `s` in `slot` from `pending` until it ends (false) or another stream arrives (true)
    fn alone(b: *Lanes, s: *lanes.Stream, slot: u32, pending0: u32, hooks: api.LoneHooks) !bool {
        const d = b.drafter.?;
        const r = b.runner;
        try d.use(slot);
        var pending = pending0;
        var out: [16]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            r.taps_on = true;
            defer r.taps_on = false;
            var sampled: Sampled = .{ .b = b, .s = s.sampling orelse .{ .seed = 0 } };
            const drawer: ?xq.drafter.Draw = if (s.sampling) |sp| .{ .ctx = &sampled, .row = Sampled.row, .gumbel = Sampled.gumbel, .temperature = sp.temperature } else null;
            const round = try xq.drafter.treeRound(d, pending, b.block - 1, b.nodes, drawer, &out);
            s.rounds += 1;
            s.drafted += round.drafted;
            s.accepted += round.accepted;
            var landed: [17]u32 = undefined;
            const n = round.tokens.len - 1;
            @memcpy(landed[0..n], round.tokens[1..]);
            landed[n] = round.terminal;
            _ = try s.commit(b.gpa, landed[0 .. n + 1], &.{});
            pending = round.terminal;
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    const Sampled = struct {
        b: *Lanes,
        s: api.Sampling,
        fn row(c_: *anyopaque, row_: u32, position: u64) !u32 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return c.b.pick(c.b.pick_ctx, try c.b.runner.rowLogits(row_), position, c.s);
        }
        fn gumbel(c_: *anyopaque, position: u64, id: u64) f64 {
            const c: *@This() = @ptrCast(@alignCast(c_));
            return -@log(-@log(lanes.sampling.uniform(c.s.seed, position, id)));
        }
    };
};
