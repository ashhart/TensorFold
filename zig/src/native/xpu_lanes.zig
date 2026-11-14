//! The XPU 27B on the lane host: a slot per stream, shared rounds; Drafted drafts from its taps, lone streams on trees.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const be = lanes.backend;
const Drafter = xq.drafter.Drafter;
const logrow = @import("xpu_logrow.zig");
const Allocator = std.mem.Allocator;

const ring = 64;
const TAP_ROW = 5 * xq.D * 2; // bf16 [5 D]

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
    first_logits: [xq.MAX_SLOTS]?[]f32 = @splat(null), // a sampled, structured or logprobs stream's prompt-row logits
    first_pick: [xq.MAX_SLOTS]u32 = @splat(0), // the token `first` drew from them, for the logprob row
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
    tapped: usize = 0, // the last verify's windows whose taps acts.taps still holds (features)
    tail: ?Tail = null, // the prompt chunk whose taps acts.taps still holds (features)
    sink: ?struct { s: *lanes.Stream, take: be.FeatureSink } = null, // a drafting prompt pass hands chunks here
    direct: bool = false, // the lone driver's prompt pass: taps straight into the slot's drafter context
    wrap: ?*lanes.drafted.Drafted = null, // the drafter in front of this backend for lane rounds
    prompt_skip: u32 = 0, // a prefill's first rows the drafter never reads (further than WIN from the prompt's end)
    costs: [16]lanes.config.Cost = undefined, // a window's ms by rows (verify and keep), timed by measure()
    timed: usize = 0,

    pub const max_slots = xq.MAX_SLOTS;
    pub const tap_row = TAP_ROW;
    pub const draft_step_ms = 0.6; // a DFlash2 draft pass's ms, as the round loop prices it
    pub const tap_layers = blk: {
        var t: [xq.TAP_LAYERS.len]u32 = undefined;
        for (xq.TAP_LAYERS, &t) |l, *x| x.* = l;
        break :blk t;
    };
    /// Rows [start, start + rows) of a prompt pass in `slot`, their taps in acts.taps.
    pub const Tail = struct { slot: u32, start: u64, rows: u32 };

    fn self(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn backend(b: *Lanes) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .prefill_features = prefillFeaturesFn, .features = featuresFn, .first = firstFn, .first_masked = firstMaskedFn, .first_row = firstRowFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    /// What the round loop reads: chains of up to 16 rows, all streams' windows in one shared forward, DFlash2 drafts
    pub fn facts(b: *const Lanes) lanes.Model {
        const head = b.drafter != null;
        return .{ .exact_width = 16, .mtp = head, .speculate = head, .speculate_early = false, .drafts = if (head) b.block - 1 else 0, .mtp_step_ms = if (head) draft_step_ms else 0, .draft_streams = head, .window_costs = b.costs[0..b.timed], .hidden_rows = true, .streams_exact = true, .batch_rows = 16, .max_streams = b.slots, .lone_copies = false };
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
        b.tapped = 0;
        b.tail = null;
        try r.reset();
    }

    /// Every forward's taps reach the drafter context of the slot it ran (with a drafter attached).
    pub fn initDrafts(b: *Lanes) void {
        b.runner.on_taps = .{ .ctx = b, .f = onTaps };
    }

    fn onTaps(ctx: *anyopaque, taps: *anyopaque, n: u32) anyerror!void {
        const b = self(ctx);
        if (b.sink) |k| { // a prompt chunk, its first row at the runner's position (prompt() moves it after)
            b.tail = .{ .slot = b.runner.slot, .start = b.runner.pos, .rows = n };
            return k.take.take(k.take.ptr, k.s, rowsAt(taps, 0, n), b.runner.pos);
        }
        if (!b.direct) return;
        const d = b.drafter.?;
        try d.use(b.runner.slot);
        const k = @min(n, b.prompt_skip);
        b.prompt_skip -= k;
        d.skip(k);
        if (k < n) try d.addTaps(xq.off(taps, @as(usize, k) * TAP_ROW), n - k);
    }

    /// `n` tapped rows from row `row` of `taps`, as a drafter reads them.
    fn rowsAt(taps: *anyopaque, row: u64, n: u32) be.Features {
        return .{ .buffer = @intFromPtr(taps), .offset = row * TAP_ROW, .rows = n, .row_bytes = TAP_ROW, .space = .device, .dtype = .bf16 };
    }

    /// A stream's rows in acts.taps: the last verify's window (a drafter absorbs its kept rows), else its prompt chunk.
    fn featuresFn(ptr: *anyopaque, s: *lanes.Stream, _: []const u32, start: u64, count: u32) anyerror!be.Features {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (std.mem.indexOfScalar(u32, b.wslots[0..b.tapped], slot)) |i| {
            if (count > b.widths[i]) return error.FeaturesGone;
            return rowsAt(b.runner.acts.taps, b.wfirst[i], count);
        }
        const t = b.tail orelse return error.FeaturesGone;
        if (t.slot != slot or start < t.start or start + count > t.start + t.rows) return error.FeaturesGone;
        return rowsAt(b.runner.acts.taps, start - t.start, count);
    }

    /// The prompt chunk in acts.taps if `slot` ran it last (the drafter takes the prompt's last row with its chunk).
    pub fn promptTail(b: *const Lanes, slot: u32) ?Tail {
        const t = b.tail orelse return null;
        return if (t.slot == slot) t else null;
    }

    /// Keeps the root path `paths[i]` of the last verify's window i.
    fn keepRows(b: *Lanes, paths: []const []const u32) !void {
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

    pub fn slotOf(b: *const Lanes, s: *const lanes.Stream) ?u32 {
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
        b.tapped = 0;
        b.tail = null;
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
            var end = xq.chunkEnd(at, ids.len);
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
        if (s.sampling != null or s.grammar != null or s.logprobs != null) {
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
            errdefer xq.freeAfterQueue(&b.runner.d, buf); // saveCtx's copies may be queued
            k.range = try d.saveCtx(slot, buf);
            k.ctx = buf;
        }
        return k;
    }

    pub fn keepDrop(b: *Lanes, k: *Kept) void {
        b.runner.dropSnap(k.snap);
        if (k.ctx) |x| xq.freeAfterQueue(&b.runner.d, x);
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

    /// A drafting stream's prompt pass (Drafted in front): each chunk's taps to `sink` while acts.taps holds them.
    fn prefillFeaturesFn(ptr: *anyopaque, s: *lanes.Stream, _: []const u32, sink: be.FeatureSink) anyerror!void {
        const b = self(ptr);
        b.sink = .{ .s = s, .take = sink };
        defer b.sink = null;
        return prefillFn(ptr, s);
    }

    fn draw(b: *Lanes, s: *const lanes.Stream, slot: u32, row: ?[]const f32, position: u64) !u32 {
        const sp = s.sampling orelse return b.lastOf(slot);
        return b.pick(b.pick_ctx, row orelse return error.NoLogits, position, sp);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        const token = try b.draw(s, slot, b.first_logits[slot], position);
        b.first_pick[slot] = token;
        return b.push(token);
    }

    /// The prompt's last row for a logprobs stream: the raw distribution the first token was drawn from.
    fn firstRowFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!lanes.LogprobRow {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        return logrow.row(b.first_logits[slot] orelse return error.NoLogits, b.first_pick[slot], s.logprobs orelse 0);
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
        const token = try b.drawMasked(s, b.first_logits[slot] orelse return error.NoLogits, position, mask);
        b.first_pick[slot] = token;
        return b.push(token);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        try b.settle();
        const r = b.runner;
        try r.select(slot);
        r.taps_on = b.drafter != null;
        defer r.taps_on = false;
        b.tapped = 0;
        b.tail = null;
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
        b.tapped = 0;
        b.tail = null;
        try r.verifyMulti(wins[0..windows.len]);
        b.tapped = windows.len;
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
        try logrow.windowRows(r, windows, out, b.widths[0..windows.len]); // logprobs, for the streams that asked
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

    /// No drafts of its own: an attached drafter drafts through lanes.drafted.Drafted in front of this backend.
    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |q| if (q.depth > 0) return error.DraftNotAttached;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        b.settle() catch {};
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
    }

    /// A stream the lone driver ended: out of its slot, and out of Drafted's streams if it came from the lanes.
    fn forget(b: *Lanes, s: *lanes.Stream) void {
        if (b.wrap) |w| return w.backend().release(s); // its release reaches releaseFn
        if (b.slotOf(s)) |slot| b.owners[slot] = null;
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
        defer if (!handed) b.forget(s);
        b.direct = true;
        defer b.direct = false;
        try b.prefillSlot(s, 0);
        const pending = try b.draw(s, 0, b.first_logits[0], s.prompt_len);
        b.first_pick[0] = pending;
        var first_row: [1]lanes.LogprobRow = undefined;
        if (s.logprobs) |k| first_row[0] = logrow.row(b.first_logits[0] orelse return error.NoLogits, pending, k);
        _ = try s.commit(b.gpa, &.{pending}, if (s.logprobs != null) &first_row else &.{});
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
        defer if (!handed) b.forget(s);
        b.direct = true;
        defer b.direct = false;
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
                if (b.wrap) |w| try w.adopt(s); // the lane core drafts for it from its slot's context
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            r.taps_on = true;
            defer r.taps_on = false;
            var sampled: Sampled = .{ .b = b, .s = s.sampling orelse .{ .seed = 0 } };
            const drawer: ?xq.drafter.Draw = if (s.sampling) |sp| .{ .ctx = &sampled, .row = Sampled.row, .gumbel = Sampled.gumbel, .temperature = sp.temperature } else null;
            var logged: logrow.Logged = .{ .runner = b.runner, .k = s.logprobs orelse 0 };
            const note: ?xq.drafter.Note = if (s.logprobs != null) .{ .ctx = &logged, .f = logrow.Logged.take } else null;
            const round = try xq.drafter.treeRound(d, pending, b.block - 1, b.nodes, drawer, note, &out);
            s.rounds += 1;
            s.drafted += round.drafted;
            s.accepted += round.accepted;
            var landed: [17]u32 = undefined;
            const n = round.tokens.len - 1;
            @memcpy(landed[0..n], round.tokens[1..]);
            landed[n] = round.terminal;
            _ = try s.commit(b.gpa, landed[0 .. n + 1], if (s.logprobs != null) logged.rows[0 .. n + 1] else &.{});
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
