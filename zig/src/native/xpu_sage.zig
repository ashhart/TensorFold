//! Sage-A1.5 (qwen3_5_moe) on the lane host: a stream per runner slot, one batched decode step over the running slots.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const lane = @import("xpu_lane.zig");
const sage = xq.sage;
const vl = @import("core").qwen_vl;
const be = lanes.backend;
const Allocator = std.mem.Allocator;

pub const model_type = "qwen3_5_moe";

/// Drawn tokens a handle can still name (a prompt's first draw, then one a stream a round).
const ring = 64;

pub const Slots = struct {
    gpa: Allocator,
    r: *sage.Runner,
    owners: [sage.MAX_SLOTS]?*lanes.Stream = @splat(null),
    half: []f16, // a row of logits as the runner reads it
    row: []f32, // and widened for the host draw
    ids: []i32, // a prompt chunk as the runner takes it
    first_rows: [sage.MAX_SLOTS]?[]f32 = @splat(null), // a structured stream's prompt-row logits, for first_masked
    masked_vals: ?[]f64 = null, // a structured sampled row's allowed candidates
    masked_ids: ?[]u64 = null,
    fed: [sage.MAX_SLOTS]u32 = @splat(1), // rows the last verify fed each slot
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    vision: ?*Vision = null, // --vision
    depth: u32 = 1, // MTP drafts a lone round (XPU_SPEC; 1 keeps 80 % of drafts on chat text)
    held: [sage.MAX_SLOTS]u32 = @splat(0), // drafts a slot holds for its next window (specMulti makes them then)
    sampled_drafts: bool = true, // sampled streams take MTP drafts too (XPU_SAMPLED_DRAFTS=0: one row a round)
    draft_q: bool = true, // a sampled stream's first draft is drawn from the MTP's filtered row (XPU_SAMPLED_Q=0: its argmax)
    spec: ?*Spec = null, // speculative sampling's host buffers, made at the first sampled window

    fn of(ptr: *anyopaque) *Slots {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn init(gpa: Allocator, r: *sage.Runner) !Slots {
        const v = r.z.s.vocab;
        const half = try gpa.alloc(f16, v);
        errdefer gpa.free(half);
        const row = try gpa.alloc(f32, v);
        errdefer gpa.free(row);
        return .{ .gpa = gpa, .r = r, .half = half, .row = row, .ids = try gpa.alloc(i32, sage.CHUNK) };
    }

    pub fn deinit(b: *Slots) void {
        if (b.spec) |x| x.deinit(b.gpa);
        b.gpa.free(b.half);
        b.gpa.free(b.row);
        b.gpa.free(b.ids);
        for (b.first_rows) |x| if (x) |row| b.gpa.free(row);
        if (b.masked_vals) |x| b.gpa.free(x);
        if (b.masked_ids) |x| b.gpa.free(x);
    }

    pub fn backend(b: *Slots) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .held = heldFn, .release = releaseFn } };
    }

    /// With an MTP head: greedy windows of the pending row and `depth` held drafts, all in one specMulti forward.

    // Without an MTP head: one-row rounds and no drafts. Streams share one forward (stepMulti).

    // The lane core shares a forward only at width 2 or more. Copy drafts off keeps width 2, so each window is one row.

    // One row uses gdn_one and down_combine. Two to 16 rows use grouped MoE and gdn_pre, gdn_step, gated_norm.

    // A prompt above 16 rows uses tiled MoE and attn_prompt. Inside one kernel a row's K order ignores the batch.

    // streams_exact is false. One row and a shared row differ by 3509 bytes (gdn_one, down_combine).

    // A prompt piece of 8 rows and the first 8 of 17 differ by 24800 bytes (grouped vs tiled MoE).

    // Nine rows after those 8 and the last 9 of 17 differ by 28240 bytes.

    // Same rows in one width match. A repeat of 2, the first 2 of 16, and swapped streams differ by 0.

    pub fn facts(b: *const Slots) lanes.Model {
        if (b.r.mtp != null) return .{ .exact_width = b.depth + 1, .copy_drafts = false, .gpu_tokens = false, .mtp = true, .speculate = true, .speculate_early = false, .sampled_drafts = b.sampled_drafts, .drafts = b.depth, .draft_streams = true, .hidden_rows = true, .streams_exact = false, .batch_rows = sage.WINDOW, .max_streams = b.r.slots };
        return .{ .exact_width = 2, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = true, .streams_exact = false, .batch_rows = b.r.slots, .max_streams = b.r.slots };
    }

    fn push(b: *Slots, token: u32) u64 {
        b.drawn[b.next % ring] = token;
        b.next += 1;
        return b.next - 1;
    }

    fn slotOf(b: *const Slots, s: *const lanes.Stream) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }

    fn freeSlot(b: *const Slots) ?u32 {
        for (b.owners[0..b.r.slots], 0..) |o, i| if (o == null) return @intCast(i);
        return null;
    }

    /// The stream's next token: the runner's greedy one, else drawn on the host from logits row `row` (null: the

    // current slot's last forward) at `position`.

    fn draw(b: *Slots, s: *const lanes.Stream, greedy: i32, row: ?u32, position: u64) !u32 {
        const sp = s.sampling orelse return @intCast(greedy);
        if (sp.temperature <= 0) return @intCast(greedy);
        return lane.draw(b.gpa, try b.fetchRow(row), sp, position);
    }

    /// Row `row` of the last forward (null: the current slot's last row) as f32 logits in b.row.
    fn fetchRow(b: *Slots, row: ?u32) ![]f32 {
        if (row) |i| try b.r.rowLogits(i, b.half) else try b.r.logits(b.half);
        for (b.row, b.half) |*w, h| w.* = h;
        return b.row;
    }

    /// Structured output: the draw among the tokens `mask` allows, greedy (first argmax) or sampled (keyed).
    fn drawMasked(b: *Slots, s: *const lanes.Stream, row: []const f32, position: u64, mask: []const u32) !u32 {
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

    /// The prompt into a free slot in chunks (a cancel lands between them); its first token drawn at once.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        const ids = s.prompt();
        if (ids.len == 0 or ids.len + s.max_new + 1 > r.ctx) return error.PromptTooLong;
        const slot = b.slotOf(s) orelse (b.freeSlot() orelse return error.NoFreeSlot);
        b.owners[slot] = s;
        errdefer b.owners[slot] = null;
        try r.use(slot);
        try r.reset();
        const v: ?*Vision = if (s.images.len == 0) null else b.vision orelse return error.NoVision;
        const layout = if (v) |x| try x.encode(r, s) else vl.rope.Layout{};
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(sage.CHUNK, ids.len - at);
            for (b.ids[0..n], ids[at..][0..n]) |*d, t| d.* = @intCast(t);
            if (v) |x| try r.promptFeed(b.ids[0..n], x.feed(r, &layout, @intCast(at), @intCast(n))) else try r.prompt(b.ids[0..n]);
            at += n;
        }
        r.shift[slot] = layout.shift;
        if (s.grammar != null) {
            const keep = b.first_rows[slot] orelse try b.gpa.alloc(f32, b.row.len);
            b.first_rows[slot] = keep;
            @memcpy(keep, try b.fetchRow(null));
        }
        const first = try b.draw(s, r.last, null, ids.len);
        if (b.sampled_drafts and s.drafts and s.grammar == null and first != r.last) try r.refollow(@intCast(first));
        _ = b.push(first);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.push(try b.drawMasked(s, b.first_rows[slot] orelse return error.NoLogits, position, mask));
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return b.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const b = of(ptr);
        if (handle >= b.next or b.next - handle > ring) return error.NoSuchToken;
        return b.drawn[handle % ring];
    }

    /// Every stream's pending token in one forward: the single-row step for one stream, else stepMulti. A window of

    // two rows (a grammar's forced token after the pending one) runs alone, row by row: its second row is fed only

    // when the first row's draw equals it, so the slot holds exactly the rows a keep can take.

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        if (windows.len == 0 or windows.len > r.slots or windows.len != out.len) return error.BadWindows;
        var slots: [sage.MAX_SLOTS]u32 = undefined;
        var toks: [sage.MAX_SLOTS]i32 = undefined;
        var at: [sage.MAX_SLOTS]usize = undefined; // the batched windows' indices in `windows`
        var n: usize = 0;
        try b.verifyDrafted(windows, out);
        for (windows, 0..) |w, i| {
            if (w.held > 0) continue;
            if (w.rows() > 2 or w.parents != null) return error.WindowTooWide;
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            const pos = if (slot == r.cur) r.pos else r.spos[slot];
            if (w.positions[0] != pos + 1) return error.PositionMismatch;
            b.fed[slot] = 1;
            if (w.rows() == 2) continue;
            slots[n] = slot;
            toks[n] = @intCast(w.pending);
            at[n] = i;
            n += 1;
        }
        if (n == 1) {
            try r.use(slots[0]);
            try r.step(toks[0]);
            toks[0] = r.last;
        } else if (n > 1) try r.stepMulti(slots[0..n], toks[0..n], toks[0..n]);
        for (at[0..n], 0..) |wi, i| {
            const row: ?u32 = if (n == 1) null else @intCast(i);
            out[wi].sampled[0] = try b.drawRow(windows[wi], 0, toks[i], row);
        }
        for (windows, out) |w, o| {
            if (w.rows() != 2 or w.held > 0) continue;
            const slot = b.slotOf(w.stream).?;
            try r.use(slot);
            try r.step(@intCast(w.pending));
            o.sampled[0] = try b.drawRow(w, 0, r.last, null);
            o.sampled[1] = o.sampled[0]; // not read unless the forced token lands
            if (o.sampled[0] == w.tokens[0]) {
                try r.step(@intCast(w.tokens[0]));
                o.sampled[1] = try b.drawRow(w, 1, r.last, null);
                b.fed[slot] = 2;
            }
            @memcpy(o.drafts, w.tokens);
        }
    }

    /// The windows with held drafts, grouped by depth, each group one specMulti: it drafts, verifies and commits.

    // A window's rows report the kept tokens, and its drafts are the kept ones then one that differs, so the lane

    // core keeps exactly the rows specMulti committed.

    fn verifyDrafted(b: *Slots, windows: []const be.Window, out: []be.Verified) !void {
        const r = b.r;
        var depth: u32 = 1;
        while (depth < sage.WINDOW) : (depth += 1) {
            var slots: [sage.MAX_SLOTS]u32 = undefined;
            var at: [sage.MAX_SLOTS]usize = undefined;
            var n: usize = 0;
            for (windows, 0..) |w, i| if (w.held == depth) {
                if (w.parents != null or w.masks.len > 0 or w.tokens.len > 0) return error.WindowTooWide;
                const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
                if (b.held[slot] < depth) return error.NoHeldDrafts;
                if (w.positions[0] != r.spos[slot] + 1 and !(slot == r.cur and w.positions[0] == r.pos + 1)) return error.PositionMismatch;
                // specMulti feeds the slot's last token. A sampled pending token replaces the greedy one.
                if (slot == r.cur) r.last = @intCast(w.pending) else r.slast[slot] = @intCast(w.pending);
                slots[n] = slot;
                at[n] = i;
                n += 1;
            };
            if (n == 0) continue;
            var toks: [sage.MAX_SLOTS][sage.WINDOW]i32 = undefined;
            var kept: [sage.MAX_SLOTS]usize = undefined;
            var judge: ?sage.Runner.Judge = null;
            for (at[0..n]) |wi| if (sampledOf(windows[wi].stream) != null) {
                const x = b.spec orelse try Spec.init(b.gpa, r.z.s.vocab);
                b.spec = x;
                for (at[0..n], slots[0..n], 0..) |wj, slot, i| x.of[i] = .{ .s = windows[wj].stream, .slot = slot, .position = windows[wj].positions[0] };
                judge = .{ .ctx = b, .keep = keepSampled, .draft = draftSampled };
                break;
            };
            try r.specMulti(slots[0..n], depth, toks[0..n], kept[0..n], judge);
            for (slots[0..n], at[0..n], toks[0..n], kept[0..n]) |slot, wi, t, k| {
                const o = out[wi];
                b.held[slot] = 0;
                b.fed[slot] = @intCast(k);
                for (o.sampled, 0..) |*x, j| x.* = @intCast(t[@min(j, k - 1)]);
                for (o.drafts, 0..) |*x, j| x.* = if (j + 1 < k) @intCast(t[j]) else @intCast(t[k - 1] ^ 1);
            }
        }
    }

    /// Judge.keep: speculative sampling of a sampled slot's block. Row j's draft x (from q) is kept with chance

    // min(1, p(x) / q(x)), p and q both filtered by the stream's sampling. The first rejected row draws from

    // max(0, p - q) normalized and ends the block; with every draft kept, the last row draws from p. A draft from

    // the MTP's argmax has q one at x. The kept tokens then have the one-row path's distribution.

    fn keepSampled(ptr: *anyopaque, i: usize, row: u32, ids: []const i32, out: *[sage.WINDOW]i32) anyerror!?u32 {
        const b = of(ptr);
        const x = b.spec.?;
        const o = &x.of[i];
        const sp = sampledOf(o.s) orelse return null;
        const v = b.r.z.s.vocab;
        const rows: u32 = @intCast(ids.len);
        try b.r.logitRows(row, rows, x.rows[0 .. rows * v]);
        var j: u32 = 0;
        while (j + 1 < rows) : (j += 1) {
            const pos = o.position + j;
            const p = try x.dist(x.rows[j * v ..][0..v], sp, &x.p);
            const draft: u32 = @intCast(ids[j + 1]);
            const q: ?*const Dist = if (j == 0 and b.r.sdrawn[o.slot]) &x.q[o.slot] else null;
            const qx = if (q) |d| d.chance(draft) else 1.0;
            if (lanes.sampling.uniform(sp.seed, pos, salt_accept) * qx <= p.chance(draft)) {
                out[j] = @intCast(draft);
                continue;
            }
            out[j] = @intCast(p.residual(q, draft, lanes.sampling.uniform(sp.seed, pos, salt_residual)));
            o.position += j + 1;
            return j + 1;
        }
        const p = try x.dist(x.rows[j * v ..][0..v], sp, &x.p);
        out[j] = @intCast(p.pick(lanes.sampling.uniform(sp.seed, o.position + j, salt_bonus)));
        o.position += rows;
        return rows;
    }

    /// Judge.draft: a sampled slot's next first draft, drawn from the MTP row's filtered distribution (kept as its q).
    fn draftSampled(ptr: *anyopaque, i: usize, row: u32) anyerror!?i32 {
        const b = of(ptr);
        const x = b.spec.?;
        const o = &x.of[i];
        const sp = sampledOf(o.s) orelse return null;
        if (!b.draft_q) return null;
        const v = b.r.z.s.vocab;
        try b.r.logitRows(row, 1, x.rows[0..v]);
        const q = try x.dist(x.rows[0..v], sp, &x.q[o.slot]);
        return @intCast(q.pick(lanes.sampling.uniform(sp.seed, o.position, salt_draft)));
    }

    /// Row `j` of window `w` drawn: masked when the window has grammar masks, else greedy or sampled.
    fn drawRow(b: *Slots, w: be.Window, j: usize, greedy: i32, row: ?u32) !u32 {
        if (w.masks.len == 0) return b.draw(w.stream, greedy, row, w.positions[j]);
        const words = w.masks.len / w.rows();
        return b.drawMasked(w.stream, try b.fetchRow(row), w.positions[j], w.masks[j * words ..][0..words]);
    }

    /// A window keeps at most the rows its verify fed (one, or two when the forced token landed). A shorter path

    // ends the stream; a next verify of it would fail its position check.

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = of(ptr);
        for (windows, paths) |w, p| {
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            if (p.len == 0 or p.len > b.fed[slot]) return error.KeepMismatch;
        }
    }

    /// The MTP absorbed the kept rows in specMulti (or the prompt in prefill): note the drafts each stream's next

    // window holds. specMulti makes them with its forward.

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = of(ptr);
        if (b.r.mtp == null) return error.NoDraftHead;
        for (requests) |q| {
            const slot = b.slotOf(q.stream) orelse return error.UnknownStream;
            b.held[slot] = @min(q.depth, b.depth);
        }
    }

    /// A lone greedy stream on MTP drafts (Runner.specStep, `depth` a round) until it ends (false) or another stream

    // arrives (true). The lane core then decodes it beside the others, one row a round, with no drafts.

    fn loneFn(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const b = of(ptr);
        const r = b.r;
        try prefillFn(ptr, s);
        const slot = b.slotOf(s).?;
        var handed = false;
        defer if (!handed) {
            b.owners[slot] = null;
        };
        var pending = b.drawn[(b.next - 1) % ring];
        _ = try s.commit(b.gpa, &.{pending}, &.{});
        hooks.committed(hooks.ctx);
        var out: [sage.WINDOW]i32 = undefined;
        var landed: [sage.WINDOW]u32 = undefined;
        while (!s.finished) {
            if (hooks.yield(hooks.ctx)) {
                s.cache_len = r.pos;
                s.pending = pending;
                handed = true;
                return true;
            }
            if (s.isCancelled()) return error.Cancelled;
            const n = try r.specStep(b.depth, &out);
            for (landed[0..n], out[0..n]) |*d, t| d.* = @intCast(t);
            s.rounds += 1;
            s.drafted += b.depth;
            s.accepted += n - 1;
            _ = try s.commit(b.gpa, landed[0..n], &.{});
            pending = landed[n - 1];
            hooks.committed(hooks.ctx);
        }
        return false;
    }

    /// A grammar stream's held draft as a host token, for the grammar to walk: the slot's MTP draft. Its window then

    // runs on the two-row path of verifyFn, which feeds the draft only when the masked draw equals it.

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return error.UnknownStream;
        if (out.len != 1) return error.GrammarDraftsDeep; // the two-row path takes one draft
        out[0] = @intCast(@max(b.r.sdraft[slot], 0)); // none yet: any token, the draw decides
        b.held[slot] = 0;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
        b.held[slot] = 0;
    }
};

/// A sampled stream's sampling, or null for a greedy one.
fn sampledOf(s: *const lanes.Stream) ?lanes.Sampling {
    const sp = s.sampling orelse return null;
    return if (sp.temperature > 0) sp else null;
}

// Keys of the speculative sampler's uniforms at a position, apart from the one-row draw's token ids.
const salt_accept: u64 = 0xFFFF_0000_0000_0001;
const salt_residual: u64 = 0xFFFF_0000_0000_0002;
const salt_bonus: u64 = 0xFFFF_0000_0000_0003;
const salt_draft: u64 = 0xFFFF_0000_0000_0004;

/// A filtered next-token distribution: the candidates chooseOrdered keeps (top_k, top_p, min_p after temperature),

// in its order, with their softmax chances.

const Dist = struct {
    ids: []u32 = &.{},
    p: []f64 = &.{},
    n: usize = 0,

    fn chance(d: *const Dist, id: u32) f64 {
        for (d.ids[0..d.n], d.p[0..d.n]) |t, c| if (t == id) return c;
        return 0;
    }

    /// The candidate at uniform `u` of the cumulative chances.
    fn pick(d: *const Dist, u: f64) u32 {
        var cum: f64 = 0;
        for (d.ids[0..d.n], d.p[0..d.n]) |t, c| {
            cum += c;
            if (u <= cum) return t;
        }
        return d.ids[d.n - 1];
    }

    /// A draw from max(0, p - q) normalized, with q the draft's distribution (null: one at `draft`).
    fn residual(p: *const Dist, q: ?*const Dist, draft: u32, u: f64) u32 {
        var total: f64 = 0;
        for (p.ids[0..p.n], p.p[0..p.n]) |t, c| total += @max(0, c - qOf(q, draft, t));
        if (!(total > 0)) return p.pick(u);
        var cum: f64 = 0;
        var last: u32 = p.ids[0];
        for (p.ids[0..p.n], p.p[0..p.n]) |t, c| {
            const w = @max(0, c - qOf(q, draft, t));
            if (w <= 0) continue;
            cum += w / total;
            last = t;
            if (u <= cum) return t;
        }
        return last;
    }

    fn qOf(q: ?*const Dist, draft: u32, t: u32) f64 {
        if (q) |d| return d.chance(t);
        return if (t == draft) 1 else 0;
    }
};

/// The host side of speculative sampling: a block's logit rows, p's scratch, each slot's q.
const Spec = struct {
    rows: []f16,
    vals: []f64, // a row's top candidates, by value then the lower id
    order: []u32,
    p: Dist,
    q: [sage.MAX_SLOTS]Dist,
    of: [sage.MAX_SLOTS]struct { s: *const lanes.Stream, slot: u32, position: u64 } = undefined, // specMulti's slot i

    fn init(gpa: Allocator, vocab: usize) !*Spec {
        const x = try gpa.create(Spec);
        errdefer gpa.destroy(x);
        x.* = .{ .rows = try gpa.alloc(f16, sage.WINDOW * vocab), .vals = try gpa.alloc(f64, vocab), .order = try gpa.alloc(u32, vocab), .p = try newDist(gpa, vocab), .q = undefined };
        for (&x.q) |*d| d.* = try newDist(gpa, vocab);
        return x;
    }

    fn newDist(gpa: Allocator, vocab: usize) !Dist {
        return .{ .ids = try gpa.alloc(u32, vocab), .p = try gpa.alloc(f64, vocab) };
    }

    fn deinit(x: *Spec, gpa: Allocator) void {
        gpa.free(x.rows);
        gpa.free(x.vals);
        gpa.free(x.order);
        gpa.free(x.p.ids);
        gpa.free(x.p.p);
        for (x.q) |d| {
            gpa.free(d.ids);
            gpa.free(d.p);
        }
        gpa.destroy(x);
    }

    /// `row`'s distribution under `sp` into `d`, with chooseOrdered's candidates and cutoffs.
    fn dist(x: *Spec, row: []const f16, sp: lanes.Sampling, d: *Dist) !*const Dist {
        const width = row.len;
        const k: usize = if (sp.top_k == 0) width else @min(width, sp.top_k);
        var filled: usize = 0;
        if (k == width) {
            for (x.order[0..width], 0..) |*o, id| o.* = @intCast(id);
            const Ctx = struct {
                r: []const f16,
                fn lt(c: @This(), a: u32, b: u32) bool {
                    return c.r[a] > c.r[b] or (c.r[a] == c.r[b] and a < b);
                }
            };
            std.mem.sort(u32, x.order[0..width], Ctx{ .r = row }, Ctx.lt);
            for (x.vals[0..width], x.order[0..width]) |*v, id| v.* = row[id];
            filled = width;
        } else for (row, 0..) |h, id| { // insertion into the k best; an equal value stays behind the lower id
            const v: f64 = h;
            if (filled == k and v <= x.vals[k - 1]) continue;
            var at = @min(filled, k - 1);
            while (at > 0 and v > x.vals[at - 1]) : (at -= 1) {
                x.vals[at] = x.vals[at - 1];
                x.order[at] = x.order[at - 1];
            }
            x.vals[at] = v;
            x.order[at] = @intCast(id);
            filled = @min(filled + 1, k);
        }
        for (x.vals[0..filled]) |v| if (!std.math.isFinite(v)) return error.NonfiniteLogits;
        const t = @max(sp.temperature, 1e-6);
        const top = x.vals[0] / t;
        var keep = filled;
        if (sp.top_p > 0.0 and sp.top_p < 1.0) {
            var total: f64 = 0;
            for (x.vals[0..filled], d.p[0..filled]) |v, *c| {
                c.* = @exp(v / t - top);
                total += c.*;
            }
            var cum: f64 = 0;
            var below: usize = 0;
            for (d.p[0..filled]) |c| {
                cum += c / total;
                if (cum < sp.top_p) below += 1;
            }
            keep = @min(keep, below + 1);
        }
        if (sp.min_p > 0.0) {
            const floor = top + @log(sp.min_p);
            var n: usize = 0;
            for (x.vals[0..keep]) |v| {
                if (v / t >= floor) n += 1;
            }
            keep = n;
        }
        var total: f64 = 0;
        for (x.vals[0..keep], d.p[0..keep], d.ids[0..keep], x.order[0..keep]) |v, *c, *id, o| {
            c.* = @exp(v / t - top);
            total += c.*;
            id.* = o;
        }
        for (d.p[0..keep]) |*c| c.* /= total;
        d.n = keep;
        return d;
    }
};

/// --vision: the checkpoint's image tower, its processor, and the rows one prompt's images take.
const Vision = struct {
    io: std.Io,
    c: vl.Config,
    tower: *xq.vision.Tower,
    rows: *anyopaque, // fp32 [image_tokens, hidden]: a prompt's image rows, image after image
    image_tokens: u32,
    per_image: u32, // tokens one image takes at most: the tower's scratch
    markers: [3]u32, // vision_start, image_pad, vision_end: text that spells one stays text
    spans: [vl.rope.max_images]struct { at: u32, n: u32, row: u32 } = undefined,
    count: u32 = 0,
    rope: [sage.CHUNK][3]i32 = undefined,
    pieces: [vl.rope.max_images]sage.Rows = undefined,

    /// The tower of the checkpoint in `dir` on the runner's card; null with `why` set when it has none it reads.
    fn open(gpa: Allocator, io: std.Io, a: Allocator, r: *sage.Runner, dir: []const u8, o: api.VisionOpen, why: *[]const u8) !?*Vision {
        const read = struct {
            fn f(al: Allocator, i: std.Io, d: []const u8, name: []const u8) ?[]const u8 {
                const path = std.fs.path.join(al, &.{ d, name }) catch return null;
                return std.Io.Dir.cwd().readFileAlloc(i, path, al, .limited(16 << 20)) catch null;
            }
        }.f;
        const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
        const c = (vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json")) catch null) orelse {
            why.* = "--vision: this checkpoint names no image tower the XPU engine reads (config.json vision_config)";
            return null;
        };
        if (c.out_hidden != r.z.s.hidden or c.patch != xq.vision.patch or c.temporal != xq.vision.temporal or c.merge != xq.vision.merge or c.positions != xq.vision.grid_side * xq.vision.grid_side or !textRope(a, config)) {
            why.* = "--vision: the image tower or the text's M-RoPE (interleaved 11, 11, 10) is not the Qwen3.5 one the XPU engine runs";
            return null;
        }
        const per_image: u32 = @intCast(@min(@as(u64, o.image_tokens), c.max_pixels / (@as(u64, c.factor()) * c.factor())));
        const prefix = if (r.ckpt.has("vision_tower.patch_embed.proj.weight")) "vision_tower." else "model.visual.";
        const tower = xq.vision.Tower.load(&r.d, &r.ckpt, prefix, .{ .hidden = c.hidden, .heads = c.heads, .depth = c.depth, .intermediate = c.intermediate, .out_hidden = c.out_hidden }, per_image * c.merge * c.merge) catch |err| {
            why.* = try std.fmt.allocPrint(a, "--vision: the XPU engine reads the tower in bf16 and could not load this one: {s}", .{@errorName(err)});
            return null;
        };
        errdefer tower.deinit();
        const v = try gpa.create(Vision);
        v.* = .{ .io = io, .c = c, .tower = tower, .rows = try r.d.alloc_(@as(usize, o.image_tokens) * c.out_hidden * 4), .image_tokens = o.image_tokens, .per_image = per_image, .markers = .{ c.vision_start, c.image_token, c.vision_end } };
        return v;
    }

    fn deinit(v: *Vision, gpa: Allocator) void {
        v.tower.deinit();
        gpa.destroy(v);
    }

    /// Info.vision's prepare: an image's bytes to the tower's patches (core/qwen_vl geometry), on the HTTP thread.
    fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const v: *Vision = @ptrCast(@alignCast(ctx));
        const p = try vl.geometry.prepare(v.io, a, bytes, v.c, @min(max_tokens, v.per_image));
        return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
    }

    fn offer(v: *Vision, o: api.VisionOpen) api.Vision {
        // the template writes Qwen's vision_start and vision_end around the image token itself
        return .{ .ctx = v, .prepare = prepare, .image_token = v.c.image_token, .markers = &v.markers, .image_tokens = v.image_tokens, .max_images = @min(o.max_images, vl.rope.max_images) };
    }

    /// `s`'s images through the tower into `rows`, and its prompt's M-RoPE layout.
    fn encode(v: *Vision, r: *sage.Runner, s: *const lanes.Stream) !vl.rope.Layout {
        const layout = try vl.rope.Layout.init(s.prompt(), s.images, .{ .start = v.c.vision_start, .image = v.c.image_token, .end = v.c.vision_end }, v.c.merge);
        const D: usize = v.c.out_hidden;
        var row: u32 = 0;
        v.count = 0;
        for (s.images) |img| {
            if (row + img.tokens > v.image_tokens or img.tokens > v.per_image) return error.ImageTooLarge;
            try v.tower.encode(&r.d, img.pixels, img.gh, img.gw, off(v.rows, row * D * 4));
            v.spans[v.count] = .{ .at = img.at, .n = img.tokens, .row = row };
            v.count += 1;
            row += img.tokens;
        }
        return layout;
    }

    /// Prompt rows [at, at + n): their positions and the image rows among them.
    fn feed(v: *Vision, r: *const sage.Runner, layout: *const vl.rope.Layout, at: u32, n: u32) sage.Feed {
        const D: usize = r.z.s.hidden;
        for (v.rope[0..n], 0..) |*p, i| p.* = layout.position(at + @as(u32, @intCast(i)));
        var k: usize = 0;
        for (v.spans[0..v.count]) |sp| {
            const lo = @max(sp.at, at);
            const hi = @min(sp.at + sp.n, at + n);
            if (lo >= hi) continue;
            v.pieces[k] = .{ .at = lo - at, .n = hi - lo, .src = off(v.rows, (sp.row + lo - sp.at) * D * 4) };
            k += 1;
        }
        return .{ .rope = v.rope[0..n], .rows = v.pieces[0..k] };
    }
};

/// The text's rotary is Qwen3.5's interleaved M-RoPE over sections 11, 11 and 10 (attn_prep's ROPE).
fn textRope(a: Allocator, config_json: []const u8) bool {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, config_json, .{}) catch return false;
    if (doc != .object) return false;
    const tc = doc.object.get("text_config");
    const text = if (tc != null and tc.? == .object) tc.?.object else doc.object;
    const rope = text.get("rope_parameters") orelse return false;
    if (rope != .object) return false;
    const inter = rope.object.get("mrope_interleaved") orelse return false;
    const sec = rope.object.get("mrope_section") orelse return false;
    if (inter != .bool or !inter.bool or sec != .array or sec.array.items.len != 3) return false;
    for (sec.array.items, [_]i64{ 11, 11, 10 }) |x, want| if (x != .integer or x.integer != want) return false;
    return true;
}

const Served = struct {
    gpa: Allocator,
    back: Slots,
    cfg: lanes.Config,
    wall: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},

    fn close(ptr: *anyopaque) void {
        const h: *Served = @ptrCast(@alignCast(ptr));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        const r = h.back.r;
        if (h.back.vision) |v| v.deinit(h.gpa);
        h.back.deinit();
        r.deinit(); // the card's memory, after the engine has stopped
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

/// Sage from `o.dir`. MLX affine 4/64 loads as stored. AutoRound needs XPU_SAGE_AS_MLX.

// Slots are 1 to 8. int8 K/V needs XPU_KV_INT8.

pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context: u32 = std.math.cast(u32, o.context orelse 32768) orelse 0;
    if (context == 0) {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs a context of 1 to 2^32-1 tokens, not {d}", .{o.context.?});
        return null;
    }
    const slots: u32 = @min(@max(o.lanes, 1), sage.MAX_SLOTS);
    // --drafter: the AutoRound checkpoint whose MTP head drafts for a lone greedy stream
    const mtp = if (o.drafts) o.drafter else null;
    const r = sage.Runner.initMtp(o.dir, context, slots, mtp) catch |err| {
        if (err == error.StoredPrecision) {
            problem.* = try std.fmt.allocPrint(a, "this build has no kernel for stored fp16 GPTQ scales or a stored bf16 MTP fc; set XPU_SAGE_AS_MLX=1 to serve those tensors as MLX affine 4/64, with fp16 scales rounded to bf16 and bf16 embedding, head, and MTP fc quantized to 4/64", .{});
            return null;
        }
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not load {s} with {d} streams of {d} tokens: {s}", .{ o.dir, slots, context, @errorName(err) });
        return null;
    };
    errdefer r.deinit();
    const h = try gpa.create(Served);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .back = try Slots.init(gpa, r), .cfg = undefined, .wall = .{ .io = io }, .core = undefined, .host = undefined };
    errdefer h.back.deinit();
    if (o.vision) |vo| h.back.vision = try Vision.open(gpa, io, a, r, o.dir, vo, problem) orelse return null;
    errdefer if (h.back.vision) |v| v.deinit(gpa);
    if (std.c.getenv("XPU_SPEC")) |v| h.back.depth = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    if (std.c.getenv("XPU_SAMPLED_DRAFTS")) |v| h.back.sampled_drafts = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("XPU_SAMPLED_Q")) |v| h.back.draft_q = !std.mem.eql(u8, std.mem.span(v), "0");
    if (h.back.depth == 0 or h.back.depth + 1 > sage.WINDOW) {
        problem.* = try std.fmt.allocPrint(a, "XPU_SPEC: 1 to {d} MTP drafts a round, not {d}", .{ sage.WINDOW - 1, h.back.depth });
        return null;
    }
    if (std.c.getenv("XPU_KV_INT8") != null) r.cap = @intCast(context);
    if (!try fitsCard(a, r, slots, context, problem)) return null;
    sage.Runner.finishContext(r) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not set up {d} streams of {d} tokens: {s}", .{ slots, context, @errorName(err) });
        return null;
    };
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), if (r.mtp != null) h.back.depth + 1 else 2, if (r.mtp != null) h.back.depth else 0);
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.startup = try std.fmt.allocPrint(gpa, "Intel GPU engine: Sage (qwen3_5_moe), {d} stream(s) of {d} tokens, {s} K/V, sampling on the host{s}", .{ slots, context, if (r.cap > 0) "int8" else "bf16", if (r.mtp == null) "" else if (h.back.sampled_drafts) ", MTP drafts for greedy and sampled streams" else ", MTP drafts for greedy streams" });
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "sage-xpu", .lanes = slots, .context_window = context, .context_fitted = true, .startup = h.startup, .prefill_step = 0, .structures = true });
    h.host.explain = .{ .text = lane.explain };
    if (h.back.vision) |v| h.host.info_.vision = v.offer(o.vision.?);
    if (r.mtp != null) h.host.lone = .{ .ctx = &h.back, .run = Slots.loneFn };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Served.close, .ctx = h };
}

/// Card bytes kept free of the engine's buffers.
const card_reserve: u64 = 64 << 20;
/// Bytes Dev.allocated counts that the card does not hold once every slot is in use. Measured with drm-total-vram0

// after 8 streams at once: 34.60 GB counted and 34.12 GB held (--parallel 8 --context 133000), 32.91 and 32.42 GB

// (the same with the MTP head at --context 100000). Taken a little under the 0.49 GB seen.

const counted_not_held: u64 = 450_000_000;

/// Whether the card holds what the load allocated. A K/V page is written only when its slot first reaches it, so

// an overcommit loses the device then, not at load. Else `problem` names the largest context for these slots and

// the most slots for this context.

fn fitsCard(a: Allocator, r: *sage.Runner, slots: u32, context: u32, problem: *[]const u8) !bool {
    const mem = r.d.g.memory() catch |err| {
        std.log.warn("XPU: no Sysman memory size ({s}), so the load does not check that the card holds every slot's K/V", .{@errorName(err)});
        return true;
    };
    const room = mem.size -| card_reserve;
    const per: u64 = r.kvBytesPerToken();
    const state: u64 = r.slotStateBytes();
    const held: u64 = r.d.allocated -| counted_not_held;
    const acts = sage.actsBytes(r.z, context);
    const need: u64 = held + acts + slots * (state + per * context);
    std.log.info("XPU: the load holds {d:.2} GB of the card's {d:.2} GB ({d:.2} GB kept free)", .{ gb(need), gb(mem.size), gb(card_reserve) });
    if (need <= room) return true;
    const fixed = held + acts; // weights and the prompt buffers: what fewer slots leave
    const left = room -| fixed;
    const max_context = (left / slots -| state) / per;
    const max_slots = left / (state + per * context);
    problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs {d:.2} GB for {d} stream(s) of {d} tokens{s}, and the card has {d:.2} GB with {d:.2} GB kept free. It fits {d} stream(s) of at most {d} tokens, or at most {d} stream(s) of {d} tokens (--parallel, --context)", .{ gb(need), slots, context, if (r.mtp != null) " with the MTP head" else "", gb(mem.size), gb(card_reserve), slots, max_context, max_slots, context });
    return false;
}

fn gb(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}
