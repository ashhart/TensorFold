//! Sage-A1.5 (qwen3_5_moe) on the lane host: a stream per runner slot, one batched decode step over the running slots.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const xq = @import("xpu_qwen27");
const lane = @import("xpu_lane.zig");
const sage = xq.sage;
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
        b.gpa.free(b.half);
        b.gpa.free(b.row);
        b.gpa.free(b.ids);
        for (b.first_rows) |x| if (x) |row| b.gpa.free(row);
        if (b.masked_vals) |x| b.gpa.free(x);
        if (b.masked_ids) |x| b.gpa.free(x);
    }

    pub fn backend(b: *Slots) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .first_masked = firstMaskedFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    /// One-row rounds, no drafts; the running streams' rows share one forward (stepMulti). The lane core shares a
    /// forward only for a width of 2 or more, so the width is 2 with copy drafts off: every window is one row.
    /// stepMulti's grouped expert kernels add in another order than step, so a near-tie can pick another token.
    pub fn facts(b: *const Slots) lanes.Model {
        return .{ .exact_width = 2, .copy_drafts = false, .gpu_tokens = false, .mtp = false, .speculate = false, .speculate_early = false, .hidden_rows = true, .streams_exact = true, .batch_rows = b.r.slots, .max_streams = b.r.slots };
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
    /// current slot's last forward) at `position`.
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
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(sage.CHUNK, ids.len - at);
            for (b.ids[0..n], ids[at..][0..n]) |*d, t| d.* = @intCast(t);
            try r.prompt(b.ids[0..n]);
            at += n;
        }
        if (s.grammar != null) {
            const keep = b.first_rows[slot] orelse try b.gpa.alloc(f32, b.row.len);
            b.first_rows[slot] = keep;
            @memcpy(keep, try b.fetchRow(null));
        }
        _ = b.push(try b.draw(s, r.last, null, ids.len));
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
    /// two rows (a grammar's forced token after the pending one) runs alone, row by row: its second row is fed only
    /// when the first row's draw equals it, so the slot holds exactly the rows a keep can take.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = of(ptr);
        const r = b.r;
        if (windows.len == 0 or windows.len > r.slots or windows.len != out.len) return error.BadWindows;
        var slots: [sage.MAX_SLOTS]u32 = undefined;
        var toks: [sage.MAX_SLOTS]i32 = undefined;
        var at: [sage.MAX_SLOTS]usize = undefined; // the batched windows' indices in `windows`
        var n: usize = 0;
        for (windows, 0..) |w, i| {
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
            if (w.rows() != 2) continue;
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

    /// Row `j` of window `w` drawn: masked when the window has grammar masks, else greedy or sampled.
    fn drawRow(b: *Slots, w: be.Window, j: usize, greedy: i32, row: ?u32) !u32 {
        if (w.masks.len == 0) return b.draw(w.stream, greedy, row, w.positions[j]);
        const words = w.masks.len / w.rows();
        return b.drawMasked(w.stream, try b.fetchRow(row), w.positions[j], w.masks[j * words ..][0..words]);
    }

    /// A window keeps at most the rows its verify fed (one, or two when the forced token landed). A shorter path
    /// ends the stream; a next verify of it would fail its position check.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = of(ptr);
        for (windows, paths) |w, p| {
            const slot = b.slotOf(w.stream) orelse return error.UnknownStream;
            if (p.len == 0 or p.len > b.fed[slot]) return error.KeepMismatch;
        }
    }

    fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {
        return error.NoDraftHead;
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = of(ptr);
        const slot = b.slotOf(s) orelse return;
        b.owners[slot] = null;
    }
};

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
        h.back.deinit();
        r.deinit(); // the card's memory, after the engine has stopped
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }
};

/// Sage from `o.dir` (MLX affine 4/64): --parallel slots (1 to 8) of --context tokens each, int8 K/V with XPU_KV_INT8.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const context: u32 = std.math.cast(u32, o.context orelse 32768) orelse 0;
    if (context == 0) {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine needs a context of 1 to 2^32-1 tokens, not {d}", .{o.context.?});
        return null;
    }
    const slots: u32 = @min(@max(o.lanes, 1), sage.MAX_SLOTS);
    const r = sage.Runner.init(o.dir, context, slots) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the XPU Sage engine could not load {s} with {d} streams of {d} tokens: {s}", .{ o.dir, slots, context, @errorName(err) });
        return null;
    };
    errdefer r.deinit();
    const h = try gpa.create(Served);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .back = try Slots.init(gpa, r), .cfg = undefined, .wall = .{ .io = io }, .core = undefined, .host = undefined };
    errdefer h.back.deinit();
    h.cfg = try lanes.Config.init(gpa, h.back.facts(), 2, 0);
    errdefer h.cfg.deinit(gpa);
    h.core = lanes.Engine.init(gpa, &h.cfg, h.back.backend(), h.wall.clock());
    errdefer h.core.deinit();
    h.startup = try std.fmt.allocPrint(gpa, "Intel GPU engine: Sage (qwen3_5_moe), {d} stream(s) of {d} tokens, {s} K/V, sampling on the host", .{ slots, context, if (r.cap > 0) "int8" else "bf16" });
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .name = "sage-xpu", .lanes = slots, .context_window = context, .context_fitted = true, .startup = h.startup, .prefill_step = 0, .structures = true });
    h.host.explain = .{ .text = lane.explain };
    try h.host.start();
    return .{ .engine = h.host.engine(), .close = Served.close, .ctx = h };
}
