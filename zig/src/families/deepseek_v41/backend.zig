//! V4.1 lane slots: a stream a slot, greedy, drafts off; a window is a slot's sub-forward, so rows keep solo bits.
const std = @import("std");
const ds = @import("model.zig");
const kv = @import("kv_state.zig");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const greedy = @import("greedy.zig");

const RING = 1024;

/// The verify's greedy sample. TF_DSV41_FAST_ARGMAX=0 is the scalar control; unset is the fast form.
var fast_argmax_cache: ?bool = null;
fn fastArgmax() bool {
    if (fast_argmax_cache) |v| return v;
    const s = std.c.getenv("TF_DSV41_FAST_ARGMAX");
    fast_argmax_cache = if (s) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    return fast_argmax_cache.?;
}

/// The tokens that followed the last earlier occurrence of `s`'s own tail, at most `out.len` of them.
fn tokAt(p: []const u32, h: []const u32, k: usize) u32 {
    return if (k < p.len) p[k] else h[k - p.len];
}

/// The tokens that followed the last earlier occurrence of the stream's own tail, at most `out.len` of them. The
/// space searched is the prompt followed by the tokens this slot has committed since.
fn lookupTokens(p: []const u32, h: []const u32, out: []u32) usize {
    const total = p.len + h.len;
    if (total < 4 or out.len == 0) return 0;
    var n: usize = @min(total - 1, 12);
    while (n >= 2) : (n -= 1) {
        const tail = total - n;
        var i: usize = 0;
        while (i + n < total) : (i += 1) {
            var j: usize = 0;
            while (j < n and tokAt(p, h, i + j) == tokAt(p, h, tail + j)) j += 1;
            if (j == n) {
                const start = i + n;
                const avail = @min(out.len, total - start);
                for (out[0..avail], 0..) |*t, k| t.* = tokAt(p, h, start + k);
                return avail;
            }
        }
    }
    return 0;
}

/// The hand-off's cap (the drafts a round); the default is the shipped three.
fn mtpDepthMax() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_DEPTH_MAX") orelse return 3;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3;
    return @min(@max(n, 1), 5);
}

var lookup_depth: ?u32 = null;
/// TF_DSV41_LOOKUP_DRAFTS=<depth>: draft that many tokens a round from the stream's own suffix matches.
fn lookupDepth() u32 {
    if (lookup_depth) |d| return d;
    var out: u32 = 0;
    if (std.c.getenv("TF_DSV41_LOOKUP_DRAFTS")) |v| {
        std.log.info("deepseek-v41: lookupDepth reads TF_DSV41_LOOKUP_DRAFTS={s}", .{std.mem.span(v)});
        out = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
        if (out > 7) out = 7;
    }
    lookup_depth = out;
    return out;
}

var no_refeed: ?bool = null;
/// TF_DSV41_NO_REFEED=0 restores the re-feed; the no-re-feed port is the default. A rejected window's kept rows are
/// the rows the verify already wrote, so the rollback restores the cache from the pool's capture instead of running
/// the model a second time -- and the re-feed's own divergence from the verify (round 252: one reject in 22) goes
/// with it.
fn noRefeedOn() bool {
    if (no_refeed) |v| return v;
    const raw = std.c.getenv("TF_DSV41_NO_REFEED") orelse {
        no_refeed = true;
        return true;
    };
    no_refeed = !std.mem.eql(u8, std.mem.span(raw), "0");
    return no_refeed.?;
}

var draft_rows_follow: ?bool = null;
/// TF_DSV41_MTP_DRAFT_ROWS=0 restores the old form (the drafter always proposes three, four rows); the default
/// sizes the proposer's own forward to the window the round will verify.
fn draftRowsFollowWindow() bool {
    if (draft_rows_follow) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_DRAFT_ROWS") orelse {
        draft_rows_follow = true;
        return true;
    };
    draft_rows_follow = !std.mem.eql(u8, std.mem.span(raw), "0");
    return draft_rows_follow.?;
}

var round_signal: ?u8 = null; // 0 = first-draft (shipped), 1 = round, 2 = hybrid
/// TF_DSV41_MTP_ADAPT_SIGNAL=round feeds the policy the round's own per-draft acceptance (accepted / offered)
/// instead of the first draft's hit. Default off (the shipped first-draft signal).
fn roundSignalMode() u8 {
    if (round_signal) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_SIGNAL") orelse {
        round_signal = 0;
        return 0;
    };
    round_signal = if (std.mem.eql(u8, std.mem.span(raw), "round"))
        1
    else if (std.mem.eql(u8, std.mem.span(raw), "hybrid"))
        2
    else
        0;
    return round_signal.?;
}

fn roundSignalOn() bool {
    return roundSignalMode() != 0;
}

var mtp_confirm: ?u32 = null;
/// TF_DSV41_MTP_ADAPT_CONFIRM=k requires the up-condition for k consecutive rounds before the deep window is taken.
/// Named a confirmed switch-up as the policy's next item and it was never built: the signal is an EMA of a
/// Bernoulli(0.62) on novel text, so a single spike above UP fires the deep window, and on the novel band those
/// rounds accepted not one extra draft (179 accepted at depth 1 and at the policy's mix alike). The default 1 is
/// the shipped one-sample form.
fn mtpConfirmRounds() u32 {
    if (mtp_confirm) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_CONFIRM") orelse {
        mtp_confirm = 1;
        return 1;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 1;
    mtp_confirm = @min(@max(n, 1), 8);
    return mtp_confirm.?;
}

/// The shipped band's first-draft signal is depth-blind (at depth 3 the first draft usually lands even when drafts
/// 2-3 do not), so on a long novel turn it re-enters the deep window round after round and pays 0.49 acceptance for
/// it. The round signal the policy already computes (`ratio`, kept/offered one round lagged) is the depth-aware one
/// named; this is the form that uses it as a one-way ratchet rather than as an EMA. `=0` restores the shipped band.
/// It is a POLICY (which window to verify), not a numerics change -- every depth is token-identical to the plain
/// path.
var ratchet_cache: ?f64 = null;
fn mtpRatchetDown() f64 {
    if (ratchet_cache) |v| return v;
    const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_RATCHET") orelse {
        ratchet_cache = 0.90;
        return 0.90;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    ratchet_cache = if (t > 0) t else 0;
    return ratchet_cache.?;
}

var draft_off_on: ?bool = null;
/// THE DRAFTING GOVERNOR. A drafted round's break-even is (the verify's window + the drafter's own forward) / (the
/// plain token). That is NOT a constant: 1.72 at short context and 1.80 at 22k, because the two-row verify grows
/// with the keys while the plain token grows less. So the governor measures the request's OWN yield (the tokens the
/// rounds have committed, over the rounds) and takes the request off the drafted path -- no MTP draft, no lookup
/// draft, a plain one-row round -- once it is below the break-even. It is a POLICY: every drafted depth is
/// token-identical to the plain path, so a request that switches mid-turn reads the same stream.
/// `TF_DSV41_DRAFT_OFF=0` is the same-binary control; `TF_DSV41_DRAFT_OFF_BELOW` is the break-even (default 1.75);
/// `TF_DSV41_DRAFT_OFF_AFTER` the rounds the estimate needs (default 24: a request shorter than this is left alone,
/// because its opening is the least predictable part of any text and the loss is bounded by its length).
fn draftOffArmed() bool {
    if (draft_off_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF") orelse {
        draft_off_on = true;
        return true;
    };
    draft_off_on = !std.mem.eql(u8, std.mem.span(raw), "0");
    return draft_off_on.?;
}

var draft_off_below: ?f64 = null;
fn draftOffBelow() f64 {
    if (draft_off_below) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_BELOW") orelse {
        draft_off_below = 1.75;
        return 1.75;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    draft_off_below = @max(t, 0);
    return draft_off_below.?;
}

var draft_off_after: ?u32 = null;
fn draftOffAfter() u32 {
    if (draft_off_after) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_AFTER") orelse {
        draft_off_after = 24;
        return 24;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 24;
    draft_off_after = @max(n, 2);
    return draft_off_after.?;
}

var draft_off_gate_ctx: ?f64 = null;
/// THE CONTEXT-GATED GOVERNOR SAMPLE. The gate takes the short sample only while the request's own CONTEXT is below
/// this threshold and the flat 24 (opening-safe) sample at or above it. It reads the cache cursor, not the prompt
/// length: a resumed turn's prompt is a short suffix (a 41k conversation's would look short and take the losing
/// arm). `TF_DSV41_DRAFT_OFF_GATE_CTX=0` disables the gate (the flat 24, The shipped policy).
fn draftOffGateCtx() f64 {
    if (draft_off_gate_ctx) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_GATE_CTX") orelse {
        draft_off_gate_ctx = 32768;
        return 32768;
    };
    const t = std.fmt.parseFloat(f64, std.mem.span(raw)) catch 0;
    draft_off_gate_ctx = @max(t, 0);
    return draft_off_gate_ctx.?;
}

var draft_off_low: ?u32 = null;
/// The short-context sample the gate arms below `draftOffGateCtx` (default 8: The chat arm, same-binary A/B against
/// the 24 default). `TF_DSV41_DRAFT_OFF_SAMPLE_LOW` moves it.
fn draftOffLow() u32 {
    if (draft_off_low) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DRAFT_OFF_SAMPLE_LOW") orelse {
        draft_off_low = 8;
        return 8;
    };
    const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 8;
    draft_off_low = @max(n, 2);
    return draft_off_low.?;
}

/// The sample this request's governor needs.  An explicit `TF_DSV41_DRAFT_OFF_AFTER` is a flat override
/// (the same-binary A/B knob); otherwise the context gate decides, and a disabled gate is the flat 24.
fn draftOffAfterFor(b: *Backend, i: usize) u32 {
    if (std.c.getenv("TF_DSV41_DRAFT_OFF_AFTER") != null) return draftOffAfter();
    const gate = draftOffGateCtx();
    if (gate <= 0 or i >= b.model.caches.len) return 24;
    return if (@as(f64, @floatFromInt(b.model.caches[i].position)) < gate) draftOffLow() else 24;
}

const MtpPolicy = struct { on: bool, up: f64, down: f64 };
var mtp_policy: ?MtpPolicy = null;
/// THE ADAPTIVE DEPTH POLICY, FACTORED OUT OF `tree` SO THE DRAFTER CAN SIZE ITSELF TO IT. The early `draft` hook
/// runs before `tree` and proposed three drafts (four rows) whatever the window; on the novel band 240 of 371 asks
/// were depth 1 (a two-row window), so two of the four rows the drafter computed were never verified. This reads
/// `last_depth` exactly as `tree` does and writes nothing: the policy's `prev` is the depth the last round ran at,
/// and only `tree` advances it.
fn mtpDepthFor(b: *Backend, slot: u32) u32 {
    const p = mtp_policy orelse blk: {
        const raw = std.c.getenv("TF_DSV41_MTP_ADAPT");
        const single: ?f64 = if (raw) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch null) else null;
        const up: f64 = if (single) |t| t else (std.fmt.parseFloat(f64, std.mem.span(std.c.getenv("TF_DSV41_MTP_ADAPT_UP") orelse "0.7")) catch 0.7);
        const down: f64 = if (single) |t| t else (std.fmt.parseFloat(f64, std.mem.span(std.c.getenv("TF_DSV41_MTP_ADAPT_DOWN") orelse "0.7")) catch 0.7);
        const off = raw != null and std.mem.eql(u8, std.mem.span(raw.?), "0");
        const v = MtpPolicy{ .on = !off, .up = up, .down = down };
        mtp_policy = v;
        break :blk v;
    };
    if (!p.on) return lookupDepth();
    const mode = roundSignalMode();
    const up_rate = if (mode == 1) b.model.mtpRoundSlot(slot) else b.model.mtpAcceptSlot(slot);
    const stay_rate = if (mode >= 1) b.model.mtpRoundSlot(slot) else up_rate;
    const prev: u32 = if (slot < b.last_depth.len) b.last_depth[slot] else 0;
    if (mtpRatchetDown() > 0 and slot < b.shallow.len and b.shallow[slot]) return 1;
    return if (up_rate > p.up)
        lookupDepth()
    else if (prev >= 2 and stay_rate >= p.down)
        lookupDepth()
    else
        1;
}

fn draftGovernorOff(b: *Backend, i: usize) bool {
    if (!draftOffArmed()) return false;
    if (i >= b.gov_off.len) return false;
    if (b.gov_off[i]) return true;
    const rounds = b.gov_rounds[i];
    const sample = draftOffAfterFor(b, i);
    if (rounds < sample) return false;
    const yield_per_round = @as(f64, @floatFromInt(b.gov_committed[i])) / @as(f64, @floatFromInt(rounds));
    const first = b.model.mtpAcceptSlot(@intCast(i));
    if (yield_per_round < draftOffBelow() and first < 0.9) {
        b.gov_off[i] = true;
        std.log.info("dsv41: drafting off at round {d}: {d:.3} tokens a round, first-draft {d:.3}, break-even {d:.2}, context {d} sample {d}", .{ rounds, yield_per_round, first, draftOffBelow(), b.model.caches[i].position, sample });
        return true;
    }
    return false;
}

/// ROUND 529: the proposer's hand-off is a per-SLOT fact (the pending token and the lane's own
/// conditioning); a single entry let a four-lane round reuse another lane's drafts.
const MtpCache = struct { pending: u32, drafts: [4]u32, count: u32 = 3 };

pub const Backend = struct {
    gpa: std.mem.Allocator,
    model: *ds.Model,
    by: std.AutoHashMapUnmanaged(*Stream, u32) = .empty,
    /// Drawn tokens by handle: every read waits for its forward anyway (one sub-forward a window).
    ring: [RING]u32 = undefined,
    puts: u64 = 0,
    first: []u32,
    /// Each slot's cache cursor before its in-flight verify, for the rollback-and-refeed keep.
    verify_len: []u32,
    /// The drafts the lookup proposes for each slot's next window (host tokens, verified like GPU-held ones).
    held: [][]u32,
    /// The trained proposer's last proposal, so the round runs the draft forward ONCE. The lane core calls both
    /// draft and tree a round, and each of them ran proposeTokens -- the draft's own forward (its three blocks, the
    /// conditioning and the head) twice, which is a quarter of the round's cost for one set of drafts. The cache is
    /// keyed by the pending token the proposal was made for, so a different pending still runs the forward.
    mtp_cache: [32]?MtpCache = @splat(null),
    /// Each slot's committed tokens after the prompt, for the lookup drafter's matches.
    history: []std.ArrayListUnmanaged(u32),
    /// What the last verify sampled per row, so a rejected window's re-feed can be compared with it.
    last_sampled: [8]u32 = @splat(0),
    /// The depth each slot's last round RAN -- the adaptive policy's state (the yield's scale depends on the depth
    /// the round ran at, so the policy must know it).
    last_depth: []u32,
    /// Consecutive rounds the up-condition has held, for the confirmed switch-up.
    up_streak: []u32,
    /// The last window's kept/offered, for the round-acceptance signal. `keep_pending` says a partial keep happened
    /// since the last verify; a full window (no keep call) means 1.0.
    keep_pending: []bool,
    keep_ratio: []f64,
    /// The ratchet's per-slot state -- the request has fallen back to the shallow window, and the consecutive deep
    /// rounds below the crossover that put it there.
    shallow: []bool,
    low_streak: []u32,
    /// The drafting governor's per-slot state -- the rounds this request has completed, the tokens they committed,
    /// the window the last round ran, what its keep took, and whether drafting is now off.
    gov_rounds: []u32,
    gov_committed: []u32,
    gov_rows: []u32,
    gov_kept: []u32,
    gov_off: []bool,
    /// The current window's presence/frequency counts (the reply before it plus the row's path), and one row's
    /// scratch copy because the model's logits slice is const.
    pen_counts: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    pen_row: []f32 = &.{},

    pub fn init(gpa: std.mem.Allocator, model: *ds.Model) !Backend {
        const held = try gpa.alloc([]u32, model.lanes);
        for (held) |*h| h.* = &.{};
        const history = try gpa.alloc(std.ArrayListUnmanaged(u32), model.lanes);
        for (history) |*h| h.* = .empty;
        const last_depth = try gpa.alloc(u32, model.lanes);
        @memset(last_depth, 0);
        const up_streak = try gpa.alloc(u32, model.lanes);
        @memset(up_streak, 0);
        const keep_pending = try gpa.alloc(bool, model.lanes);
        @memset(keep_pending, false);
        const keep_ratio = try gpa.alloc(f64, model.lanes);
        @memset(keep_ratio, 1.0);
        const shallow = try gpa.alloc(bool, model.lanes);
        @memset(shallow, false);
        const low_streak = try gpa.alloc(u32, model.lanes);
        @memset(low_streak, 0);
        const gov_rounds = try gpa.alloc(u32, model.lanes);
        @memset(gov_rounds, 0);
        const gov_committed = try gpa.alloc(u32, model.lanes);
        @memset(gov_committed, 0);
        const gov_rows = try gpa.alloc(u32, model.lanes);
        @memset(gov_rows, 0);
        const gov_kept = try gpa.alloc(u32, model.lanes);
        @memset(gov_kept, 0);
        const gov_off = try gpa.alloc(bool, model.lanes);
        @memset(gov_off, false);
        return .{
            .last_depth = last_depth,
            .up_streak = up_streak,
            .keep_pending = keep_pending,
            .keep_ratio = keep_ratio,
            .shallow = shallow,
            .low_streak = low_streak,
            .gov_rounds = gov_rounds,
            .gov_committed = gov_committed,
            .gov_rows = gov_rows,
            .gov_kept = gov_kept,
            .gov_off = gov_off,
            .gpa = gpa,
            .model = model,
            .first = try gpa.alloc(u32, model.lanes),
            .verify_len = try gpa.alloc(u32, model.lanes),
            .held = held,
            .history = history,
        };
    }

    pub fn deinit(b: *Backend) void {
        b.by.deinit(b.gpa);
        b.gpa.free(b.first);
        b.gpa.free(b.verify_len);
        for (b.held) |h| if (h.len > 0) b.gpa.free(h);
        b.gpa.free(b.held);
        for (b.history) |*h| h.deinit(b.gpa);
        b.gpa.free(b.history);
        b.gpa.free(b.up_streak);
        b.gpa.free(b.keep_pending);
        b.gpa.free(b.keep_ratio);
        b.gpa.free(b.shallow);
        b.gpa.free(b.low_streak);
        b.gpa.free(b.gov_rounds);
        b.gpa.free(b.gov_committed);
        b.gpa.free(b.gov_rows);
        b.gpa.free(b.gov_kept);
        b.gpa.free(b.gov_off);
        b.pen_counts.deinit(b.gpa);
        if (b.pen_row.len > 0) b.gpa.free(b.pen_row);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .penalties = true, .vtable = &.{
            .prefill = prefill,
            .first = firstFn,
            .queue = queue,
            .read = read,
            .verify = verify,
            .keep = keep,
            .draft = draft,
            .tree = tree,
            .release = release,
        } };
    }

    /// A window of any width runs exactly as its slot's own forward; the lane core may bundle streams freely.
    pub fn facts(b: *const Backend) lanes.Model {
        const rows = b.model.workspace.plan.rows;
        return .{
            // The widest window whose rows reproduce one-token steps: the split dispatch's row invariant
            // covers rows <= 8 (a wider window switches the dense calls to the matmul path and its own k
            // order). Claiming the workspace's width here let the planner build windows it cannot verify
            // exactly -- and at depth >= 10 the draft table segfaults -- so the model states what it can do.
            .exact_width = @min(rows, 8),
            .first_copy_rows = 1,
            .mtp = lookupDepth() != 0,
            .speculate = lookupDepth() != 0,
            .speculate_early = false,
            .gpu_tokens = false,
            .plain_guard = true,
            .drafts = @max(lookupDepth(), 1),
            .window_costs = &.{},
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = rows,
            .max_streams = @intCast(b.model.lanes),
            .draft_streams = false,
        };
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
        var used: [256]bool = @splat(false);
        var it = b.by.valueIterator();
        while (it.next()) |v| used[v.*] = true;
        for (used, 0..) |u, i| if (b.model.lanes > i and !u) return @intCast(i);
        return null;
    }

    /// The prefill's first draw and the queue's next (the queue is unreachable for this family: gpu_tokens is
    /// false, so the lanes never pipeline). A sampled stream samples HERE too -- the first token used to be the
    /// greedy argmax under any temperature, which is the one draw the owner's echo-prone long system prompts could
    /// seed a degenerate turn with. The key is the drawn token's absolute position (drawFirst passes prompt_len;
    /// queueNext passes the new cache length), the same key verify's rows use.
    fn draw(b: *Backend, slot: u32, s: *Stream, position: u64) !u32 {
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info("draw: slot {d} pos {d}", .{ slot, b.model.caches[slot].position });
        const row = b.model.workspace.logits(1);
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const first = try sampleRow(row, sp, position);
            if (std.c.getenv("TF_DSV41_PEN_DEBUG") != null) std.log.info("first draw: sampled {d} at {d}", .{ first, position });
            return first;
        };
        var best: u32 = 0;
        var maximum = -std.math.inf(f32);
        for (row, 0..) |value, id| {
            if (!std.math.isFinite(value)) return error.NonfiniteLogits;
            if (value > maximum) {
                maximum = value;
                best = @intCast(id);
            }
        }
        return best;
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const b = self(ptr);
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        const i = b.free() orelse return error.NoFreeSlot;
        const state = &b.model.caches[i];
        if (prompt.len + s.max_new + 1 > state.plan.capacity) return error.ContextFull;
        try b.by.put(b.gpa, s, i);
        try b.model.resetSlot(i);
        b.history[i].clearRetainingCapacity();
        s.cached = 0;
        var at: u32 = 0;
        // A KEPT PROMPT STATE RESUMES THE PASS WHERE IT STOOD. The store hands the family's copy of the state after
        // `s.reuse.at` tokens; restoring it (the whole cache, its cursor, the Engram history and the taps) and
        // starting the loop there is the same state a fresh pass would stand on, so `resumed` equals `fresh` token
        // for token. A rejected copy falls back to the fresh pass, which is what `reuse_failed` says.
        if (s.reuse.saved) |saved| {
            const snap: *ds.Model.Snapshot = @ptrCast(@alignCast(saved));
            const want = s.reuse.at;
            if (want > 0 and want < prompt.len) blk: {
                b.model.snapshotRestore(i, snap) catch |err| {
                    s.reuse_failed = true;
                    s.reuse_error = err;
                    break :blk;
                };
                at = want;
                s.cached = at;
                b.history[i].appendSlice(b.gpa, prompt[0..at]) catch {};
            }
        }
        const rows = b.model.workspace.plan.rows;
        // THE TAIL MARK IS SNAPSHOTTED INSIDE THE CHUNK THAT RUNS PAST IT, not cut into a second forward. The mark
        // is a state, not a barrier: the merged chunk runs [at, end) at the same tensor-tile class, and the model
        // keeps the window ring, the taps ring and the four-token Engram history at the mark for the copy
        // (beginSnapTail). A tail in another class (the store leaves snap_at 0 there) keeps today's cut, as does an
        // env control.
        const merge_cold = std.c.getenv("TF_DSV41_SNAP_COLD") == null;
        // TF_DSV41_SNAP_TOGGLE + the file /tmp/r503-nomerge lets ONE process run both arms (the box's one serve at
        // a time would otherwise need two boots to compare states).
        const toggled_off = std.c.getenv("TF_DSV41_SNAP_TOGGLE") != null and std.c.access("/tmp/r503-nomerge", 0) == 0;
        const snap_at: u32 = if (std.c.getenv("TF_DSV41_NO_TAIL_SNAP") == null and !toggled_off and b.model.snapTailAllowed() and (merge_cold or s.reuse.at > 0)) s.reuse.snap_at else 0;
        while (at < prompt.len) {
            if (s.isCancelled()) return error.Cancelled;
            var end = @min(at + rows, @as(u32, @intCast(prompt.len)));
            // The store's marks always end a chunk: the pass stands exactly there when it calls the hook.
            for (s.reuse.marks) |m| if (m > at and m < end and m != snap_at) {
                end = m;
                break;
            };
            const last = end == prompt.len;
            var merge: u32 = if (snap_at > at and snap_at < end and end - at >= 64) snap_at else 0;
            if (merge != 0 and !b.model.beginSnapTail(i, prompt[@max(0, @as(usize, merge) - 4)..merge], merge, at)) merge = 0;
            if (last) {
                _ = try b.model.forwardOn(i, prompt[at..end], .last);
                b.first[i] = try b.draw(i, s, @intCast(prompt.len));
            } else {
                _ = try b.model.forwardOn(i, prompt[at..end], .none);
            }
            if (merge != 0) {
                // the state at the mark, copied from the stash the chunk's own publishes wrote
                if (s.reuse.hook) |hook| hook.at(hook.ptr, s, merge);
                b.model.endSnapTail();
            }
            at = end;
            // The prompt cache's keep hook, at the chunk end it asked to see. The forward committed and waited, so
            // the state the family copies is exactly the one standing at `at`.
            if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, at) != null) hook.at(hook.ptr, s, at);
        }
    }

    /// The prompt cache's Snapshots functions over the resident model (`ptr`: the *Backend, `owner`: the stream).
    pub fn snapBytes(ptr: *anyopaque, _: u32) u64 {
        return self(ptr).model.snapshotBytes();
    }

    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, _: u32) anyerror!*anyopaque {
        const b = self(ptr);
        const s: *Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const i = try b.slotOf(s);
        return @ptrCast(try b.model.snapshotSave(i));
    }

    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        self(ptr).model.snapshotDrop(@ptrCast(@alignCast(saved)));
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const i = try b.slotOf(s);
        if (position != b.model.caches[i].position) return error.PositionMismatch;
        return b.put(b.first[i]);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
            const v: u32 = switch (feed) {
                .value => |t| t,
                else => 0xFFFF_FFFF,
            };
            std.log.info("queue feed: {d}", .{v});
        }
        const b = self(ptr);
        const i = try b.slotOf(s);
        const state = &b.model.caches[i];
        if (position != state.position) return error.PositionMismatch;
        const token: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| try read(ptr, h),
        };
        _ = try b.model.forwardOn(i, &.{token}, .last);
        try b.history[i].append(b.gpa, token);
        return b.put(try b.draw(i, s, position));
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
            const v = self(ptr).ring[handle % RING];
            std.log.info("read: token {d}", .{v});
        }
        const b = self(ptr);
        if (handle >= b.puts or b.puts - handle > RING) return error.StaleHandle;
        return b.ring[handle % RING];
    }

    /// The port's sampled draw: temperature, then top_k, then softmax, min_p, top_p, and a draw from a
    /// generator keyed at (seed, absolute position). Position keying makes the same (prompt, seed) draw
    /// the same ids in every window shape, so a drafted stream equals the plain stream at the same seed.
    /// `top_k` caps at 256 candidates (0: off, keep 256); the shipped default (20) keeps this a single
    /// pass. The greedy path (temperature 0) never reaches here.
    fn sampleRow(logits_row: []const f32, sp: lanes.sampling.Sampling, position: u64) !u32 {
        var best: [256]f32 = undefined;
        var ids: [256]u32 = undefined;
        var n: usize = 0;
        var k: usize = @min(@as(usize, sp.top_k), 256);
        if (k == 0) k = 256;
        for (logits_row, 0..) |v, id| {
            if (!std.math.isFinite(v)) return error.NonfiniteLogits;
            if (n < k) {
                var j = n;
                while (j > 0 and best[j - 1] < v) : (j -= 1) {
                    best[j] = best[j - 1];
                    ids[j] = ids[j - 1];
                }
                best[j] = v;
                ids[j] = @intCast(id);
                n += 1;
            } else if (v > best[n - 1]) {
                var j = n - 1;
                while (j > 0 and best[j - 1] < v) : (j -= 1) {
                    best[j] = best[j - 1];
                    ids[j] = ids[j - 1];
                }
                best[j] = v;
                ids[j] = @intCast(id);
            }
        }
        const temp: f64 = if (sp.temperature > 0) sp.temperature else 1.0;
        var p: [256]f64 = undefined;
        var sum: f64 = 0;
        for (0..n) |i| {
            p[i] = @exp((@as(f64, best[i]) - @as(f64, best[0])) / temp);
            sum += p[i];
        }
        for (0..n) |i| p[i] /= sum;
        var cut: usize = n;
        if (sp.min_p > 0.0) {
            for (0..n) |i| {
                if (p[i] < sp.min_p * p[0]) {
                    cut = i;
                    break;
                }
            }
        }
        if (sp.top_p < 1.0) {
            var acc: f64 = 0;
            for (0..cut) |i| {
                acc += p[i];
                if (acc >= sp.top_p) {
                    cut = i + 1;
                    break;
                }
            }
        }
        var h = std.hash.Wyhash.init(sp.seed);
        h.update(std.mem.asBytes(&position));
        var prng = std.Random.DefaultPrng.init(h.final());
        const draw_u = prng.random().float(f64);
        var acc2: f64 = 0;
        for (0..cut) |i| {
            acc2 += p[i];
            if (draw_u <= acc2) return ids[i];
        }
        return ids[0];
    }

    /// /533/535: the shared-round batch; the one-row windows of a plain shared round are the only admitted shape,
    /// at most four a forward, and a wider round runs consecutive four-window groups. ON by default -- the served
    /// shape of several cached conversations is exactly those windows, and the batch is token-identical to solo on
    /// every lane (The logits, The 256-token cached streams, The 16/16 multi-turn turns). `TF_DSV41_BATCH=0` is the
    /// same-binary control; any other value (or none) arms it.
    var batch_on: ?bool = null;
    /// The instruments that read a forward's internals take the one-stream path; the batch (and its
    /// four-window grouping) never runs under them.
    fn noBatchInstrument() bool {
        return std.c.getenv("TF_DSV41_STAGE_CB") == null and std.c.getenv("TF_DSV41_LAYER_CB") == null and
            std.c.getenv("TF_DSV41_ATTN_PROF") == null and std.c.getenv("TF_DSV41_HID_HASH") == null and
            std.c.getenv("TF_DSV41_DRAFT_DEBUG") == null;
    }

    fn batchOn() bool {
        if (batch_on) |v| return v;
        const v = std.c.getenv("TF_DSV41_BATCH");
        batch_on = if (v) |e| !std.mem.eql(u8, std.mem.span(e), "0") else true;
        return batch_on.?;
    }

    /// One more occurrence of `t` in the window's penalty counts.
    fn penBump(gpa: std.mem.Allocator, m: *std.AutoHashMapUnmanaged(u32, u32), t: u32) !void {
        const gop = try m.getOrPut(gpa, t);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        for (windows) |w| {
            if (w.parents != null or w.early or w.masks.len > 0) return error.TreeWindowsUnsupported;
        }
        // A round wider than the exact four-lane rung (the server's `--parallel auto` is 8) is verified as
        // consecutive groups of at most four one-row windows. Each group is /532's forward (token-identical to solo
        // at 1/2/3/4 windows); a group of one takes the one-stream path. Before this, five to eight active windows
        // fell back to ALL-single forwards, so the batch switched itself off exactly when the serving load was
        // highest.
        if (windows.len > 4 and batchOn() and noBatchInstrument()) {
            var all_one = true;
            for (windows) |w| if (w.rows() != 1) {
                all_one = false;
                break;
            };
            if (all_one) {
                var done: usize = 0;
                while (done < windows.len) {
                    const n = @min(windows.len - done, 4);
                    try verify(ptr, windows[done..][0..n], out[done..][0..n]);
                    done += n;
                }
                return;
            }
        }
        // A shared round's plain windows are one row each. When the batch is armed, every slot's row goes through
        // ONE forward: the row-wise layers run once over the concatenated rows and the attention core once a
        // segment against that slot's own state. The per-window transaction and the sampling below are unchanged;
        // only the forward's launch is shared.
        const batch = batchOn() and windows.len > 1 and windows.len <= 4 and noBatchInstrument() and blk: {
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
                try b.history[slot].append(b.gpa, w.pending);
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
                if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info("verify: rows={d} held={d} tokens={d}", .{ w.rows(), w.held, w.tokens.len });
                if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null)
                    std.log.info("window: rows={d} held={d} tokens={d} positions={d} pos0={d} state={d}", .{ w.rows(), w.held, w.tokens.len, w.positions.len, if (w.positions.len > 0) w.positions[0] else 0, state.position });
                if (w.positions.len != rows) return error.WindowOutOfStep;
                for (w.positions, 0..) |p, r| if (p != state.position + r + 1) return error.PositionMismatch;
                b.model.commitSlot(i) catch {}; // a window the engine accepted whole leaves its transaction to close here
                try b.model.beginSlot(i);
                opened = true;
                feeds = try b.gpa.alloc(u32, rows);
                feeds[0] = w.pending;
                try b.history[i].append(b.gpa, w.pending);
                const held0 = b.held[i];
                if (w.held > held0.len) return error.WindowOutOfStep;
                @memcpy(feeds[1 .. 1 + w.held], held0[0..w.held]);
                @memcpy(feeds[1 + w.held ..], w.tokens);
                // A row that is not a token (a stale host-held draft) is a wrong guess, never an error: it is replaced
                // with the pending token so the verify simply rejects it and the round costs one round, nothing else.
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
            // TF_DSV41_ROUTE_DUMP=N: the last layer's routes of the first N windows, for the window's
            // expert-distinct count (round 169's question: the five-row MoE's cost is routing-dependent).
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
            // A stream with a positive temperature samples every row; the greedy path below is untouched
            // (byte-identical to the frozen receipts). Draws key at (seed, absolute position), so the same
            // (prompt, seed) produces the same stream in every window shape and a drafted run equals the
            // plain run at the same seed: the port's exactness property, extended to sampling.
            const sampled_stream: ?lanes.sampling.Sampling = if (w.stream.sampling) |sp|
                (if (sp.temperature > 0) sp else null)
            else
                null;
            // Presence/frequency penalties, upstream's semantics (lanes.Penalties, the CUDA lane's tf_penalize_rows
            // form): the reply's tokens before the window -- the stream's context after its prompt, which already
            // carries the pending token -- plus the row's own path. The amount is presence + frequency * count,
            // subtracted from the row's logits before every draw (greedy too).
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
                // With penalties on, row r draws from a penalized copy of its logits. Row r follows the token fed
                // before it (feeds[r]); row 0's history is the context itself. The model's own logits slice is
                // const, so the copy is the row scratch -- only when a request asks.
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
                // The row's own logits and its top-2, for the long-prompt equivalence gate. One 129,280-float file
                // a row, named by position and a process counter; the serve's stderr carries the top-2 and the
                // margin. Nothing in the default path reads this block.
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
            // The target's pick per row, joined to the proposal's top-2 by the round counter (proposal round R step
            // s <-> verify round R row s).
            if (std.c.getenv("TF_DSV41_TREE_PRICE") != null) {
                std.log.info("treeprice verify round {d} rows {d} held {d} pending {d} sampled {any}", .{ ds.price_round, rows, w.held, w.pending, o.sampled[0..rows] });
                ds.price_round += 1;
            }
            // THE ADAPTIVE DEPTH POLICY'S SIGNAL, FED FROM THE VERIFY (the one hook that runs every round -- the
            // keep hook only runs when a draft was REJECTED). THE WINDOW'S ROW 1 IS THE FIRST DRAFT: the host
            // drafts (w.tokens) when the trained proposer is the drafter, the backend's held drafts (held[]) when
            // they are on the device. (The pending code compared against held[] only -- and on the MTP path held[]
            // is EMPTY (the draft hook is never called for this family), so the signal was always ZERO and the
            // policy always ran the cheap window.) The first draft was taken when the verify's row 1 sampled it.
            const first_draft: ?u32 = if (w.held > 0)
                (if (held.len > 0) held[0] else null)
            else if (w.tokens.len > 0)
                w.tokens[0]
            else
                null;
            // THE ACCEPTANCE TEST IS THE ENGINE'S OWN (core/lanes/accept.zig's acceptPath): the path's row `at`
            // continues with the row whose parent is `at` and whose token equals `picks[at]` -- so the FIRST DRAFT
            // (row 1, the pending's child) is accepted iff the model's ROW-0 argmax equals it. The port's test
            // compared `sampled[1]` (the model's prediction AFTER the first draft) with the draft itself: an
            // off-by-one-row test that measures a coincidence, not the acceptance -- which is why the policy could
            // not separate the novel text from the predictable one.
            const hit_dbg: bool = rows > 1 and first_draft != null and o.sampled[0] == first_draft.?;
            // The signal. The round form consumes the PREVIOUS window's outcome (keep sets it, a window that landed
            // whole leaves `keep_pending` false and reads 1.0), so the policy sees the same one-round-lagged
            // quantity the first-draft form sees. The ratio EMA is maintained in every mode (the hybrid's stay
            // branch reads it); the first-draft EMA is the up signal, and in `round` mode it is not fed at all. The
            // governor's accounting. The previous round's commit is not always seen by `keep` (a whole window
            // landing skips it), so the totals are retired here, from the window that round ran and whether its
            // keep fired -- the same one-round lag the ratio reads.
            if (i < b.gov_rounds.len) {
                if (b.gov_rows[i] > 0) {
                    b.gov_rounds[i] += 1;
                    b.gov_committed[i] += if (b.keep_pending[i]) b.gov_kept[i] else b.gov_rows[i];
                }
                b.gov_rows[i] = @intCast(rows);
            }
            const ratio: f64 = if (b.keep_pending[i]) b.keep_ratio[i] else 1.0;
            b.keep_pending[i] = false;
            // The ratchet. Only a deep round can ratchet (a shallow round's ratio IS the first-draft hit); two in a
            // row below the crossover put this request on the shallow window.
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
            if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info("verify sampled: {d}", .{o.sampled[0]});
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            const i = try b.slotOf(w.stream);
            const rows = w.rows();
            const kept = path.len;
            if (i < b.gov_kept.len) b.gov_kept[i] = @intCast(kept);
            b.keep_pending[i] = true;
            b.keep_ratio[i] = if (rows > 1) @as(f64, @floatFromInt(kept)) / @as(f64, @floatFromInt(rows - 1)) else 1.0;
            if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info("keep: slot {d} kept {d} of {d}", .{ i, kept, rows });
            // THE ADAPTIVE DEPTH POLICY'S SIGNAL, TAKEN WHERE THE KEPT COUNT IS KNOWN. It was recorded in
            // `portKeep`, which the whole-window case never reaches (`kept >= rows` returns early) -- and the whole
            // window landing IS the high-yield case, so the policy never saw the predictable text's yield and never
            // switched up.
            if (kept >= rows) continue; // the whole window landed
            try b.model.rollbackSlot(i);
            // The ring rows the window's rejected rows published are aliases the next forward can still read; put
            // the stashed slots back before anything runs.
            try b.model.restoreRing(i, @intCast(kept), @intCast(rows));
            if (kept == 0) continue; // the rollback closed the transaction
            var feeds = try b.gpa.alloc(u32, kept);
            defer b.gpa.free(feeds);
            feeds[0] = w.pending;
            const held = b.held[i];
            const held_kept = @min(w.held, kept - 1);
            if (held_kept > held.len) return error.WindowOutOfStep;
            try b.history[i].appendSlice(b.gpa, feeds[1..kept]);
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
            // Diagnostic (TF_DSV41_CLEAR_STALE=1): the rejected rows the window wrote stay in the ring; zero them so
            // the re-feed cannot read them, and see whether the divergence survives.
            // The compressed path's partial sums for the block the rejected draft row started are scratch that the next
            // forward would read as complete; clearing them is the round-257 finding's fix candidate.
            if (std.c.getenv("TF_DSV41_CLEAR_PARTIALS") != null) b.model.caches[i].clearPartials();
            if (std.c.getenv("TF_DSV41_CLEAR_STALE") != null) {
                const start = b.model.caches[i].position;
                b.model.caches[i].clearRows(start + kept, start + rows);
            }
            const again = b.model.forwardOn(i, feeds, .last) catch |e| {
                std.log.err("keep: refeed failed: {s}", .{@errorName(e)});
                return e;
            };
            if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
                var best: u32 = 0;
                var maximum = -std.math.inf(f32);
                for (again, 0..) |v, id| if (v > maximum) {
                    maximum = v;
                    best = @intCast(id);
                };
                std.log.info("keep: refeed sampled {d}, verify sampled {d}{s}", .{ best, b.last_sampled[0], if (best == b.last_sampled[0]) "" else "  <-- DIVERGES" });
            }
            // The refeed's own forward closed the transaction; commit only if one is still open.
            b.model.commitSlot(i) catch {};
        }
    }

    /// Nothing to absorb: the lookup drafter reads the stream's own context each round.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info(
            "draft() called: requests={d} mtp_env={} dspark={} pending={?} depth={d}",
            .{ requests.len, std.c.getenv("TF_DSV41_MTP_DRAFTS") != null, b.model.weights.dspark != null, if (requests.len > 0) requests[0].stream.pending else null, if (requests.len > 0) requests[0].depth else 0 },
        );
        for (requests) |r| {
            const i = b.by.get(r.stream) orelse continue;
            if (b.held[i].len > 0) b.gpa.free(b.held[i]);
            b.held[i] = &.{};
            if (r.depth == 0) continue;
            if (draftGovernorOff(b, i)) {
                // The switch must reach the lane planner, not only the held list. `Stream.drafts` is what
                // `windows.plan` reads before it queues a window at all: clearing it is the clean off switch.
                r.stream.drafts = false;
                continue;
            }
            // The trained proposer (TF_DSV41_MTP_DRAFTS) runs HERE, between rounds: its own forward writes the
            // shared logits buffer from row 1 on, so it must not run inside a forward whose logits a verify reads.
            // The pending token is the prompt's last or the last committed one; the lane does not always carry it.
            const last_tok: ?u32 = r.stream.pending orelse
                (if (b.history[i].items.len > 0) b.history[i].items[b.history[i].items.len - 1] else null);
            if (std.c.getenv("TF_DSV41_MTP_DRAFTS") != null and b.model.weights.dspark != null and
                last_tok != null and r.depth <= mtpDepthMax())
            mtp: {
                // The drafter's own forward is sized to the window the round will verify (the policy's depth), not
                // a hardcoded three: its rows are `count + 1`, so a depth-1 window needs two rows.
                // TF_DSV41_MTP_DRAFT_ROWS=0 is the same-binary control.
                const count: u32 = if (draftRowsFollowWindow()) mtpDepthFor(b, @intCast(i)) else 3;
                if (count == 0) break :mtp;
                const drafts = b.model.proposeTokens(@intCast(i), last_tok.?, count) catch |err| {
                    std.log.warn("mtp: drafts unavailable ({s})", .{@errorName(err)});
                    break :mtp;
                };
                // The engine plans its window from r.depth, so the hand-off must always carry that many rows; the
                // proposer's last computed slot fills any it does not have (the verify rejects a stale repeat).
                const out = try b.gpa.alloc(u32, r.depth);
                const n = @min(r.depth, count);
                @memcpy(out[0..n], drafts[0..n]);
                for (out[n..]) |*t| t.* = drafts[n - 1];
                b.held[i] = out;
                // The round's other hook (tree) asks for the same drafts; keep them here.
                if (i < b.mtp_cache.len) b.mtp_cache[i] = .{ .pending = last_tok.?, .drafts = drafts, .count = count };
                continue;
            }
            // Diagnostic (TF_DSV41_DRAFT_COPY=1): every draft repeats the pending token, so a two-row window is
            // [pending, pending]. If the verify's row 0 still matches a plain forward, its mask is sound.
            const want: usize = r.depth;
            if (std.c.getenv("TF_DSV41_DRAFT_COPY") != null) {
                const out0 = try b.gpa.alloc(u32, want);
                const pend = r.stream.pending orelse 0;
                for (out0) |*t| t.* = pend;
                b.held[i] = out0;
                continue;
            }
            const out = try b.gpa.alloc(u32, want);
            const got = lookupTokens(r.stream.prompt(), b.history[i].items, out);
            const fill = if (got > 0) out[got - 1] else (r.stream.pending orelse 0);
            for (out[got..]) |*t| t.* = fill;
            b.held[i] = out;
        }
    }

    /// A suffix lookup drafter (TF_DSV41_LOOKUP_DRAFTS=<depth>): the tokens that followed the last earlier
    /// occurrence of the stream's own tail are this round's drafts, offered as host tokens so the engine's verify
    /// and rollback do the accepting. A wrong guess costs a round, never a different token.
    fn tree(ptr: *anyopaque, s: *Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
        const b = self(ptr);
        // The proposer's forward needs the lane's slot (the trunk's key ring lives there).
        const slot_i = b.by.get(s) orelse return null;
        if (draftGovernorOff(b, slot_i)) {
            s.drafts = false;
            return null;
        }
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) std.log.info("tree() called: pending={?} depth={d}", .{ s.pending, lookupDepth() });
        var depth = lookupDepth();
        // THE ADAPTIVE DEPTH POLICY. The engine measures the yield it just got (the accepted drafts a round) and
        // runs the cheap depth-1 window when the yield is low (novel text) and the deep window when it is high
        // (predictable text). TF_DSV41_MTP_ADAPT=<threshold> arms the policy; the default is the fixed depth. THE
        // ADAPTIVE DEPTH POLICY IS THE DEFAULT. THE SIGNAL is the first draft's acceptance -- an EMA over the
        // request's rounds, reset at the request's start. THE BAND: the deep window when the signal passes UP, the
        // cheap window otherwise, the deep window kept while the signal stays at least DOWN. TF_DSV41_MTP_ADAPT=0
        // the fixed depth (the control arm) TF_DSV41_MTP_ADAPT_UP=<up> the band TF_DSV41_MTP_ADAPT_DOWN=<d> the
        // band's floor (the default 0.7) TF_DSV41_MTP_ADAPT=<t> the single-threshold form (the deep window while
        // the signal > t)
        const adapt_raw = std.c.getenv("TF_DSV41_MTP_ADAPT");
        const adapt_off = adapt_raw != null and std.mem.eql(u8, std.mem.span(adapt_raw.?), "0");
        if (!adapt_off) {
            const prev: u32 = if (slot_i < b.last_depth.len) b.last_depth[slot_i] else 0;
            depth = mtpDepthFor(b, slot_i);
            // THE CONFIRMED SWITCH-UP. A single sample above UP is not a change of text: on the novel band the deep
            // rounds accepted not one extra draft, so the up-transition must hold for `confirm` consecutive rounds.
            // The deep-stay branch (prev >= 2) is not a transition and is not gated; a depth-1 round resets the
            // streak.
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
        // The trained proposer (TF_DSV41_MTP_DRAFTS) is the drafter when it is bound and has a pending token; the
        // suffix lookup below stays the fallback. This is the hook the lane core actually calls (round 227: draft()
        // is never invoked for this family).
        if (depth > 0 and std.c.getenv("TF_DSV41_MTP_DRAFTS") != null and b.model.weights.dspark != null and s.pending != null) {
            // The round's other hook (draft) already ran the proposer's forward for this pending token; reuse it
            // instead of running the draft's three blocks, the conditioning and the head twice. The hand-off's cap
            // is three a round; TF_DSV41_MTP_DEPTH_MAX lifts it so the depth-4 window (five rows) can be offered.
            // The default is the shipped cap.
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
        if (depth == 0) return null;
        const ctx = s.context.items;
        if (ctx.len < 4) return null;
        const tail: usize = @min(ctx.len - 1, 12);
        var n = tail;
        while (n >= 2) : (n -= 1) {
            const suffix = ctx[ctx.len - n ..];
            var i: usize = 0;
            while (i + n < ctx.len) : (i += 1) {
                if (std.mem.eql(u32, ctx[i..][0..n], suffix)) {
                    const start = i + n;
                    const avail = @min(@as(usize, depth), ctx.len - start);
                    if (avail == 0) return null;
                    const tokens = try gpa.dupe(u32, ctx[start..][0..avail]);
                    return .{ .count = @intCast(avail), .tokens = tokens };
                }
            }
        }
        return null;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
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
};
