//! V4.1 lane slots: a stream a slot, greedy, drafts off; a window is a slot's sub-forward, so rows keep solo bits.
const std = @import("std");
const ds = @import("model.zig");
const kv = @import("metal_kv_state.zig");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const greedy = @import("greedy.zig");

const RING = 1024;

/// The verify's greedy sample. TF_DSV41_FAST_ARGMAX=0 is the scalar control; unset is the fast form.
const policy = @import("backend_policy.zig");
const MtpCache = policy.MtpCache;
const fastArgmax = policy.fastArgmax;
const tokAt = policy.tokAt;
const lookupTokens = policy.lookupTokens;
const mtpDepthMax = policy.mtpDepthMax;
const lookupDepth = policy.lookupDepth;
const noRefeedOn = policy.noRefeedOn;
const draftRowsFollowWindow = policy.draftRowsFollowWindow;
const roundSignalMode = policy.roundSignalMode;
const roundSignalOn = policy.roundSignalOn;
const mtpConfirmRounds = policy.mtpConfirmRounds;
const mtpRatchetDown = policy.mtpRatchetDown;
const draftOffArmed = policy.draftOffArmed;
const draftOffBelow = policy.draftOffBelow;
const draftOffAfter = policy.draftOffAfter;
const draftOffGateCtx = policy.draftOffGateCtx;
const draftOffLow = policy.draftOffLow;
const draftOffAfterFor = policy.draftOffAfterFor;
const mtpDepthFor = policy.mtpDepthFor;
const draftGovernorOff = policy.draftGovernorOff;

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
    /// The trained proposer's last proposal, so the round runs the draft forward ONCE.
    mtp_cache: [32]?MtpCache = @splat(null),
    /// Each slot's committed tokens after the prompt, for the lookup drafter's matches.
    history: []std.ArrayListUnmanaged(u32),
    /// What the last verify sampled per row, so a rejected window's re-feed can be compared with it.
    last_sampled: [8]u32 = @splat(0),
    /// The depth each slot's last round RAN -- the adaptive policy's state.
    last_depth: []u32,
    /// Consecutive rounds the up-condition has held, for the confirmed switch-up.
    up_streak: []u32,
    /// The last window's kept/offered, for the round-acceptance signal.
    keep_pending: []bool,
    keep_ratio: []f64,
    /// The ratchet's per-slot state -- the request has fallen back to the shallow window.
    shallow: []bool,
    low_streak: []u32,
    /// The drafting governor's per-slot state -- the rounds this request has completed, the tokens they committed.
    gov_rounds: []u32,
    gov_committed: []u32,
    gov_rows: []u32,
    gov_kept: []u32,
    gov_off: []bool,
    /// The current window's presence/frequency counts (the reply before it plus the row's path).
    pen_counts: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    pen_row: []f32 = &.{},

    const run = @import("backend_run.zig");
    pub const verify = run.verify;
    pub const keep = run.keep;
    pub const draft = run.draft;
    pub const tree = run.tree;
    pub const release = run.release;

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

    /// The lane vtable; `prefill_step` (one chunk a call, decode rounds between) unless TF_DSV41_PREFILL_INTERLEAVE=0.
    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .penalties = true, .vtable = if (pass.interleaveOn()) &vtable_stepped else &vtable_whole };
    }

    const vtable_whole: be.Backend.VTable = .{
        .prefill = prefill,
        .first = firstFn,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .tree = tree,
        .release = release,
    };
    const vtable_stepped: be.Backend.VTable = blk: {
        var v = vtable_whole;
        v.prefill_step = prefill_step;
        v.reports_fill = true;
        break :blk v;
    };

    /// A window of any width runs exactly as its slot's own forward; the lane core may bundle streams freely.
    pub fn facts(b: *const Backend) lanes.Model {
        const rows = b.model.workspace.plan.rows;
        return .{
            // The widest window whose rows reproduce one-token steps: rows <= 8; wider cannot verify.
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
            // The served lanes' exactness contract: a shared round runs every stream's plain window.
            .shared_plain = true,
            .batch_rows = rows,
            .max_streams = @intCast(b.model.lanes),
            .draft_streams = false,
        };
    }

    pub fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn slotOf(b: *Backend, s: *Stream) !u32 {
        return b.by.get(s) orelse error.UnknownStream;
    }

    pub fn put(b: *Backend, v: u32) u64 {
        const h = b.puts;
        b.ring[h % RING] = v;
        b.puts += 1;
        return h;
    }

    pub fn free(b: *const Backend) ?u32 {
        var used: [256]bool = @splat(false);
        var it = b.by.valueIterator();
        while (it.next()) |v| used[v.*] = true;
        for (used, 0..) |u, i| if (b.model.lanes > i and !u) return @intCast(i);
        return null;
    }

    /// The prefill's first draw and the queue's next.
    pub fn draw(b: *Backend, slot: u32, s: *Stream, position: u64) !u32 {
        _ = slot;
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

    const pass = @import("backend_prefill.zig");
    const prefill = pass.whole;
    const prefill_step = pass.step;

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
        const b = self(ptr);
        if (handle >= b.puts or b.puts - handle > RING) return error.StaleHandle;
        return b.ring[handle % RING];
    }

    /// The port's sampled draw: temperature, then top_k, then softmax, min_p, top_p.
    pub fn sampleRow(logits_row: []const f32, sp: lanes.sampling.Sampling, position: u64) !u32 {
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

    /// The shared-round batch; the one-row windows of a plain shared round are the only admitted shape.
    var batch_on: ?bool = null;
    var batch_max: ?usize = null;
    /// The instruments that read a forward's internals take the one-stream path; the batch never runs under them.
    pub fn noBatchInstrument() bool {
        return std.c.getenv("TF_DSV41_LAYER_CB") == null and
            std.c.getenv("TF_DSV41_ATTN_PROF") == null and std.c.getenv("TF_DSV41_HID_HASH") == null;
    }

    /// TF_DSV41_BATCH_MAX: the most one-row windows one batched forward carries (default 4, ceiling 8).
    pub fn batchMax() usize {
        if (batch_max) |v| return v;
        const raw = std.c.getenv("TF_DSV41_BATCH_MAX");
        const parsed = if (raw) |e| std.fmt.parseInt(usize, std.mem.span(e), 10) catch 4 else 4;
        batch_max = std.math.clamp(parsed, 1, 8);
        return batch_max.?;
    }

    pub fn batchOn() bool {
        if (batch_on) |v| return v;
        const v = std.c.getenv("TF_DSV41_BATCH");
        batch_on = if (v) |e| !std.mem.eql(u8, std.mem.span(e), "0") else true;
        return batch_on.?;
    }

    /// One more occurrence of `t` in the window's penalty counts.
    pub fn penBump(gpa: std.mem.Allocator, m: *std.AutoHashMapUnmanaged(u32, u32), t: u32) !void {
        const gop = try m.getOrPut(gpa, t);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
};
