//! Stable buffers for captured decode graphs (cuda_exl3_graph.zig): a graph holds a sequence's and its draft window's

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("../../cuda_prompt.zig");
const Draft = @import("../../cuda_mtp_draft.zig").Draft;

pub const bucket: usize = 4096;
const pool_n = 4;

pub fn enabled() bool {
    return @import("../graph/cuda_exl3_graph.zig").enabled();
}

/// A stream's capacity: `cap` rounded up to the bucket (at most `limit`) when recycling.
pub fn roundCap(cap: usize, limit: usize) usize {
    if (!enabled()) return cap;
    return @min(limit, (cap + bucket - 1) / bucket * bucket);
}

var seqs: [pool_n]?prompt.Seq = @splat(null);
var drafts: [pool_n]?Draft = @splat(null);
var draining = false;
pub var stats: struct { seq_reused: u64 = 0, seq_kept: u64 = 0, seq_evicted: u64 = 0, shared_lane_reassigned: u64 = 0, draft_reused: u64 = 0, draft_kept: u64 = 0 } = .{};

fn zero(bufs: []cuda.DeviceBuffer, s: cuda.Stream) !void {
    for (bufs) |b| try b.fill8(0, s.handle);
}

/// Engine.newSeq's hook: a kept sequence of capacity `cap`, zeroed as newSeq zeroes a new one; null: allocate.
pub fn takeSeq(e: *prompt.Engine, cap: usize) !?prompt.Seq {
    const reuse = enabled();
    for (&seqs) |*slot| if (slot.*) |*sequence| {
        if (sequence.recycle_owner == e and (!reuse or !matches(sequence, e, cap))) {
            discardSeq(slot, e.gpa);
            stats.seq_evicted += 1;
        }
    };
    if (!reuse) return null;
    for (&seqs) |*slot| if (slot.*) |*q| if (matches(q, e, cap)) {
        var s = q.*;
        slot.* = null;
        errdefer {
            s.tail.d.synchronize() catch @panic("recycled sequence reset did not drain");
            s.invalidated = true;
            s.deinit(e.gpa);
        }
        const st = e.stream.*;
        var previous_lane: ?u32 = null;
        if (s.paged) |pg| {
            if (pg.cfg.shared_lanes != 0) previous_lane = pg.shared_sequence.lane;
            try pg.resetForReuse(st);
            if (previous_lane) |lane| if (pg.shared_sequence.lane != lane) {
                stats.shared_lane_reassigned += 1;
                @import("../graph/cuda_exl3_graph.zig").releaseSeq(&s);
            };
        }
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| try zero(bs, st);
        try s.tail.fill8(0, st.handle);
        @memset(s.logits, 0);
        s.history = .{ e.eos, e.eos };
        s.fed = 0;
        s.rank_id = 0;
        s.media = null;
        s.untaught = false;
        s.invalidated = false;
        stats.seq_reused += 1;
        return s;
    };
    return null;
}

fn discardSeq(slot: *?prompt.Seq, allocator: std.mem.Allocator) void {
    var sequence = slot.*.?;
    slot.* = null;
    sequence.recycle_owner = null;
    sequence.deinit(allocator);
}

fn canRetainSeq(s: *const prompt.Seq) bool {
    if (s.invalidated or s.media != null or s.kv8 != null or s.recycle_owner == null) return false;
    const placement = @import("../kv/cuda_exl3_kv.zig").of(s.recycle_owner.?);
    return placement.k8 == null and placement.heads == @import("../../cuda_prompt_wt.zig").kv_heads and (s.ikc_host == null or s.paged != null);
}

fn matches(sequence: *const prompt.Seq, engine: *prompt.Engine, cap: usize) bool {
    if (sequence.recycle_owner != engine or sequence.tail.d != engine.driver or sequence.cap != cap) return false;
    const placement = @import("../kv/cuda_exl3_kv.zig").of(engine);
    if (placement.k8 != null or placement.heads != @import("../../cuda_prompt_wt.zig").kv_heads or placement.ikc_host != (sequence.ikc_host != null)) return false;
    const cfg = if (engine.page_config) |config| config.forChunk(engine.chunk) else null;
    if (sequence.paged) |pg| return cfg != null and std.meta.eql(pg.cfg, cfg.?) and pg.shared_arena == engine.shared_arena;
    return cfg == null;
}

/// Seq.deinit's hook: true when the sequence's BF16 allocations were kept for reuse.
pub fn giveSeq(s: *prompt.Seq) bool {
    if (draining or !enabled() or !canRetainSeq(s)) return false;
    for (&seqs) |*slot| if (slot.* == null) {
        if (s.paged) |pg| {
            @import("../graph/cuda_exl3_graph.zig").releaseDecodeSeq(s);
            pg.retireForReuse() catch @panic("recycled shared KV owner did not retire");
        } else s.tail.d.synchronize() catch @panic("recycled sequence consumers did not drain");
        slot.* = s.*;
        stats.seq_kept += 1;
        return true;
    };
    return false;
}

test "exl3: invalidated restore destinations are never retained for reuse" {
    var sequence: prompt.Seq = undefined;
    sequence.invalidated = true;
    try std.testing.expect(!canRetainSeq(&sequence));
}

/// Draft.init's hook: a kept draft window of the same shape, its caches zeroed and its state reset; null: allocate.
pub fn takeDraft(stream: cuda.Stream, cap: usize, draft_n: usize, rows: usize) !?Draft {
    if (!enabled()) return null;
    for (&drafts) |*slot| if (slot.*) |*d| if (d.cap == cap and d.dlogits.len == draft_n * 2 and d.rows_cap == @max(rows, @import("../../cuda_mtp.zig").max_rows)) {
        var x = d.*;
        x.stream = stream;
        x.verify_cache.invalidate();
        slot.* = null;
        for ([_]cuda.DeviceBuffer{ x.kc, x.vc, x.ikc, x.pooled }) |b| try b.fill8(0, stream.handle);
        x.media = null;
        x.len = 0;
        x.drafted = 0;
        x.rows = 0;
        x.open = false;
        x.prompt_last = 0;
        x.fed0 = 0;
        x.history = .{ 0, 0 };
        x.unused_history = .{ 0, 0 };
        stats.draft_reused += 1;
        return x;
    };
    return null;
}

/// Draft.deinit's hook: true when the window was kept.
pub fn giveDraft(d: *Draft) bool {
    if (draining or !enabled() or d.media != null) return false;
    for (&drafts) |*slot| if (slot.* == null) {
        slot.* = d.*;
        stats.draft_kept += 1;
        return true;
    };
    return false;
}

/// Frees everything kept (the engine's deinit).
pub fn drain(engine: *prompt.Engine) void {
    draining = true;
    defer draining = false;
    for (&seqs) |*slot| if (slot.*) |*sequence| if (sequence.recycle_owner == engine) {
        discardSeq(slot, engine.gpa);
    };
    for (&drafts) |*slot| if (slot.*) |*d| {
        d.deinit();
        slot.* = null;
    };
}

test {
    _ = @import("cuda_exl3_recycle_test.zig");
}
