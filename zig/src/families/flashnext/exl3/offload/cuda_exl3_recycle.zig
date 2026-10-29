//! Stable buffers for captured decode graphs: a closed stream's caches and draft window are kept for the next one.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("../../cuda_prompt.zig");
const Draft = @import("../../cuda_mtp_draft.zig").Draft;

pub const bucket: usize = 4096;
const pool_n = 4;

pub fn enabled() bool {
    return @import("../graph/cuda_exl3_graph.zig").enabled();
}

/// A stream's capacity for `want` positions: a pool lane's own size, else rounded up to the bucket (at most `limit`).
pub fn capFor(pooled: bool, want: usize, limit: usize) usize {
    const cap = @min(want, prompt.max_tokens);
    if (pooled or !enabled()) return cap;
    return @min(@min(limit, prompt.max_tokens), (cap + bucket - 1) / bucket * bucket);
}

var seqs: [pool_n]?prompt.Seq = @splat(null);
var drafts: [pool_n]?Draft = @splat(null);
var draining = false;
pub var stats: struct { seq_reused: u64 = 0, seq_kept: u64 = 0, seq_evicted: u64 = 0, draft_reused: u64 = 0, draft_kept: u64 = 0 } = .{};

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
            s.recycle_owner = null; // a half-reset sequence is freed, not kept again
            s.deinit(e.gpa);
        }
        const st = e.stream.*;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| try zero(bs, st);
        try s.tail.fill8(0, st.handle);
        @memset(s.logits, 0);
        s.history = .{ e.eos, e.eos };
        s.fed = 0;
        s.rank_id = 0;
        s.media = null;
        s.untaught = false;
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
    return s.media == null and s.paged == null and s.recycle_owner != null;
}

fn matches(sequence: *const prompt.Seq, engine: *prompt.Engine, cap: usize) bool {
    if (sequence.recycle_owner != engine or sequence.tail.d != engine.driver or sequence.cap != cap) return false;
    return sequence.paged == null;
}

/// Seq.deinit's hook: true when the sequence's BF16 allocations were kept for reuse.
pub fn giveSeq(s: *prompt.Seq) bool {
    if (draining or !enabled() or !canRetainSeq(s)) return false;
    for (&seqs) |*slot| if (slot.* == null) {
        s.tail.d.synchronize() catch @panic("recycled sequence consumers did not drain");
        slot.* = s.*;
        stats.seq_kept += 1;
        return true;
    };
    return false;
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
    if (draining or !enabled() or d.media != null or d.pg.pt != 0) return false; // a pool lane's head is the pool's
    // A peer's contiguous window can swap in a paged sequence's borrowed run states.
    for (d.spare) |state| if (state.borrowed) return false;
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
