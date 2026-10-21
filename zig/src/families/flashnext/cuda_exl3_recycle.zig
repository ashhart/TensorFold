//! Stable buffers for captured decode graphs (cuda_exl3_graph.zig): a graph holds a sequence's and its draft window's
//! device addresses, and every request used to allocate new ones (its own capacity, its own buffers), so verify
//! windows rarely repeated a graph key and ran eagerly or paid a capture. With graphs on, a stream's capacity is
//! rounded up to a multiple of `bucket` tokens and a finished stream's sequence and draft buffers are kept (zeroed on
//! reuse exactly as a new one is) for the next stream of the same capacity, on both ranks.
//! The capacity only bounds the caches and strides the block scores; no kernel reads past a row's own positions
//! (fn_qsa_select uses the stride as a stride), so the bits do not change (tf-cuda-test flashnext-exl3-graph eq and
//! the serve gate's hash). TF_FLASHNEXT_EXL3_RECYCLE=0 turns it off.

const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const Draft = @import("cuda_mtp_draft.zig").Draft;

pub const bucket: usize = 4096;
const pool_n = 4;

pub fn enabled() bool {
    if (!@import("cuda_exl3_graph.zig").enabled()) return false;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_RECYCLE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// A stream's capacity: `cap` rounded up to the bucket (at most `limit`) when recycling.
pub fn roundCap(cap: usize, limit: usize) usize {
    if (!enabled()) return cap;
    return @min(limit, (cap + bucket - 1) / bucket * bucket);
}

var seqs: [pool_n]?prompt.Seq = @splat(null);
var drafts: [pool_n]?Draft = @splat(null);
var draining = false;
pub var stats: struct { seq_reused: u64 = 0, seq_kept: u64 = 0, draft_reused: u64 = 0, draft_kept: u64 = 0 } = .{};

fn zero(bufs: []cuda.DeviceBuffer, s: cuda.Stream) !void {
    for (bufs) |b| try b.fill8(0, s.handle);
}

/// Engine.newSeq's hook: a kept sequence of capacity `cap`, zeroed as newSeq zeroes a new one; null: allocate.
pub fn takeSeq(e: *prompt.Engine, cap: usize) !?prompt.Seq {
    if (!enabled()) return null;
    for (&seqs) |*slot| if (slot.*) |*q| if (q.cap == cap) {
        var s = q.*;
        slot.* = null;
        const st = e.stream.*;
        for ([_][]cuda.DeviceBuffer{ s.conv, s.state, s.kc, s.vc, s.ikc, s.pooled }) |bs| try zero(bs, st);
        try s.tail.fill8(0, st.handle);
        s.history = .{ e.eos, e.eos };
        s.fed = 0;
        s.rank_id = 0;
        s.media = null;
        stats.seq_reused += 1;
        return s;
    };
    return null;
}

fn canRetainSeq(s: *const prompt.Seq) bool {
    return !s.invalidated and s.media == null and s.kv8 == null and s.ikc_host == null and s.paged == null;
}

/// Seq.deinit's hook: true when the sequence was kept for reuse (a plain one: no media, int8 or paged KV).
pub fn giveSeq(s: *prompt.Seq) bool {
    if (draining or !enabled() or !canRetainSeq(s)) return false;
    for (&seqs) |*slot| if (slot.* == null) {
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
    for (&drafts) |*slot| if (slot.*) |*d| if (d.cap == cap and d.dlogits.len == draft_n * 2 and d.rows_cap == @max(rows, @import("cuda_mtp.zig").max_rows)) {
        var x = d.*;
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
pub fn drain(gpa: std.mem.Allocator) void {
    draining = true;
    defer draining = false;
    for (&seqs) |*slot| if (slot.*) |*q| {
        q.deinit(gpa);
        slot.* = null;
    };
    for (&drafts) |*slot| if (slot.*) |*d| {
        d.deinit();
        slot.* = null;
    };
}
