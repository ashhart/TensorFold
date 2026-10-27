//! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const snapshots = @import("cuda_snapshots.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// The fewest rows a paired step's second chunk takes: a shorter one's sites end before the first's exchanges do.
pub const min_rows = fw.prefill_rows / 4;

/// A paired step's second chunk: twins of the prompt buffers, pieces on their side stream, and its first position.
pub const Pair = struct {
    bp: fw.Buffers,
    pieces: fw.Pieces,
    pos: cuda.DeviceBuffer, // an i32: where the second chunk's view of the sequence starts
    steps: u64 = 0, // paired steps so far (glm-pair-check counts them)

    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, bp: *const fw.Buffers, pieces: *const fw.Pieces, capacity: usize) !Pair {
        var b = try fw.Buffers.initTwin(d, w, bp, capacity);
        errdefer b.deinit();
        var p = try fw.Pieces.initTwin(d, pieces);
        errdefer p.deinit();
        return .{ .bp = b, .pieces = p, .pos = try cuda.DeviceBuffer.alloc(d, 4) };
    }

    pub fn deinit(p: *Pair) void {
        p.pos.free();
        p.pieces.deinit();
        p.bp.deinit();
        p.* = undefined;
    }
};

/// Whether a step at `start` of a `len`-id prompt takes the next chunk too: min_rows or more there, and no mark at it.
pub fn takesTwo(start: usize, len: usize, world: usize, marks: []const u32) bool {
    const next = start + fw.prefill_rows;
    if (next >= len or len - next < @max(min_rows, world)) return false;
    return std.mem.indexOfScalar(u32, marks, @intCast(next)) == null;
}

/// Whether a pair at `start` (b of `rb` rows) rebases the index rings where chunks one at a time do: at a, or not.
pub fn ringsAgree(base: usize, ring: usize, start: usize, rb: usize) bool {
    const ra = fw.prefill_rows;
    return start + ra > base + ring or start + ra + rb <= base + ring;
}

/// Rank 0's choice for the step at `start` (the others take the ids it sends); not with a drafter, prompt rows or prof.
pub fn pairedAt(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    if (e.pair == null or !e.settings.pair or e.drafter != null or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    if (e.pieces.count == 1 and !e.settings.split) return false; // run takes such a chunk whole
    if (!takesTwo(start, len, e.share.world, if (marks) |m| m.at else &.{})) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, @min(len - start - fw.prefill_rows, fw.prefill_rows));
}

/// Engine.prefillChunk of two chunks (`part`: both, then the next id): one pass, then each chunk's MTP rows and commit.
pub fn prefill(e: *Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    const ra: usize = fw.prefill_rows;
    if (len <= start + ra) return error.BadChunk;
    const rb: usize = @min(len - start - ra, fw.prefill_rows);
    const pair = if (e.pair) |*p| p else return error.BadChunk;
    if (rb < e.share.world or part.len != @min(len - start, ra + rb + 1) or e.drafter != null or e.prompt_logprobs != null) return error.BadChunk;
    if (!ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rb)) return error.BadChunk; // pairedAt's rule, every rank
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    e.gate(cancel) catch |err| {
        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
        return err;
    };
    var one: [1]fw.Rank = undefined;
    var two: [1]fw.Rank = undefined;
    var fa = e.view(&e.bp, &one);
    var fb = e.view(&pair.bp, &two);
    fa.prof = null;
    fb.prof = null;
    fa.index_split = split; // the step's choice (TF_GLM_INDEX_SPLIT) for both chunks
    fb.index_split = split;
    const st = &e.cur.st;
    try st.ready(e.stream, ra + rb); // one rebase for both chunks, at a as a chunk at a time, before b's view copies it
    var sb = st.*;
    sb.pos_dev = pair.pos.ptr;
    try sb.setPos(e.stream, st.pos + ra);
    two[0].st = &sb;
    st.pending = ra;
    st.prompt = true;
    const ids: []const i32 = @ptrCast(part);
    try cuda.copy_source.enqueue(e.d, e.bp.ids, std.mem.sliceAsBytes(ids[0..ra]), e.stream);
    try cuda.copy_source.enqueue(e.d, pair.bp.ids, std.mem.sliceAsBytes(ids[ra .. ra + rb]), e.stream);
    try fw.Forward.pairBody(&fa, &fb, ra, rb);
    pair.steps += 1;
    e.index_splits += @intFromBool(split);
    try mtpRows(e, &fa, ids[1 .. ra + 1], e.bp.fnormed);
    try fa.commit(ra);
    st.pending = rb;
    st.prompt = true;
    var first: ?u32 = null;
    if (start + ra + rb == len) {
        var t: [1]u32 = undefined;
        try e.draw(pair.bp.logits, 1, len, 0, &t);
        first = t[0];
        try e.holdPrompt(pair.bp.fnormed + (rb - 1) * e.c.hidden * 2);
    }
    try mtpRows(e, &fa, ids[ra + 1 ..], pair.bp.fnormed); // the MTP head on a's buffers, over b's rows
    try fa.commit(rb);
    return first;
}

/// decode._absorb_rows of a chunk: the MTP head over its next ids `nxt` on its final-norm rows `hidden`.
fn mtpRows(e: *Engine, f: *fw.Forward, nxt: []const i32, hidden: u64) !void {
    if (!e.mtp or nxt.len == 0) return;
    try f.mtp(nxt, &.{hidden});
    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
}

test "a prompt step takes the next chunk when it holds min_rows or more and no kept state starts there" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, 2 * C, 3, &.{}));
    try std.testing.expect(takesTwo(0, C + min_rows, 3, &.{}));
    try std.testing.expect(!takesTwo(0, C + min_rows - 1, 3, &.{})); // a shorter tail goes alone
    try std.testing.expect(!takesTwo(0, C, 3, &.{}));
    try std.testing.expect(!takesTwo(C, 2 * C, 3, &.{})); // the prompt's last chunk
    try std.testing.expect(takesTwo(2 * C, 5 * C, 3, &.{}));
    try std.testing.expect(!takesTwo(0, 3 * C, 3, &.{C})); // a state kept at the next chunk's start: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, 3, &.{ 2 * C, 3 * C })); // marks past the pair
}

test "a pair rebases the index rings where a chunk at a time would, so a ring based at chunk 0 pairs from chunk 2" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C)); // a chunk at a time rebases at b's start, a pair at a's
    try std.testing.expect(!ringsAgree(0, 2 * C, C, 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, C)); // both at a's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, C));
}
