//! A solo prompt pass (TF_GLM_CHUNK_SOLO): two grid chunks in one pass, while no other stream decodes.
const std = @import("std");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// Whether a stream besides `s` is past its prompt pass (a lane with no fill): a solo pass would hold up its rounds.
pub fn othersDecode(lanes: anytype, s: anytype) bool {
    var it = lanes.iterator();
    while (it.next()) |kv| if (kv.key_ptr.* != s and kv.value_ptr.fill == null) return true;
    return false;
}

/// Before the weights load: a drafter's steps take a chunk, so the buffers for two would hold memory no pass uses.
pub fn refuse(solo: bool, drafter: ?[]const u8) !void {
    if (solo and drafter != null) return error.GlmSoloWithDrafter;
}

/// The rows an engine sizes its prompt buffers for: at start, as they hold memory all the time.
pub fn bufferRows(on: bool) usize {
    return if (on) fw.solo_rows else fw.prefill_rows;
}

/// Whether a pass at `start` of a `len`-id prompt can reach past the next grid point: two rows, and no mark at it.
pub fn takesTwo(start: usize, len: usize, marks: []const u32) bool {
    const next = start + fw.prefill_rows;
    return next + 1 < len and std.mem.indexOfScalar(u32, marks, @intCast(next)) == null; // one row: a chunk's part as sent
}

/// Whether a pass at `start` to `rows` rebases the index rings where chunks one at a time do: at its start, or not.
pub fn ringsAgree(base: usize, ring: usize, start: usize, rows: usize) bool {
    return start + fw.prefill_rows > base + ring or start + rows <= base + ring;
}

/// The rows a solo pass takes at `start` of a `len`-id prompt.
fn rowsAt(start: usize, len: usize) usize {
    return @min(len - start, fw.solo_rows);
}

/// Rank 0's choice for the step at `start` (the others take the ids it sends); not beside what reads a chunk's rows.
pub fn at(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    if (e.bp.rows < fw.solo_rows or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    if (e.images) |v| if (v.rowCount() > 0) return false;
    if (!takesTwo(start, len, if (marks) |m| m.at else &.{})) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, rowsAt(start, len));
}

/// Whether a pass of `rows` past a chunk may run: `at`'s rule on every rank, its kept state restored.
pub fn allowed(e: *const Engine, start: usize, rows: usize) bool {
    if (e.bp.rows < rows or e.prompt_logprobs != null) return false;
    return ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rows);
}

/// decode._absorb_rows of a step, a chunk's rows at a time: the head's ring holds a window more than a pass.
pub fn mtpRows(e: *Engine, f: *fw.Forward, ids: []const i32) !void {
    const mp = f.prof;
    defer f.prof = mp;
    f.prof = null; // the head's pass as one section
    var done: usize = 0;
    while (done + 1 < ids.len) {
        const nxt = ids[done + 1 .. @min(ids.len, done + 1 + fw.prefill_rows)];
        try f.mtp(nxt, &.{e.bp.fnormed + done * e.c.hidden * 2});
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
        done += nxt.len;
    }
}

test "a solo pass: rows past the next grid point, and no mark at it" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, C + 2, &.{}));
    try std.testing.expect(!takesTwo(0, C + 1, &.{})); // its part would read as a chunk and the next id
    try std.testing.expect(!takesTwo(0, C, &.{})); // one chunk to the end
    try std.testing.expect(!takesTwo(C, 2 * C, &.{}));
    try std.testing.expect(takesTwo(2 * C, 5 * C, &.{}));
    try std.testing.expect(!takesTwo(0, 3 * C, &.{C})); // a state kept at the next chunk: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, &.{ 2 * C, 3 * C })); // marks past the pass
}

test "a solo pass rebases the rings where chunks one at a time do" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, 2 * C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, 2 * C)); // chunks rebase at the second, a pass at its first
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C + 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, 2 * C)); // both at the pass's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, 2 * C));
}

test "a solo pass fits a whole ring once it rebases, and its buffers only when asked" {
    try std.testing.expectEqual(@as(usize, fw.solo_rows), @import("cuda_state.zig").index_ring);
    try std.testing.expectEqual(@as(usize, fw.prefill_rows), bufferRows(false));
    try std.testing.expectEqual(@as(usize, fw.solo_rows), bufferRows(true));
}
