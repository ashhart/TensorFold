//! Decisions' label scores on CUDA: the prompt through a fresh cache, then its last row's label logits and logsumexp.
const std = @import("std");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");

/// One logit per label at the row after `ids` (bf16 widened), and the row's logsumexp in f64 when `mass` (else 0).
pub fn labels(comptime Tri: type, e: *prompt.Engine, tri: Tri, context: usize, ids: []const u32, wanted: []const u32, logits: []f64, mass: bool) !f64 {
    // every refusal before a cache is made
    if (ids.len == 0 or ids.len > context or ids.len > prompt.max_tokens) return error.PromptOutsideContext;
    if (wanted.len == 0 or wanted.len != logits.len) return error.UnexpectedTensor;
    for (wanted) |t| if (t >= prompt.vocabulary) return error.LabelOutsideVocabulary;
    var seq = try e.newSeq(ids.len);
    defer seq.deinit(e.gpa);
    try e.forward(Tri, tri, &seq, ids, .{});
    const row = std.mem.bytesAsSlice(u16, seq.logits[0 .. prompt.vocabulary * 2]);
    for (wanted, logits) |t, *v| {
        v.* = mtp.promote(std.mem.littleToNative(u16, row[t]));
        if (!std.math.isFinite(v.*)) return error.NonFiniteScore;
    }
    return if (mass) logsumexp(row) else 0;
}

/// The bf16 row's logsumexp in f64; a non-finite row is refused.
fn logsumexp(row: []align(1) const u16) !f64 {
    var top: f64 = -std.math.inf(f64);
    for (row) |b| {
        const v: f64 = mtp.promote(std.mem.littleToNative(u16, b));
        if (!std.math.isFinite(v)) return error.NonFiniteScore;
        top = @max(top, v);
    }
    var sum: f64 = 0;
    for (row) |b| sum += @exp(@as(f64, mtp.promote(std.mem.littleToNative(u16, b))) - top);
    return top + @log(sum);
}

test "the logsumexp of a bf16 row in f64, and a non-finite row refused" {
    const one: u16 = 0x3f80; // 1.0
    const two: u16 = 0x4000; // 2.0
    const row = [_]u16{ one, two, one };
    const want = 2.0 + @log(1.0 + 2.0 * @exp(@as(f64, -1.0)));
    try std.testing.expectApproxEqRel(want, try logsumexp(&row), 1e-15);
    try std.testing.expectError(error.NonFiniteScore, logsumexp(&[_]u16{ one, 0x7f80 }));
}
