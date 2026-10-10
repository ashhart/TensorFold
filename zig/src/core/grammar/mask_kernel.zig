//! The grammar mask kernel (PTX from Zig's NVPTX backend): grid (ceil(span / 128), rows), a thread per 32-token word.

const rows = @import("rows.zig");

/// Logits rows idx[r] (stride ld) under bits row r: this rank's `n` words from `w0`, `width` columns.
export fn tf_grammar_mask(logits: [*]addrspace(.global) f32, ld: u64, idx: [*]addrspace(.global) const u32, bits: [*]addrspace(.global) const u32, words: u32, w0: u32, n: u32, width: u32) callconv(.kernel) void {
    const r = @workGroupId(1);
    const k = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (k >= n) return;
    const word = bits[@as(u64, r) * words + w0 + k];
    const base: [*]f32 = @addrSpaceCast(logits + @as(u64, idx[r]) * ld);
    rows.maskWord(base, word, 32 * k, width);
}
