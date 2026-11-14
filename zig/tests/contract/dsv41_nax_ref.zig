//! DeepSeek V4.1's tensor-unit FP8 prompt rows (fp8_nax) for the contract: the k16-step depth and sums.
const contract = @import("contract");

const c = contract.spec;
const rows = contract.rows;
const mma_step = contract.attention_range.mma_step;

/// One fp32 sum over K on the tensor units: an mma_step of |acc| + sum |ab| for every k16 step, nothing after.
pub fn depth(s: c.Shape) usize {
    return mma_step * (s.k / 16);
}

/// A sum as the tensor units take it: each 16 terms (summed in `order`) onto one running sum, in k order.
pub fn steps(acc: c.Accum, order: rows.Order, t: []const f64, part: []f64) f64 {
    const n = (t.len + 15) / 16;
    for (part[0..n], 0..) |*p, i| p.* = rows.sumIn(acc, order, t[16 * i .. @min(t.len, 16 * i + 16)]);
    return rows.sumIn(acc, .sequential, part[0..n]);
}
