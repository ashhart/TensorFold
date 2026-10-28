//! Conservative up-front reserve for the learner, leaving serving slots unable to spend its future allocations.
const std = @import("std");
const Config = @import("config.zig").Config;
const memory = @import("prompt_memory.zig");

pub fn bytes(g: Config, frame: u32, tensor_units: bool) !u64 {
    const R: u64 = 1024;
    const M: u64 = 512;
    const D: u64 = g.hidden;
    const I: u64 = g.intermediate;
    const vi: u64 = g.gdnValueDim();
    const conv: u64 = g.gdnQkvDim();
    const qi: u64 = g.heads * g.head_dim;
    const kv: u64 = g.kvDim();
    const ki: u64 = g.k_heads * g.dk;
    const proj = conv + vi + 2 * g.v_heads;
    var n: u64 = R * g.layers * D * 2; // Every layer's pocket input.
    n += R * D * 10 + R * g.vocab * 6 + 8 * 3 * g.layers * D * 8 + R * 16;
    n += R * (3 * I + @max(vi, qi)) * 4;
    n += M * (proj + conv + vi) * 4;
    n += g.v_heads * 4 * 4 * 32 * 128 * 4;
    n += M * g.v_heads * 4 * (128 * 2 + 2) * 4;
    n += M * (qi * 4 + kv * 4) * 4 + 2 * g.heads * M * M * 4;
    n += 3 * conv * 2 + q4(g.vocab, D);
    for (g.layers / 5..g.layers) |l| {
        n += R * (2 * D + 2 * I) * 2 + 2 * q4(I, D) + q4(D, I);
        if (g.kind(l) == .linear) {
            n += R * (proj * 2 + conv * 2 + 2 * ki * 4 + 2 * g.v_heads * 4 + vi * 2);
            n += (R / 4 + 8) * g.v_heads * g.dv * g.dk * 4;
            n += q4(proj, D) + q4(D, vi);
        } else {
            n += R * (2 * qi + 2 * kv + qi + 2 * kv + qi) * 2;
            n += q4(2 * qi, D) + 2 * q4(kv, D) + q4(D, qi);
        }
    }
    n += try memory.runnerBytes(g, M + 1, 8, frame, tensor_units, 0);
    // Scalar copies, Metal pipeline state and allocator rounding stay inside this explicit spare.
    return std.math.add(u64, n, 512 << 20);
}
fn q4(n: u64, k: u64) u64 {
    return n * k / 2 + n * k / 16;
}
