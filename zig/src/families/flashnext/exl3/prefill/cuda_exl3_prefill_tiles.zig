//! Dense EXL3 projections of a deep prefill on the prompt tiles (the hook cuda_exl3_pass.zig calls).

const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const prompt = @import("../../cuda_prompt.zig");

/// Set while a deep prefill runs with Options.tiles: cuda_exl3_pass.part sends prompt rows to Dense.prompt on tiles.
pub var tiles_on: bool = false;
/// Rows below which a projection keeps the reference kernels (tiles lose at a few rows).
const tiles_min_rows: usize = 64;

/// The hook in cuda_exl3_pass.part: true when it ran the part on the tiles (bf16 or fp32 out).
pub fn tilesPart(de: *cuda.exl3.Dense, s: cuda.Stream, w: cuda.exl3.Weight, x: u64, m: usize, y: u64, f32_out: bool) !bool {
    if (!tiles_on or m < tiles_min_rows) return false;
    _ = cuda.exl3.Dense.candidateWeight(w) catch return false;
    if (de.candidate_k == null or de.candidate_k.?.k_max < w.k) return false;
    de.implementation = .tiles;
    defer de.implementation = .reference;
    try de.prompt(s, w, x, .bf16, m, y, if (f32_out) .f32 else .bf16);
    return true;
}

/// Loads the tiles once with the widest input any EXL3 projection has (no allocation inside a pass).
pub fn warmTiles(e: *prompt.Engine) !void {
    const x = e.layers[0].exl3 orelse return;
    const de = try x.ops.linear();
    const k_max = @max(x.ops.max_k, wt.dims * 3);
    if (de.candidate_k != null and de.candidate_k.?.k_max >= k_max) return;
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    var mod = try cuda.Module.load(de.d, cuda.kernels.exl3_candidate);
    errdefer mod.unload();
    const k = try cuda.exl3_candidate.Kernels.resolve(de.d, mod, k_max);
    if (de.candidate_k) |*old| old.deinit();
    if (de.candidate_mod) |*old| old.unload();
    de.candidate_mod = mod;
    de.candidate_k = k;
}
