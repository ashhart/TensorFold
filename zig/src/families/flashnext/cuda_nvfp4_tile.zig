//! Flash Next's NVFP4 expert tile: prompt tiles from factRows() rows, the 16-pair decode form below; same bits.
const std = @import("std");
const cuda = @import("cuda");

/// Rows from which NVFP4 experts take the prompt tiles (TF_FLASHNEXT_FACT_ROWS, 0 never), read once at load.
pub fn factRows() usize {
    // the startup check refused every malformed value already, so a plain default stands here
    const r = @import("sys").environment.read(usize, "TF_FLASHNEXT_FACT_ROWS", 64) catch 64;
    return if (r == 0) std.math.maxInt(usize) else r;
}

/// Pairs a routed item holds for an n-row pass: the prompt tile's, an expert's weights read once an item, or 16.
pub fn size(k: anytype, n: usize) usize {
    return if (n >= k.forms.fact_rows) cuda.experts.promptTile(.nvfp4) else 16;
}

/// Prompt tiles may select the K32 pipeline; decode calls take the decode form in equal waves; same bits.
pub fn run(k: anytype, pt: usize, s: cuda.Stream, epi: cuda.experts.Epi, in: cuda.experts.Rows, l: cuda.experts.Layer, p: cuda.grouped.Plan, out: u64, items: usize, skip: i32) !void {
    if (pt != 16) if (k.expert_tile) |*tile| if (tile.takes(epi)) return tile.run(s, epi, in, l, p, out, items, skip);
    const launch = if (pt == 16) &cuda.experts_waves.run else &cuda.experts.Nvfp4Experts.runFact;
    return launch(&k.nvfp4, s, epi, in, l, p, out, items, skip);
}

test {
    _ = @import("cuda_nvfp4_tile_test.zig");
}
