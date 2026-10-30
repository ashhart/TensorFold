//! Flash Next's NVFP4 expert tile: prompt tiles from factRows() rows, the 16-pair decode form below; same bits.
const std = @import("std");
const cuda = @import("cuda");

var fact_rows: ?usize = null;

/// Rows from which NVFP4 experts take the prompt tiles (TF_FLASHNEXT_FACT_ROWS, 0 never): same bits, only speed.
fn factRows() usize {
    if (fact_rows == null) {
        const r = if (std.c.getenv("TF_FLASHNEXT_FACT_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 64 else 64;
        fact_rows = if (r == 0) std.math.maxInt(usize) else r;
    }
    return fact_rows.?;
}

/// Pairs a routed item holds for an n-row pass: the prompt tile's, an expert's weights read once an item, or 16.
pub fn size(n: usize) usize {
    return if (n >= factRows()) cuda.experts.promptTile(.nvfp4) else 16;
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
