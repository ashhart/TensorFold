//! Per-row pocket scales over the landed image, position and projection paths.
const E = @import("cuda_engine.zig");
pub fn stageScales(e: *E.Engine, parts: []const E.Part, total: usize) !void {
    if (e.pockets.any() and e.learning == null) {
        var scales: [E.max_rows]f32 = undefined;
        var row: usize = 0;
        for (parts) |p| {
            @memset(scales[row..][0..p.rows], if (p.cache.untaught) 0 else 1);
            row += p.rows;
        }
        try e.pockets.uploadScales(e.stream, scales[0..total]);
        e.taught = e.pockets.rowsBuffer();
    }
}
