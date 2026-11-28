//! The indexer's top-512 of long rows on Metal: chunk bounds over the row, then one group a row over the candidates.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Keys a bound covers: one a lane of a simdgroup (topk512_bounded.metal's TF_CH).
pub const chunk = 32;
/// Rows past this many visible keys bound their chunks (TF_LONG): 512 chunks or more, so the 512th bound exists.
pub const long = 16384;
/// Chunks a bounds group takes: one a simdgroup.
const simdgroups = 4;

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Pipelines = struct {
    /// each chunk's largest key: four simdgroups a group, a chunk each
    bounds: mtl.Pipeline,
    /// a row's top-512 from its bounds' candidates: tf_topk512's indices
    select: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.topk512_bounded);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const bounds = try mtl.Pipeline.init(device, lib, "tf_topk512_bounds", false);
        errdefer bounds.deinit();
        return .{ .bounds = bounds, .select = try mtl.Pipeline.init(device, lib, "tf_topk512_bounded", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.bounds.deinit();
        p.select.deinit();
    }
};

/// Bounds a row of `width` scores keeps.
pub fn chunksOf(width: u32) u32 {
    return (width + chunk - 1) / chunk;
}

/// tf_topk512's selected [rows][512] of scores [rows][width], visible[row] keys a row; bounds [rows][chunksOf(width)].
pub fn encode(p: *const Pipelines, e: mtl.ComputeEncoder, scores: Ref, visible: Ref, selected: Ref, bounds: Ref, rows: u32, width: u32) void {
    const chunks = chunksOf(width);
    e.setPipeline(p.bounds);
    e.setValue([2]u32{ width, chunks }, 0);
    for ([_]Ref{ scores, visible, selected, bounds }, 1..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.dispatchGroups(mtl.Size.of((chunks + simdgroups - 1) / simdgroups, rows, 1), mtl.Size.of(32 * simdgroups, 1, 1));
    e.barrier();
    e.setPipeline(p.select);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

test "a row's bounds cover its keys a chunk of 32 at a time, and a long row has at least 512 of them" {
    try std.testing.expectEqual(@as(u32, 1), chunksOf(1));
    try std.testing.expectEqual(@as(u32, 1), chunksOf(32));
    try std.testing.expectEqual(@as(u32, 2), chunksOf(33));
    try std.testing.expectEqual(@as(u32, 513), chunksOf(long + 1));
    try std.testing.expectEqual(@as(u32, 32768), chunksOf(1048576));
}
