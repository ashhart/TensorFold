//! DeepSeek-V4.1 hand-off geometry and CUDA record constants; `guard_test.zig` fails when the engine's constants drift.
const std = @import("std");

comptime {
    if (@import("builtin").cpu.arch.endian() != .little) @compileError("the hand-off's bf16/f32 parts are little-endian host memory");
}

/// CUDA comp row / SWA ring row: 448 e4m3 NoPE codes then 64 bf16 RoPE values (576 B), then 8 UE8M0 scale bytes.
pub const cuda_values_bytes = 576;
pub const cuda_scale_bytes = 8;
pub const cuda_record_bytes = cuda_values_bytes + cuda_scale_bytes; // 584: the pool's comp row
pub const cuda_nope = 448;
pub const cuda_tile = 64; // NoPE dims a UE8M0 scale covers; scale byte 7 is padding (the kernel writes 0)
pub const cuda_tiles = cuda_nope / cuda_tile;
/// CUDA index row: 128 e4m3 codes then one little-endian fp32 power-of-two scale.
pub const cuda_index_fp8_bytes = 132;
pub const cuda_index_bf16_bytes = 256;

pub const Source = struct { layer: u32, ratio: u32 };

pub const Geometry = struct {
    /// backbone layers (each has a SWA ring)
    layers: u32 = 40,
    /// DSpark blocks after the backbone (layers `layers ..`); their rings ride only when the drafter's rows are given
    mtp_layers: u32 = 3,
    window: u32 = 128,
    head_dim: u32 = 512,
    rope_dim: u32 = 64,
    index_dim: u32 = 128,
    /// tokens a pool page (the residue unit of TF_DSV41_KV_SPLIT)
    page: u32 = 256,
    sources: []const Source = &release_sources,
    /// Engram's lookback ids: max_ngram - 1
    lookback: u32 = 3,

    pub fn ratioOf(g: Geometry, layer: u32) ?u32 {
        for (g.sources) |s| if (s.layer == layer) return s.ratio;
        return null;
    }

    pub fn sourceIndex(g: Geometry, layer: u32) ?u32 {
        for (g.sources, 0..) |s, i| if (s.layer == layer) return @intCast(i);
        return null;
    }

    /// Compressed rows a source holds after `upto` tokens: one per completed group.
    pub fn rowsOf(g: Geometry, source: u32, upto: u64) u64 {
        return upto / g.sources[source].ratio;
    }

    pub fn pageRows(g: Geometry, source: u32) u32 {
        return g.page / g.sources[source].ratio;
    }

    pub fn check(g: Geometry) error{BadGeometry}!void {
        if (g.head_dim != cuda_nope + g.rope_dim or g.rope_dim != 64 or g.index_dim != 128) return error.BadGeometry;
        if (g.layers == 0 or g.layers > 64 or g.mtp_layers > 8 or g.window == 0 or g.sources.len == 0 or g.lookback == 0) return error.BadGeometry;
        if (g.page == 0 or !std.math.isPowerOfTwo(g.page)) return error.BadGeometry;
        var last: ?u32 = null;
        for (g.sources) |s| {
            if (s.layer >= g.layers or s.ratio == 0 or s.ratio > 2 or g.page % s.ratio != 0) return error.BadGeometry;
            if (last) |l| if (s.layer <= l) return error.BadGeometry;
            last = s.layer;
        }
    }
};

const release_sources = [_]Source{ .{ .layer = 2, .ratio = 2 }, .{ .layer = 8, .ratio = 2 }, .{ .layer = 14, .ratio = 2 }, .{ .layer = 20, .ratio = 1 } };

/// The release: 40 layers + 3 DSpark blocks, KV sources 2, 8, 14 (ratio 2) and 20 (ratio 1), window 128.
pub const release: Geometry = .{};

/// The CUDA ring a hand-off reads (ring rows are a power of two; position p lives in row p % rows).
pub const Rings = struct { rows: u32 = 256, drafter_rows: u32 = 256 };

test "the release geometry is valid and prices the release's rows" {
    try release.check();
    try std.testing.expectEqual(@as(u64, 50), release.rowsOf(0, 101));
    try std.testing.expectEqual(@as(u64, 101), release.rowsOf(3, 101));
    try std.testing.expectEqual(@as(u32, 128), release.pageRows(1));
    try std.testing.expectEqual(@as(u32, 256), release.pageRows(3));
    try std.testing.expectEqual(@as(?u32, 2), release.sourceIndex(14));
    try std.testing.expectEqual(@as(?u32, null), release.ratioOf(3));
    var bad = release;
    bad.page = 100;
    try std.testing.expectError(error.BadGeometry, bad.check());
}
