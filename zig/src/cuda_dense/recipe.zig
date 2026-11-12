//! A dense decoder's recipe contract: its geometry, the decls the engine reads, and the attention launch shape.
const std = @import("std");

/// A dense GQA decoder's sizes, read from its checkpoint's config before the weights load.
pub const Geometry = struct {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize, // 64 or 128
    vocab: usize,
    eps: f32,
    context: usize, // the checkpoint's own window
    acc: usize, // q|k|v width: a fused projection's f32 sums
    wide: usize, // gate|up width
    mid: usize, // the MLP's inner width
    theta: ?f64 = null, // 1-D RoPE base (rope.zig's inverse frequencies on the device); null: no rotary positions
    inv_freq: ?[]const f32 = null, // the reference's own inverse frequencies, when its pow is not correctly rounded

    pub fn qWidth(g: Geometry) usize {
        return g.heads * g.head_dim;
    }

    pub fn kvWidth(g: Geometry) usize {
        return g.kv_heads * g.head_dim;
    }

    pub fn group(g: Geometry) usize {
        return g.heads / g.kv_heads;
    }
};

/// Query heads a KV head the decode kernels take at most (paged_attn.cuh's GMAX).
pub const max_group = 16;

/// The attention launches of one geometry: prompt pattn and its block, decode chunk and merge.
pub const Shape = struct {
    pattn: [:0]const u8,
    chunk: [:0]const u8,
    merge: [:0]const u8,
    hpc: usize, // query heads a pattn block takes from one KV head
    rows: usize, // prompt rows a pattn block takes
    grid_y: usize, // pattn blocks over the heads
    shared: u32, // pattn's dynamic shared bytes

    pub fn blocks(s: Shape, rows: usize) usize {
        return (rows + s.rows - 1) / s.rows;
    }
};

fn pattnSymbol(comptime d: usize, comptime hpc: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN20tf_prefill_attention12pattn_kernelILi{d}ELi8ELi{d}ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", .{ d, hpc });
}

/// The pattn instances prefill_attention.cu builds at eight warps and eight slots: head dim, query heads a block.
const Instance = struct { d: usize, hpc: usize, symbol: [:0]const u8 };
const instances = [_]Instance{
    .{ .d = 128, .hpc = 8, .symbol = pattnSymbol(128, 8) },
    .{ .d = 128, .hpc = 2, .symbol = pattnSymbol(128, 2) },
    .{ .d = 64, .hpc = 4, .symbol = pattnSymbol(64, 4) },
    .{ .d = 64, .hpc = 2, .symbol = pattnSymbol(64, 2) },
};

/// pattn's shared bytes: eight staging slots of 32 keys, a key row in 16-byte chunks (prefill_attention_rows.cuh).
fn pattnShared(d: usize) u32 {
    const chunks = (2 * ((d + 15) / 16) + 7) / 8 * 8;
    return @intCast(8 * 32 * chunks * 16);
}

/// The launches for `g`: the largest built pattn instance whose heads a block divide the group; refusals by name.
pub fn attentionShape(g: Geometry) error{ UnsupportedDenseHeadDim, UnsupportedDenseGroup }!Shape {
    if (g.head_dim != 64 and g.head_dim != 128) return error.UnsupportedDenseHeadDim;
    if (g.kv_heads == 0 or g.heads % g.kv_heads != 0 or g.group() > max_group) return error.UnsupportedDenseGroup;
    const group_n = g.group();
    for (instances) |in| if (in.d == g.head_dim and group_n % in.hpc == 0) return .{
        .pattn = in.symbol,
        .chunk = if (g.head_dim == 128) "tf_nemo_attn_chunk_segs" else "tf_paged_attn_chunk_segs_64",
        .merge = if (g.head_dim == 128) "tf_nemo_attn_merge_segs" else "tf_paged_attn_merge_segs_64",
        .hpc = in.hpc,
        .rows = 16 * (8 / in.hpc),
        .grid_y = g.kv_heads * (group_n / in.hpc),
        .shared = pattnShared(g.head_dim),
    };
    return error.UnsupportedDenseGroup;
}

fn need(comptime R: type, comptime name: []const u8) void {
    if (!@hasDecl(R, name)) @compileError("a cuda_dense recipe declares " ++ name);
}

fn absent(comptime R: type, comptime name: []const u8) void {
    if (@hasDecl(R, name) and @field(R, name) != null) @compileError(name ++ " is not built in the dense engine yet: set it null");
}

/// The decls a recipe gives the engine, at compile time; the capabilities not built yet must stay null.
pub fn check(comptime R: type) void {
    const required = .{ "name", "model_type", "formats", "default_context", "Weights", "Kernels", "geometry", "load", "weightBytes", "embed", "layer", "final", "head" };
    inline for (required) |name| need(R, name);
    inline for (.{ "post_embed", "Tower", "scores", "propose" }) |name| absent(R, name);
}

fn shapeOf(heads: usize, kv_heads: usize, head_dim: usize) !Shape {
    return attentionShape(.{ .hidden = heads * head_dim, .layers = 1, .heads = heads, .kv_heads = kv_heads, .head_dim = head_dim, .vocab = 1, .eps = 1e-6, .context = 1, .acc = 1, .wide = 1, .mid = 1 });
}

test "attention shapes: dots 12/2 x 128, 16/8 x 128, 16/8 x 64 and 32/8 x 64" {
    const dots = try shapeOf(12, 2, 128);
    try std.testing.expectEqual(@as(usize, 2), dots.hpc);
    try std.testing.expectEqual(@as(usize, 64), dots.rows);
    try std.testing.expectEqual(@as(usize, 6), dots.grid_y);
    try std.testing.expectEqual(@as(u32, 65536), dots.shared);
    try std.testing.expectEqualStrings("_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", dots.pattn);
    try std.testing.expectEqualStrings("tf_nemo_attn_chunk_segs", dots.chunk);
    try std.testing.expectEqual(@as(usize, 2), dots.blocks(65));
    const glm = try shapeOf(16, 8, 128);
    try std.testing.expect(glm.hpc == 2 and glm.rows == 64 and glm.grid_y == 8 and glm.shared == 65536);
    const parse = try shapeOf(16, 8, 64);
    try std.testing.expect(parse.hpc == 2 and parse.rows == 64 and parse.grid_y == 8 and parse.shared == 32768);
    try std.testing.expectEqualStrings("_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", parse.pattn);
    try std.testing.expectEqualStrings("tf_paged_attn_merge_segs_64", parse.merge);
    const lfm = try shapeOf(32, 8, 64);
    try std.testing.expect(lfm.hpc == 4 and lfm.rows == 32 and lfm.grid_y == 8 and lfm.shared == 32768);
    try std.testing.expectEqualStrings("_ZN20tf_prefill_attention12pattn_kernelILi64ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifPKix", lfm.pattn);
}

test "attention shapes refuse groups past 16, odd groups, uneven heads and head dims other than 64 and 128" {
    try std.testing.expectError(error.UnsupportedDenseGroup, shapeOf(34, 2, 128));
    try std.testing.expectError(error.UnsupportedDenseGroup, shapeOf(12, 4, 128));
    try std.testing.expectError(error.UnsupportedDenseGroup, shapeOf(12, 5, 64));
    try std.testing.expectError(error.UnsupportedDenseGroup, shapeOf(8, 8, 64));
    try std.testing.expectError(error.UnsupportedDenseHeadDim, shapeOf(16, 8, 80));
    try std.testing.expectError(error.UnsupportedDenseHeadDim, shapeOf(16, 8, 256));
    try std.testing.expectEqual(@as(usize, 2), (try shapeOf(32, 8, 128)).hpc); // group 4 at head 128: no 4-head block
    try std.testing.expectEqual(@as(usize, 8), (try shapeOf(32, 2, 128)).hpc);
    try std.testing.expectEqual(@as(usize, 8), (try shapeOf(64, 8, 128)).hpc);
}
