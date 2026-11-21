//! The dense engine's shared launches: the lane matmul, norms, embedding rows, bias, draws, paged attention, SwiGLU.
const std = @import("std");
const cuda = @import("cuda");
const recipe = @import("recipe.zig");
const pages = @import("pages.zig");

const Args = cuda.Args;
const Function = cuda.Function;
const Stream = cuda.Stream;
const Geometry = recipe.Geometry;

/// Streams one shared-window attention launch takes (paged_attn.cuh's MAXSEG).
pub const max_segs = 8;

/// paged_attn.cuh's AttnSegs: each stream's caches, position, page table and first row.
pub const AttnSegs = extern struct { kc: [max_segs]u64, vc: [max_segs]u64, meta: [max_segs]u64, pt: [max_segs]u64, ps: i64, row0: [max_segs]c_int, n: c_int };

const images = .{ "sample", "tree_round", "prefill_attention", "nemotron_attention", "lfm2" };

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [images.len]cuda.Module,
    g: Geometry,
    shape: recipe.Shape,
    lane: cuda.qmmf.Lane,
    norms: cuda.norms.Norms,
    embed: cuda.embed_rows.EmbedRows,
    vit: cuda.vit.Kernels,
    draws: cuda.sample.Draws,
    round: cuda.tree_round.Kernels,
    pattn: Function,
    kv_write: Function,
    chunk: Function,
    merge: Function,
    silu_mul: Function,

    /// The shared modules, and the attention launches `g`'s shape names; a geometry the kernels lack refuses by name.
    pub fn load(ctx: *const cuda.Context, g: Geometry) !Kernels {
        const d = ctx.d;
        const shape = try recipe.attentionShape(g);
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.g = g;
        k.shape = shape;
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        inline for (images, 0..) |name, i| {
            k.mods[i] = try cuda.Module.load(d, @field(cuda.kernels, name));
            loaded += 1;
        }
        const m = &k.mods;
        k.draws = try cuda.sample.Draws.resolve(m[0]);
        k.round = try cuda.tree_round.Kernels.resolve(m[1]);
        k.pattn = try m[2].function(shape.pattn);
        try k.pattn.allowDynamicShared(shape.shared);
        k.kv_write = try m[3].function("tf_nemo_kv_write_segs");
        k.chunk = try m[3].function(shape.chunk);
        k.merge = try m[3].function(shape.merge);
        k.silu_mul = try m[4].function("tf_lfm2_silu_mul");
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.vit = try cuda.vit.Kernels.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.vit.unload();
        k.embed.unload();
        k.norms.unload();
        k.lane.unload();
        for (&k.mods) |*m| m.unload();
    }

    fn go(f: Function, s: Stream, grid: [3]usize, block: usize, smem: u32, a: *Args) !void {
        const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = @intCast(block) }, .shared = smem }, s, a);
    }

    /// out [rows, f] = bf16(bf16(silu(gate)) * up) from rows gate | up [rows, 2 f] (lfm2.cu).
    pub fn siluMul(k: *const Kernels, s: Stream, gu: u64, out: u64, rows: usize, f: usize) !void {
        var a: Args = .{};
        a.add(gu);
        a.add(out);
        a.add(int(f));
        try go(k.silu_mul, s, .{ rows, (f + 255) / 256, 1 }, 256, 0, &a);
    }

    /// Rows k, v (kv [rows, 2 kv width]) into each stream's pages at its position on.
    pub fn kvWrite(k: *const Kernels, s: Stream, sg: AttnSegs, kv: u64, rows: usize) !void {
        const kvw = k.g.kvWidth();
        var a: Args = .{};
        a.add(sg);
        a.add(kv);
        a.add(int(2 * kvw));
        a.add(@as(c_int, 0));
        a.add(int(kvw));
        try go(k.kv_write, s, .{ rows, 1, 1 }, 256, 0, &a);
    }

    fn scale(k: *const Kernels) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(k.g.head_dim)));
    }

    /// Decode rows: q [rows, heads, dim] over each stream's pages in 512-key chunks, merged in order into out and xs.
    pub fn attend(k: *const Kernels, s: Stream, sg: AttnSegs, q: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, nch: usize, z: usize) !void {
        const g = k.g;
        var a: Args = .{};
        a.add(sg);
        a.add(q);
        for ([_]u64{ po, pm, pl }) |v| a.add(v);
        for ([_]usize{ g.qWidth(), g.heads, g.kv_heads, g.group(), pages.page_rows, nch }) |v| a.add(int(v));
        a.add(k.scale());
        try go(k.chunk, s, .{ rows, g.kv_heads, z }, 256, 0, &a);
        var b: Args = .{};
        b.add(sg);
        for ([_]u64{ po, pm, pl, out, xs }) |v| b.add(v);
        for ([_]usize{ g.heads, g.kv_heads, g.group(), pages.page_rows, nch }) |v| b.add(int(v));
        try go(k.merge, s, .{ rows, g.kv_heads, 1 }, 256, 0, &b);
    }

    /// A prompt chunk's rows at positions p0.. against its stream's pages (written through them): out [rows, H, dim].
    pub fn prompt(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, pt: u64, ps: usize) !void {
        const g = k.g;
        var a: Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, g.heads, g.kv_heads, g.group() }) |v| a.add(int(v));
        a.add(k.scale());
        a.add(pt);
        a.add(@as(i64, @intCast(ps)));
        try go(k.pattn, s, .{ k.shape.blocks(rows), k.shape.grid_y, 1 }, 256, k.shape.shared, &a);
    }
};

test "AttnSegs is paged_attn.cuh's struct, padded to 8 bytes: 304" {
    try std.testing.expectEqual(@as(usize, 304), @sizeOf(AttnSegs));
    try std.testing.expectEqual(@as(usize, 256), @offsetOf(AttnSegs, "ps"));
    try std.testing.expectEqual(@as(usize, 296), @offsetOf(AttnSegs, "n"));
}
