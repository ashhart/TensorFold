//! DeepSeek V4.1's vision tower on Metal: the release's ViT (2D RoPE, full attention) and 3x3 aligner.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const glue = @import("glue.zig");

pub const Ref = glue.Ref;

/// The release's `vision_config` (and the text hidden width the aligner maps to); defaults are the shipped values.
pub const Config = struct {
    layers: u32 = 32,
    dim: u32 = 1024,
    heads: u32 = 16,
    inter: u32 = 2816,
    patch: u32 = 14,
    theta: f64 = 10000,
    ratio: u32 = 3,
    max_tokens: u32 = 1024,
    min_pixels: u32 = 295936,
    max_wh_ratio: ?u32 = null,
    out_dim: u32 = 5120,
    image_token_id: u32 = 129264,
    eps: f32 = 1e-6,

    pub fn headDim(c: Config) u32 {
        return c.dim / c.heads;
    }

    /// Values in a flattened patch: three channels of patch x patch.
    pub fn patchValues(c: Config) u32 {
        return 3 * c.patch * c.patch;
    }

    /// Values in an aligner input row: ratio x ratio patches of the tower's width.
    pub fn alignIn(c: Config) u32 {
        return c.dim * c.ratio * c.ratio;
    }

    /// The most patches an image can have: every token of the budget a ratio x ratio block.
    pub fn maxPatches(c: Config) u32 {
        return c.max_tokens * c.ratio * c.ratio;
    }

    /// The `vision_config` and `image_token_id` of a config.json (null: no tower); `out_dim` is the text hidden width.
    pub fn parse(a: std.mem.Allocator, text: []const u8, out_dim: u32) !?Config {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
        if (root != .object) return error.BadConfig;
        const v = (root.object.get("vision_config") orelse return null);
        if (v != .object) return error.BadConfig;
        var c: Config = .{ .out_dim = out_dim };
        const U = struct {
            fn num(o: std.json.ObjectMap, key: []const u8) ?f64 {
                const x = o.get(key) orelse return null;
                return switch (x) {
                    .integer => |i| @floatFromInt(i),
                    .float => |f| f,
                    else => null,
                };
            }
        };
        if (U.num(v.object, "num_hidden_layers")) |x| c.layers = @intFromFloat(x);
        if (U.num(v.object, "hidden_size")) |x| c.dim = @intFromFloat(x);
        if (U.num(v.object, "num_attention_heads")) |x| c.heads = @intFromFloat(x);
        if (U.num(v.object, "intermediate_size")) |x| c.inter = @intFromFloat(x);
        if (U.num(v.object, "patch_size")) |x| c.patch = @intFromFloat(x);
        if (U.num(v.object, "rope_theta")) |x| c.theta = x;
        if (U.num(v.object, "downsample_ratio")) |x| c.ratio = @intFromFloat(x);
        if (U.num(v.object, "max_image_tokens")) |x| c.max_tokens = @intFromFloat(x);
        if (U.num(v.object, "min_pixels")) |x| c.min_pixels = @intFromFloat(x);
        if (U.num(v.object, "max_wh_ratio")) |x| c.max_wh_ratio = @intFromFloat(x);
        if (U.num(root.object, "image_token_id")) |x| c.image_token_id = @intFromFloat(x);
        try c.check();
        return c;
    }

    pub fn check(c: Config) !void {
        if (c.headDim() != 64 or c.dim % c.heads != 0) return error.UnsupportedVision;
        if (c.layers == 0 or c.ratio == 0 or c.patch == 0 or c.inter == 0) return error.UnsupportedVision;
    }
};

const names = [_][:0]const u8{ "tf_dsv41_vis_epi", "tf_dsv41_vis_qkv", "tf_dsv41_vis_attn", "tf_dsv41_vis_swiglu", "tf_dsv41_vis_unshuffle", "tf_dsv41_vis_merge" };

/// The tower's kernels (deepseek_v41/vision.metal).
pub const Pipelines = struct {
    p: [names.len]mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.vision);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var out: Pipelines = undefined;
        for (names, 0..) |n, i| out.p[i] = try mtl.Pipeline.init(device, lib, n, false);
        return out;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.p) |x| x.deinit();
    }
};

fn pipe(p: *const Pipelines, k: usize) mtl.Pipeline {
    return p.p[k];
}

const k_epi = 0;
const k_qkv = 1;
const k_attn = 2;
const k_swiglu = 3;
const k_unshuffle = 4;
const k_merge = 5;

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

fn grid(e: mtl.ComputeEncoder, x: usize, y: usize, z: usize) void {
    e.dispatchThreads(mtl.Size.of(x, y, z), mtl.Size.of(@min(x, 32), 1, 1));
}

/// A linear's fp32 sums acc [rows][n] to bf16 y: + bias (bf16 [n]) when given, GELU when asked, then + res [rows][n].
pub fn epilogue(p: *const Pipelines, e: mtl.ComputeEncoder, acc: Ref, bias: ?Ref, res: ?Ref, y: Ref, rows: u32, n: u32, gelu: bool) void {
    e.setPipeline(pipe(p, k_epi));
    bind(e, &.{ acc, bias orelse acc, res orelse acc, y });
    e.setValue(extern struct { rows: u32, n: u32, gelu: u32, has_bias: u32, has_res: u32 }{ .rows = rows, .n = n, .gelu = @intFromBool(gelu), .has_bias = @intFromBool(bias != null), .has_res = @intFromBool(res != null) }, 4);
    grid(e, n, rows, 1);
}

/// The qkv sums acc [rows][3 * dim] to bf16 qkv with the bias; q and k rotated by cs [rows][head_dim / 2] (cos, sin).
pub fn qkv(p: *const Pipelines, e: mtl.ComputeEncoder, acc: Ref, bias: Ref, cs: Ref, out: Ref, rows: u32, heads: u32, hd: u32) void {
    e.setPipeline(pipe(p, k_qkv));
    bind(e, &.{ acc, bias, cs, out });
    e.setValue(extern struct { rows: u32, heads: u32, hd: u32 }{ .rows = rows, .heads = heads, .hd = hd }, 4);
    grid(e, hd / 2, rows, 3 * heads);
}

/// Full attention of every row over all rows, one simdgroup a (row, head): out [rows][dim] from qkv [rows][3 * dim].
pub fn attention(p: *const Pipelines, e: mtl.ComputeEncoder, qkv_: Ref, out: Ref, rows: u32, heads: u32) void {
    e.setPipeline(pipe(p, k_attn));
    bind(e, &.{ qkv_, out });
    e.setValue(extern struct { n: u32, heads: u32, scale: f32 }{ .n = rows, .heads = heads, .scale = 0.125 }, 2);
    e.dispatchGroups(mtl.Size.of(rows, heads, 1), mtl.Size.of(32, 1, 1));
}

/// The MLP's w1 sums acc [rows][2 * inter] to y [rows][inter] bf16: silu(gate) * up.
pub fn swiglu(p: *const Pipelines, e: mtl.ComputeEncoder, acc: Ref, y: Ref, rows: u32, inter: u32) void {
    e.setPipeline(pipe(p, k_swiglu));
    bind(e, &.{ acc, y });
    e.setValue(extern struct { rows: u32, inter: u32 }{ .rows = rows, .inter = inter }, 2);
    grid(e, inter, rows, 1);
}

/// The aligner's unfold: x [nh * nw][dim] to y [lh * lw][dim * r * r], the grid zero-padded to whole blocks.
pub fn unshuffle(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, y: Ref, nh: u32, nw: u32, dim: u32, r: u32) void {
    const lh = (nh + r - 1) / r;
    const lw = (nw + r - 1) / r;
    e.setPipeline(pipe(p, k_unshuffle));
    bind(e, &.{ x, y });
    e.setValue(extern struct { nh: u32, nw: u32, lh: u32, lw: u32, dim: u32, r: u32 }{ .nh = nh, .nw = nw, .lh = lh, .lw = lw, .dim = dim, .r = r }, 2);
    grid(e, dim * r * r, lh * lw, 1);
}

/// Stream rows [rows][4][dim] take feat's rows or the delimiters as `src` [rows] (int32) names them; others stay.
pub fn merge(p: *const Pipelines, e: mtl.ComputeEncoder, feat: Ref, delims: Ref, src: Ref, y: Ref, rows: u32, dim: u32) void {
    e.setPipeline(pipe(p, k_merge));
    bind(e, &.{ feat, delims, src, y });
    e.setValue(extern struct { dim: u32, rows: u32 }{ .dim = dim, .rows = rows }, 4);
    grid(e, dim, rows, 1);
}

/// The tower's kernels and the glue ones it shares (RMS norm, bf16 matvecs), loaded together.
pub const Kit = struct {
    v: Pipelines,
    g: glue.Pipelines,

    pub fn load(device: mtl.Device) !Kit {
        var v = try Pipelines.load(device);
        errdefer v.deinit();
        return .{ .v = v, .g = try glue.Pipelines.load(device, sources.dsv41.glue_tmp) };
    }

    pub fn deinit(k: *Kit) void {
        k.v.deinit();
        k.g.deinit();
    }
};

/// The 2D RoPE's (cos, sin) pairs [nh * nw][head_dim / 2]: the first half rotates by row, the second by column.
pub fn ropeTable(out: [][2]f32, nh: u32, nw: u32, hd: u32, theta: f64) void {
    const half = hd / 2;
    const m = half / 2;
    std.debug.assert(out.len == @as(usize, nh) * nw * half);
    var inv: [32]f32 = undefined;
    for (0..m) |i| inv[i] = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(half))));
    for (0..nh) |h| for (0..nw) |w| for (0..half) |j| {
        const pos: f32 = @floatFromInt(if (j < m) h else w);
        const angle: f64 = pos * inv[j % m];
        out[(h * nw + w) * half + j] = .{ @floatCast(@cos(angle)), @floatCast(@sin(angle)) };
    };
}

test "the release's vision_config parses to the defaults" {
    const text =
        \\{"image_token_id": 129264, "vision_config": {"num_hidden_layers": 32, "hidden_size": 1024, "num_attention_heads": 16,
        \\ "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
        \\ "min_pixels": 295936, "max_wh_ratio": null}}
    ;
    const c = (try Config.parse(std.testing.allocator, text, 5120)).?;
    try std.testing.expectEqual(Config{}, c);
    try std.testing.expect((try Config.parse(std.testing.allocator, "{}", 5120)) == null);
}

test "the 2D RoPE table: row angles first, column angles after, position 0 is the identity" {
    var t: [6 * 32][2]f32 = undefined;
    ropeTable(&t, 2, 3, 64, 10000);
    for (t[0..32]) |cs| try std.testing.expectEqual([2]f32{ 1, 0 }, cs);
    // patch (1, 2): the first 16 pairs rotate by 1 * inv_freq, the next 16 by 2 * inv_freq
    const at = (1 * 3 + 2) * 32;
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(@as(f64, 1.0)))), t[at][0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(@as(f64, 2.0)))), t[at + 16][0], 1e-7);
}
