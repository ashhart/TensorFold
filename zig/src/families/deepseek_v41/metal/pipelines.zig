//! Every DeepSeek V4.1 Metal kernel a layer launches, compiled once at the config's literal shapes.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const Config = @import("dsv41").config.Config;
pub const fp8 = @import("fp8_rows.zig");
pub const fp8_mma = @import("fp8_mma.zig");
pub const attn = @import("csa2_attn.zig");
pub const index = @import("index.zig");
pub const experts = @import("experts.zig");
pub const glue = @import("glue.zig");
pub const compress = @import("compress.zig");
pub const engram = @import("engram.zig");
pub const cand = @import("cand.zig");
pub const xmma = @import("experts_mma.zig");
pub const fused = @import("fused.zig");
pub const hc = @import("hc.zig");
pub const glue_k = @import("glue_k.zig");
pub const fold = @import("fold.zig");

/// Rows a decode launch carries at most (the row kernels' widest tier).
pub const max_rows = 16;

/// The shapes the kernels fix at compile time: a config that differs is refused before anything is built.
pub fn check(c: *const Config) !void {
    if (c.head_dim != attn.head_dim or c.rope_dim != attn.rope_dims or c.window != attn.window) return error.UnsupportedAttention;
    if (c.index_heads != index.heads or c.index_dim != index.dim or c.index_topk != attn.max_selected) return error.UnsupportedIndexer;
    if (c.candidate_block_size != index.block) return error.UnsupportedIndexer;
    if (c.topk != experts.topk or c.experts > 1024 or c.experts < c.topk) return error.UnsupportedExperts;
    if (c.mtp_layers > 0 and c.dspark_experts > 0) {
        if (c.dspark_experts > c.experts or c.dspark_experts % 8 != 0 or c.dspark_topk == 0 or c.dspark_topk > c.topk) return error.UnsupportedDspark;
    }
    if (c.hidden > experts.hidden or c.expert_width > 3072 or c.shared_experts != 1) return error.UnsupportedExperts;
    if (c.hc_mult != 4 or c.q_lora % 32 != 0 or c.o_lora % 32 != 0) return error.UnsupportedShape;
    if (c.heads % c.o_groups != 0 or (c.heads / c.o_groups * c.head_dim) % 32 != 0) return error.UnsupportedShape;
    const eng_k = c.hashCols() * c.engram_head_dim;
    if (c.engram_layers.len > 0 and (c.engram_head_dim != engram.head or c.engram_gate_clamp != 1e-6 or fp8.tilesOf(eng_k, 1) > fp8.max_tiles)) return error.UnsupportedEngram;
}

pub const Pipelines = struct {
    fp8: fp8.Pipelines,
    mma: fp8_mma.Pipelines,
    attn: attn.Pipelines,
    index: index.Pipelines,
    experts: experts.Pipelines,
    glue: glue.Pipelines,
    compress: compress.Pipelines,
    engram: engram.Pipelines,
    route: mtl.Pipeline,
    /// the DSpark blocks' route at their own expert count and top-k (null without blocks)
    route_dspark: ?mtl.Pipeline,
    topk512: mtl.Pipeline,
    cand: mtl.Pipeline,
    xmma: xmma.Pipelines,
    fused: fused.Pipelines,
    hc: hc.Pipelines,
    glue_k: glue_k.Pipelines,
    fold: fold.Pipelines,

    pub fn load(a: std.mem.Allocator, device: mtl.Device, c: *const Config) !Pipelines {
        try check(c);
        var p: Pipelines = undefined;
        p.fp8 = try fp8.Pipelines.load(device);
        errdefer p.fp8.deinit();
        p.mma = try fp8_mma.Pipelines.load(device);
        errdefer p.mma.deinit();
        p.attn = try attn.Pipelines.load(device);
        errdefer p.attn.deinit();
        p.index = try index.Pipelines.load(device);
        errdefer p.index.deinit();
        p.experts = try experts.Pipelines.load(a, device);
        errdefer p.experts.deinit();
        p.glue = try glue.Pipelines.load(device, sources.dsv41.glue_tmp);
        errdefer p.glue.deinit();
        p.compress = try compress.Pipelines.load(device);
        errdefer p.compress.deinit();
        p.engram = try engram.Pipelines.load(device);
        errdefer p.engram.deinit();
        p.route = try routeAt(a, device, c.experts, c.topk);
        errdefer p.route.deinit();
        p.route_dspark = if (c.mtp_layers > 0 and c.dspark_experts > 0) try routeAt(a, device, c.dspark_experts, c.dspark_topk) else null;
        errdefer if (p.route_dspark) |r| r.deinit();
        p.topk512 = try one(device, sources.dsv41.topk512, "tf_topk512");
        errdefer p.topk512.deinit();
        p.cand = try cand.load(device);
        errdefer p.cand.deinit();
        p.xmma = try xmma.Pipelines.load(a, device);
        errdefer p.xmma.deinit();
        p.fused = try fused.Pipelines.load(device);
        errdefer p.fused.deinit();
        p.hc = try hc.Pipelines.load(device);
        errdefer p.hc.deinit();
        p.glue_k = try glue_k.Pipelines.load(device);
        errdefer p.glue_k.deinit();
        p.fold = try fold.Pipelines.load(device);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        p.fp8.deinit();
        p.mma.deinit();
        p.attn.deinit();
        p.index.deinit();
        p.experts.deinit();
        p.glue.deinit();
        p.compress.deinit();
        p.engram.deinit();
        p.route.deinit();
        if (p.route_dspark) |r| r.deinit();
        p.topk512.deinit();
        p.cand.deinit();
        p.xmma.deinit();
        p.fused.deinit();
        p.hc.deinit();
        p.glue_k.deinit();
        p.fold.deinit();
    }
};

/// The route kernel compiled for `e` experts and `topk` picks.
fn routeAt(a: std.mem.Allocator, device: mtl.Device, e: u32, topk: u32) !mtl.Pipeline {
    const text = try std.fmt.allocPrint(a, "#define TF_DSV41_E {d}\n#define TF_DSV41_TOPK {d}\n{s}", .{ e, topk, sources.dsv41.route });
    defer a.free(text);
    return one(device, text, "tf_dsv41_route_select");
}

fn one(device: mtl.Device, text: []const u8, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}
