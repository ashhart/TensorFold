//! Indexed, scalar and explicitly masked prompt kernels on their production source and argument layouts.
const std = @import("std");
const metal = @import("metal");
const src = @import("kernel_sources");
const held = @import("masked_sources");
const frags = @import("tensorfold").frags;
const data = @import("masked_value_data.zig");
const rig = @import("rig.zig");
const mla = @import("masked_attention_mla.zig");
const sparse = @import("masked_attention_sparse.zig");
pub const Kind = enum { glm_dense, glm_nax64, glm_nax32, glm_padding, glm_split, glm_indexed64, glm_indexed32, glm_dsa, kimi, step_full_nax, step_sliding_nax, step_sparse_nax, step_full_scalar, step_sliding_scalar, step_sparse_scalar, qwen4_dense, qwen4_sparse, drama, shared, shared_split, qwen35_pages, lfm3, lfm3_sliding, step_padding_nax, step_padding_scalar, deepseek_pre, h3_padding };
pub fn geometry(k: Kind) data.Geometry {
    return switch (k) {
        .glm_nax32, .glm_indexed32 => .{ .heads = 32, .kv = 1, .dim = 512, .latent = true },
        .glm_padding => .{ .heads = 64, .kv = 1, .dim = 512, .latent = true, .poison_key = 0, .skip_zero_first = true },
        .glm_dense, .glm_nax64, .glm_split, .glm_indexed64 => .{ .heads = 64, .kv = 1, .dim = 512, .latent = true },
        .glm_dsa => .{ .heads = 32, .kv = 1, .dim = 512, .latent = true },
        .kimi => .{ .heads = 96, .kv = 1, .dim = 512, .latent = true },
        .step_full_nax, .step_full_scalar => .{ .heads = 64, .kv = 4, .dim = 192 },
        .step_sliding_nax, .step_sliding_scalar => .{ .heads = 64, .kv = 4, .dim = 192, .cap = 2560, .window = 512, .ring = 2560 },
        .step_padding_nax, .step_padding_scalar => .{ .heads = 64, .kv = 4, .dim = 192, .cap = 8192, .first = 4104, .selected_prefix = 4096, .selected_shift_first = 8, .poison_key = 0 },
        .step_sparse_nax, .step_sparse_scalar => .{ .heads = 64, .kv = 4, .dim = 192, .cap = 8192, .first = 4104, .selected_prefix = 4096 },
        .qwen4_dense, .qwen4_sparse => .{ .heads = 24, .kv = 4, .dim = 256 },
        .drama => .{ .heads = 16, .kv = 2, .dim = 256, .gated = true },
        .shared, .shared_split => .{ .heads = 16, .kv = 4, .dim = 128 },
        .qwen35_pages => .{ .heads = 8, .kv = 2, .dim = 256, .cache = .head, .query = .head },
        .lfm3 => .{ .heads = 16, .kv = 4, .dim = 128, .cache = .head },
        .lfm3_sliding => .{ .heads = 16, .kv = 4, .dim = 128, .cache = .head, .window = 2048 },
        .deepseek_pre => .{ .heads = 64, .kv = 1, .dim = 512, .window = 128, .latent = true, .sink = true },
        .h3_padding => .{ .heads = 2, .kv = 2, .dim = 128, .cap = 1024, .cache = .head },
    };
}
pub fn label(k: Kind) []const u8 {
    return @tagName(k);
}
pub fn known(k: Kind, poison: data.Poison) bool {
    return switch (k) {
        .glm_nax64, .glm_nax32, .glm_padding, .glm_split, .step_padding_nax, .h3_padding => true,
        .step_full_nax, .step_sliding_nax, .step_sparse_nax => poison == .inf,
        else => false,
    };
}
pub fn finding(k: Kind) []const u8 {
    return switch (k) {
        .glm_padding => "#1009: padding reads unselected latent key 0 at zero weight; visible NaN scores are also dropped",
        .glm_nax64, .glm_nax32, .glm_split => "#1009: a visible exceptional latent key gives NaN scores in binary64, but the tensor path drops it",
        .step_padding_nax => "#1009: padding reads unselected V at key 0 with zero weight; visible Inf also multiplies zero residual P parts",
        .step_full_nax, .step_sliding_nax, .step_sparse_nax => "#1009: zero residual probability parts multiply visible Inf, giving NaN instead of the IEEE infinity",
        .h3_padding => "#1009: V in a tile's SIZES padding enters P x V at zero probability and gives a hidden output NaN",
        else => unreachable,
    };
}
pub fn output(k: Kind) @import("contract").spec.Elem {
    return if (k == .kimi or k == .glm_dsa) .f32 else .bf16;
}
pub fn available(k: Kind) bool {
    return if (k == .lfm3 or k == .lfm3_sliding) held.lfm3_attention.len > 0 else true;
}

pub fn load(p: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(p.a);
    defer arena.deinit();
    const a = arena.allocator();
    var text: []const u8 = "";
    var names: []const []const u8 = &.{};
    switch (p.kind) {
        .glm_dense => {
            text = src.glm_attn;
            names = &.{ "glm_latent_scores", "glm_latent_values" };
            p.pipes[2] = try rig.pipeline(p.rig.device, src.ops_softmax, "tf_softmax_bf16");
        },
        .glm_nax64, .glm_nax32, .glm_padding, .glm_split => {
            text = try std.fmt.allocPrint(a, "#define GLM_HEADS {d}\n{s}", .{ p.geo.heads, src.glm_sparse_nax });
            names = if (p.kind == .glm_split) &.{ "glm_sparse_split", "glm_sparse_combine" } else &.{"glm_sparse_nax"};
        },
        .glm_indexed64, .glm_indexed32 => {
            const key = if (p.kind == .glm_indexed32) "sparse_attention_tp" else "sparse_attention";
            for (src.glm.all) |s| if (std.mem.eql(u8, s.key, key)) {
                text = s.source;
                names = s.functions;
            };
        },
        .glm_dsa => {
            text = src.glm_dsa_mla;
            names = &.{ "glm_mla_attend", "glm_mla_merge" };
        },
        .kimi => {
            text = try std.mem.replaceOwned(u8, a, held.kimi_mla, "#include \"kimi_common.h\"", held.kimi_common);
            names = &.{ "k3_mla_attend", "k3_mla_merge" };
        },
        .step_full_nax, .step_sliding_nax, .step_sparse_nax, .step_padding_nax => {
            text = src.step5_attend_nax;
            names = &.{"s5_attend_nax"};
        },
        .step_full_scalar, .step_sliding_scalar, .step_sparse_scalar, .step_padding_scalar => {
            text = src.step5_sparse;
            names = &.{"s5_attend"};
        },
        .qwen4_dense, .qwen4_sparse => {
            text = try @import("qwen4").family.metal_attention.source(a, .{ .heads = 24, .kv = 4, .index_heads = 4, .index_dim = 128, .rotary = 64, .ratio = 2, .top = 512 });
            names = &.{ "qwen4_attn_parts", "qwen4_attn_merge" };
        },
        .drama => {
            text = try std.mem.replaceOwned(u8, a, src.drama3.attn, "#include \"d3_common.h\"", src.drama3.common);
            names = &.{"d3_attn"};
        },
        .shared, .shared_split => {
            text = try std.fmt.allocPrint(a, "#define TF_QH 16\n#define TF_KVH 4\n#define TF_D 128\n#define TF_OUT bfloat\n{s}", .{src.core_shared_attention});
            names = if (p.kind == .shared_split) &.{ "tf_shared_lane_attention_split", "tf_shared_lane_attention_merge" } else &.{"tf_shared_lane_attention"};
        },
        .qwen35_pages => {
            text = src.qwen35_paged;
            names = &.{"qwen35_attn_partial_paged"};
            for (src.qwen35.all) |s| if (std.mem.eql(u8, s.key, "attn_merge")) {
                p.pipes[1] = try rig.pipeline(p.rig.device, s.source, s.function);
            };
        },
        .lfm3, .lfm3_sliding => {
            text = held.lfm3_attention;
            names = &.{ "lfm3_attn_partial", "lfm3_attn_merge" };
        },
        .deepseek_pre => {
            text = src.dsv41.csa2_attn;
            names = &.{"tf_dsv41_attn_pre_bf16"};
        },
        .h3_padding => {
            text = try std.mem.concat(a, u8, &.{ "#define H3_HEADS 2\n#define H3_SLOTS 1024\n#define H3_TILES 16\n", @import("h3_kernels").source });
            names = &.{"h3_attention"};
        },
    }
    const body = try frags.source(p.rig.device, a, text);
    var opts = metal.CompileOptions.mlx();
    if (std.mem.startsWith(u8, @tagName(p.kind), "step_") or p.kind == .drama) opts.math_functions = .precise;
    const lib = try metal.Library.fromSource(p.rig.device, body, opts);
    defer lib.deinit();
    for (names, 0..) |name, i| p.pipes[i] = try metal.Pipeline.init(p.rig.device, lib, name, false);
}

pub fn prepare(p: anytype) !void {
    const d = p.fixture;
    p.q = try p.rig.input(std.mem.sliceAsBytes(d.q));
    p.k = try p.rig.input(std.mem.sliceAsBytes(d.k));
    p.v = try p.rig.input(std.mem.sliceAsBytes(d.v));
    p.out = try p.rig.output(d.count() * @import("contract").spec.size(output(p.kind)));
    switch (p.kind) {
        .glm_dsa, .kimi => try mla.prepare(p),
        else => try sparse.prepare(p),
    }
}
pub fn encode(p: anytype, e: metal.ComputeEncoder) void {
    switch (p.kind) {
        .glm_dsa, .kimi => mla.encode(p, e),
        else => sparse.encode(p, e),
    }
}
pub fn reference(p: anytype, want: []f64, mag: []f64) void {
    if (p.kind != .h3_padding) return p.fixture.reference(want, mag);
    // A tile has 17 visible keys; head 0's poison is padding, head 1's poison is a visible key.
    const d = p.fixture;
    for (0..d.rows) |r| for (0..2) |h| for (0..128) |col| {
        var numerator: f64 = 0;
        for (0..17) |key| numerator += if (d.poisoned and h == 1 and key == 0 and col == 7)
            @as(f64, if (d.poison == .inf) std.math.inf(f64) else std.math.nan(f64))
        else
            1;
        const i = d.outputAt(r, h, col);
        want[i] = numerator / 17;
        mag[i] = @abs(want[i]);
    };
}
