//! Unlimited-OCR-2's geometry (baidu/Unlimited-OCR-2), and the check that a checkpoint's config.json is that geometry.
const std = @import("std");

// The decoder: DeepSeek-V2's skeleton with plain multi-head attention, the first layer dense, the rest routed.
pub const hidden = 1280;
pub const layers = 12;
pub const heads = 10;
pub const head_dim = 128;
pub const dense_ffn = 6848;
pub const experts = 64;
pub const expert_ffn = 896;
pub const shared_ffn = 2 * expert_ffn;
pub const top_k = 6;
pub const vocab = 129280;
pub const eps: f32 = 1e-6;

// The attention every query reads: its own last 128 keys, and 12 remote blocks of 32 a router picks.
pub const block = 32;
pub const local_blocks = 4;
pub const local = block * local_blocks;
pub const read_blocks = 12;
pub const ranked_blocks = 16;
pub const router_dim = 128;
pub const router_floor: f32 = 1e-4;
/// The router's score scale: dash_scale_per_layer (128) over sqrt(index_dim).
pub const router_scale: f32 = 128.0 / @sqrt(@as(f32, router_dim));

// The rotary: 63 token frequencies at theta 1e4 resampled over the half, and one page pair at page mod 2.
pub const rope_token_pairs = 63;
pub const rope_token_theta: f64 = 10000.0;
pub const rope_page_mod = 2;

/// The page2d rotary's 63 token frequencies 1 / 10000^(i / 63) as torch's float32 pow gives them (bit patterns).
pub const inv_tok_bits = [rope_token_pairs]u32{
    0x3f800000, 0x3f5d2e59, 0x3f3f190e, 0x3f251b3a, 0x3f0ea665, 0x3ef67ef2, 0x3ed4f836,
    0x3eb800d6, 0x3e9efa07, 0x3e895aa1, 0x3e6d5835, 0x3e4d101f, 0x3e312c0b, 0x3e191319,
    0x3e044133, 0x3de48876, 0x3dc5732b, 0x3daa982c, 0x3d936440, 0x3d7eb076, 0x3d5c0c72,
    0x3d3e1e95, 0x3d2442d2, 0x3d0deb6d, 0x3cf53bdc, 0x3cd3e113, 0x3cb70faa, 0x3c9e29aa,
    0x3c88a69b, 0x3c6c2121, 0x3c4c035a, 0x3c3043d4, 0x3c184a75, 0x3c0393d9, 0x3be35cea,
    0x3bc47060, 0x3ba9b892, 0x3b92a311, 0x3b7d62a4, 0x3b5aec07, 0x3b3d2564, 0x3b236b86,
    0x3b0d3169, 0x3af3fa6f, 0x3ad2cb5d, 0x3ab61fb9, 0x3a9d5a5c, 0x3a87f37e, 0x3a6aeba2,
    0x3a4af7f3, 0x3a2f5ccc, 0x3a1782dc, 0x3a02e766, 0x39e232ec, 0x39c36ee9, 0x39a8da22,
    0x3991e2e1, 0x397c168a, 0x3959cd1a, 0x393c2d7d, 0x39229557, 0x390c785b, 0x38f2baaa,
};

pub fn invTok() [rope_token_pairs]f32 {
    var out: [rope_token_pairs]f32 = undefined;
    for (&out, inv_tok_bits) |*o, b| o.* = @bitCast(b);
    return out;
}

// The ids the prompt and the reply use.
pub const bos = 0;
pub const eos = 1;
pub const image_token = 128815;
pub const page_token = 128827;
/// "Multi page parsing." as the release's tokenizer splits it: the default instruction after the pages.
pub const instruction = [_]u32{ 37460, 4366, 76466, 16 };
/// Tokens a page may run to before the reply's page guard closes it (infer's max_page_tokens).
pub const max_page_tokens = 20000;
/// The longest sequence the publisher generates to (its max_length).
pub const max_context = 131072;

// The image encoder: SAM ViT-B's 12 blocks (global attention in four), its neck, then CLIP-L's 24 on SAM's map.
pub const base_size = 1024;
pub const tile_size = 640;
pub const max_tiles = 16;
pub const patch = 16;
pub const sam_width = 768;
pub const sam_layers = 12;
pub const sam_heads = 12;
pub const sam_head_dim = 64;
pub const sam_ffn = 3072;
pub const sam_window = 14;
pub const sam_global = [_]usize{ 2, 5, 8, 11 };
pub const sam_grid = base_size / patch;
pub const neck_width = 256;
pub const net2_width = 512;
pub const net3_width = 1024;
pub const sam_eps: f32 = 1e-6;
pub const clip_width = 1024;
pub const clip_layers = 24;
pub const clip_heads = 16;
pub const clip_head_dim = 64;
pub const clip_ffn = 4096;
pub const clip_positions = 257;
pub const clip_eps: f32 = 1e-5;
/// CLIP's rows then SAM's, side by side, into the projector.
pub const fused = clip_width + net3_width;

/// Feature rows a side: a 1024 view's 16, a 640 tile's 10 (SAM's grid after its two stride-2 convolutions).
pub fn side(size: usize) usize {
    return size / patch / 4;
}

/// Whether SAM block `i` attends globally rather than in 14 x 14 windows.
pub fn isGlobal(i: usize) bool {
    return std.mem.indexOfScalar(usize, &sam_global, i) != null;
}

/// Why `config.json` is not Unlimited-OCR-2's geometry, or null when it is.
pub fn problem(gpa: std.mem.Allocator, text: []const u8) !?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return "config.json is not JSON";
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return "config.json is not an object";
    const o = root.object;
    const want_ints = .{
        .{ "hidden_size", hidden },        .{ "num_hidden_layers", layers },    .{ "num_attention_heads", heads },
        .{ "num_key_value_heads", heads }, .{ "intermediate_size", dense_ffn }, .{ "moe_intermediate_size", expert_ffn },
        .{ "n_routed_experts", experts },  .{ "n_shared_experts", 2 },          .{ "num_experts_per_tok", top_k },
        .{ "first_k_dense_replace", 1 },   .{ "moe_layer_freq", 1 },            .{ "vocab_size", vocab },
        .{ "bos_token_id", bos },          .{ "eos_token_id", eos },
    };
    inline for (want_ints) |w| {
        const v = o.get(w[0]) orelse return "config.json lacks " ++ w[0];
        if (v != .integer or v.integer != w[1]) return w[0] ++ " is not Unlimited-OCR-2's";
    }
    const mt = o.get("model_type") orelse return "config.json has no model_type";
    if (mt != .string or !std.mem.eql(u8, mt.string, "unlimited-ocr")) return "model_type is not unlimited-ocr";
    const flags = .{ .{ "norm_topk_prob", false }, .{ "use_mla", false }, .{ "attention_bias", false }, .{ "tie_word_embeddings", false } };
    inline for (flags) |f| {
        if (o.get(f[0])) |v| if (v != .bool or v.bool != f[1]) return f[0] ++ " is not Unlimited-OCR-2's";
    }
    const strings = .{ .{ "topk_method", "greedy" }, .{ "scoring_func", "softmax" }, .{ "hidden_act", "silu" } };
    inline for (strings) |s| {
        const v = o.get(s[0]) orelse return "config.json lacks " ++ s[0];
        if (v != .string or !std.mem.eql(u8, v.string, s[1])) return s[0] ++ " is not Unlimited-OCR-2's";
    }
    if (o.get("routed_scaling_factor")) |v| if (number(v) != 1.0) return "routed_scaling_factor is not 1";
    if (o.get("rms_norm_eps")) |v| if (number(v) != 1e-6) return "rms_norm_eps is not 1e-6";
    const sp = (o.get("sparse_attention_config") orelse return "config.json has no sparse_attention_config");
    if (sp != .object) return "sparse_attention_config is not an object";
    const sp_ints = .{
        .{ "sparse_block_size", block },            .{ "sparse_local_block", local_blocks }, .{ "dash_topk_blocks", read_blocks },
        .{ "image_token_id", image_token },         .{ "page_token_id", page_token },        .{ "sparse_rope_page_pairs", 1 },
        .{ "sparse_rope_page_mod", rope_page_mod },
    };
    inline for (sp_ints) |w| {
        if (sp.object.get(w[0])) |v| {
            if (v != .integer or v.integer != w[1]) return "sparse_attention_config." ++ w[0] ++ " is not Unlimited-OCR-2's";
        } else return "sparse_attention_config lacks " ++ w[0];
    }
    const sp_strings = .{ .{ "sparse_main_rope_mode", "page2d" }, .{ "sparse_rope_mode", "page2d" }, .{ "dash_stage0", "head_cls" } };
    inline for (sp_strings) |s| {
        const v = sp.object.get(s[0]) orelse return "sparse_attention_config lacks " ++ s[0];
        if (v != .string or !std.mem.eql(u8, v.string, s[1])) return "sparse_attention_config." ++ s[0] ++ " is not Unlimited-OCR-2's";
    }
    const sp_numbers = .{ .{ "sparse_rope_tok_theta", rope_token_theta }, .{ "sparse_rope_page_theta", 256.0 }, .{ "dash_sigma", 1.0 } };
    inline for (sp_numbers) |s| {
        const v = sp.object.get(s[0]) orelse return "sparse_attention_config lacks " ++ s[0];
        if (number(v) != s[1]) return "sparse_attention_config." ++ s[0] ++ " is not Unlimited-OCR-2's";
    }
    const tok_mode = sp.object.get("sparse_main_rope_tok_freq_mode") orelse return "sparse_attention_config lacks sparse_main_rope_tok_freq_mode";
    if (tok_mode != .string or !std.mem.eql(u8, tok_mode.string, "resample")) return "sparse_attention_config.sparse_main_rope_tok_freq_mode is not resample";
    const per_layer = .{ .{ "dash_scale_per_layer", @as(f64, router_dim) }, .{ "dash_floor_per_layer", @as(f64, router_floor) } };
    inline for (per_layer) |pl| {
        const v = sp.object.get(pl[0]) orelse return "sparse_attention_config lacks " ++ pl[0];
        if (v != .array or v.array.items.len != layers) return "sparse_attention_config." ++ pl[0] ++ " is not one value a layer";
        for (v.array.items) |x| if (@abs(number(x) - pl[1]) > 1e-6 * pl[1]) return "sparse_attention_config." ++ pl[0] ++ " is not Unlimited-OCR-2's";
    }
    return null;
}

fn number(v: std.json.Value) f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => std.math.nan(f64),
    };
}

/// Fails with the reason when `config.json` is not Unlimited-OCR-2's geometry.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    if (try problem(gpa, text)) |why| {
        std.log.err("unlimited ocr: {s}", .{why});
        return error.UnsupportedConfig;
    }
}

test "uocr: the release's geometry is admitted and a changed one is refused with its reason" {
    const good =
        \\{"model_type":"unlimited-ocr","hidden_size":1280,"num_hidden_layers":12,"num_attention_heads":10,
        \\"num_key_value_heads":10,"intermediate_size":6848,"moe_intermediate_size":896,"n_routed_experts":64,
        \\"n_shared_experts":2,"num_experts_per_tok":6,"first_k_dense_replace":1,"moe_layer_freq":1,"vocab_size":129280,
        \\"bos_token_id":0,"eos_token_id":1,"norm_topk_prob":false,"use_mla":false,"attention_bias":false,
        \\"tie_word_embeddings":false,"topk_method":"greedy","scoring_func":"softmax","hidden_act":"silu",
        \\"routed_scaling_factor":1.0,"rms_norm_eps":1e-06,"sparse_attention_config":{"sparse_block_size":32,
        \\"sparse_local_block":4,"dash_topk_blocks":12,"image_token_id":128815,"page_token_id":128827,
        \\"sparse_rope_page_pairs":1,"sparse_rope_page_mod":2,"sparse_main_rope_mode":"page2d","sparse_rope_mode":"page2d",
        \\"dash_stage0":"head_cls","sparse_rope_tok_theta":10000.0,"sparse_rope_page_theta":256.0,"dash_sigma":1.0,
        \\"sparse_main_rope_tok_freq_mode":"resample","dash_scale_per_layer":[128,128,128,128,128,128,128,128,128,128,128,128],
        \\"dash_floor_per_layer":[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]}}
    ;
    try std.testing.expect(try problem(std.testing.allocator, good) == null);
    const bad = try std.mem.replaceOwned(u8, std.testing.allocator, good, "\"num_experts_per_tok\":6", "\"num_experts_per_tok\":8");
    defer std.testing.allocator.free(bad);
    try std.testing.expectEqualStrings("num_experts_per_tok is not Unlimited-OCR-2's", (try problem(std.testing.allocator, bad)).?);
}
