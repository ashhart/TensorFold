//! Clean-room Metal Vision Encoders for VLM inference (zero Apple MLX dependency).
//!
//! Supported VLM vision towers:
//!   - Qwen3.5-VL / Qwen3.8-VL (`vision_tower.*` MLX layout & `model.visual.*` HF layout):
//!     Still-image 3D-collapsed patch embed, 2D bilinear pos_embed interpolation, 2D Vision RoPE,
//!     LayerNorm + bidirectional MHA + gelu_pytorch_tanh MLP, 2x2 spatial patch merger (exact GELU).
//!   - Gemma 3-VL SigLIP (`vision_tower.vision_model.*` + `multi_modal_projector.*`):
//!     2D patch embed + learned pos_embed, LayerNorm + bidirectional MHA + gelu_pytorch_tanh MLP,
//!     post_layernorm, AvgPool2d + (+1)-centered Gemma3RMSNorm + multimodal linear projector.
const std = @import("std");
const shaders = @import("shaders");
const mtl = @import("metal4.zig");
const st = @import("safetensors.zig");
const img = @import("image.zig");
const tok_mod = @import("tokenizer.zig");

const Buffer = mtl.Buffer;
const Pipeline = mtl.Pipeline;

pub const VisionKind = enum {
    none,
    qwen35_vl,
    gemma3_siglip,
};

pub const VisionConfig = struct {
    kind: VisionKind = .none,
    hidden_size: usize = 768,
    intermediate_size: usize = 3072,
    depth: usize = 12,
    num_heads: usize = 12,
    patch_size: usize = 16,
    temporal_patch_size: usize = 2,
    spatial_merge_size: usize = 2,
    num_position_embeddings: usize = 2304,
    image_size: usize = 896, // SigLIP fixed square size
    mm_tokens_per_image: usize = 256, // SigLIP pooled tokens
    out_hidden_size: usize = 1024,
    layer_norm_eps: f32 = 1e-6,
    image_token_id: u32 = 248056,
    vision_start_token_id: u32 = 248053,
    vision_end_token_id: u32 = 248054,
    min_pixels: u64 = 65536,
    max_pixels: u64 = 1_048_576, // default cap ~1024x1024 for fast, sharp local inference

    pub fn headDim(self: VisionConfig) usize {
        return self.hidden_size / self.num_heads;
    }

    pub fn load(a: std.mem.Allocator, model_dir: []const u8) !?VisionConfig {
        const cfg_path = try std.fs.path.join(a, &.{ model_dir, "config.json" });
        defer a.free(cfg_path);
        const raw = st.mapFile(a, cfg_path) catch return null;
        defer st.unmapFile(raw);

        const parsed = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        const root = parsed.value.object;

        const vc_val = root.get("vision_config") orelse return null;
        if (vc_val != .object) return null;
        const vc = vc_val.object;

        const root_mt = if (root.get("model_type")) |v| (if (v == .string) v.string else "") else "";
        const vis_mt = if (vc.get("model_type")) |v| (if (v == .string) v.string else "") else "";

        const kind: VisionKind = if (std.mem.indexOf(u8, vis_mt, "siglip") != null or std.mem.indexOf(u8, root_mt, "gemma3") != null)
            .gemma3_siglip
        else if (std.mem.indexOf(u8, vis_mt, "qwen") != null or std.mem.indexOf(u8, root_mt, "qwen3_5") != null or std.mem.indexOf(u8, root_mt, "qwen3_vl") != null)
            .qwen35_vl
        else
            return null;

        const tc_obj: ?std.json.ObjectMap = if (root.get("text_config")) |tv| (if (tv == .object) tv.object else null) else null;
        const text_hidden = if (tc_obj) |tc| getUsize(tc, "hidden_size", 1024) else getUsize(root, "hidden_size", 1024);

        var out = VisionConfig{ .kind = kind };
        if (kind == .qwen35_vl) {
            out.hidden_size = getUsize(vc, "hidden_size", 768);
            out.intermediate_size = getUsize(vc, "intermediate_size", 3072);
            out.depth = getUsize(vc, "depth", getUsize(vc, "num_hidden_layers", 12));
            out.num_heads = getUsize(vc, "num_heads", getUsize(vc, "num_attention_heads", 12));
            out.patch_size = getUsize(vc, "patch_size", 16);
            out.temporal_patch_size = getUsize(vc, "temporal_patch_size", 2);
            out.spatial_merge_size = getUsize(vc, "spatial_merge_size", 2);
            out.num_position_embeddings = getUsize(vc, "num_position_embeddings", 2304);
            out.out_hidden_size = getUsize(vc, "out_hidden_size", text_hidden);
            out.layer_norm_eps = getF32(vc, "layer_norm_eps", 1e-6);
            out.image_token_id = @intCast(getUsize(root, "image_token_id", 248056));
            out.vision_start_token_id = @intCast(getUsize(root, "vision_start_token_id", 248053));
            out.vision_end_token_id = @intCast(getUsize(root, "vision_end_token_id", 248054));

            // Optional preprocessor_config.json for min_pixels
            const pp_path = try std.fs.path.join(a, &.{ model_dir, "preprocessor_config.json" });
            defer a.free(pp_path);
            if (st.mapFile(a, pp_path)) |pp_raw| {
                defer st.unmapFile(pp_raw);
                if (std.json.parseFromSlice(std.json.Value, a, pp_raw, .{})) |pp_p| {
                    defer pp_p.deinit();
                    if (pp_p.value == .object) {
                        if (pp_p.value.object.get("size")) |sz| {
                            if (sz == .object) {
                                const se = getUsize(sz.object, "shortest_edge", 65536);
                                if (se > 0) out.min_pixels = se;
                            }
                        }
                    }
                } else |_| {}
            } else |_| {}
        } else {
            // Gemma 3-VL SigLIP
            out.hidden_size = getUsize(vc, "hidden_size", 1152);
            out.intermediate_size = getUsize(vc, "intermediate_size", 4304);
            out.depth = getUsize(vc, "num_hidden_layers", getUsize(vc, "depth", 27));
            out.num_heads = getUsize(vc, "num_attention_heads", getUsize(vc, "num_heads", 16));
            out.patch_size = getUsize(vc, "patch_size", 14);
            out.image_size = getUsize(vc, "image_size", 896);
            out.mm_tokens_per_image = getUsize(root, "mm_tokens_per_image", 256);
            out.out_hidden_size = text_hidden;
            out.layer_norm_eps = getF32(vc, "layer_norm_eps", 1e-6);
            out.image_token_id = @intCast(getUsize(root, "image_token_index", getUsize(root, "image_token_id", 262144)));
            out.vision_start_token_id = @intCast(getUsize(root, "boi_token_index", 255999));
            out.vision_end_token_id = @intCast(getUsize(root, "eoi_token_index", 256000));
        }

        return out;
    }
};

fn getUsize(obj: std.json.ObjectMap, key: []const u8, default: usize) usize {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else default,
        .float => |f| if (f >= 0) @intFromFloat(f) else default,
        else => default,
    };
}

fn getF32(obj: std.json.ObjectMap, key: []const u8, default: f32) f32 {
    const v = obj.get(key) orelse return default;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => default,
    };
}

inline fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

inline fn f32ToBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% 0x7FFF +% lsb) >> 16);
}

inline fn f16BitsToF32(b: u16) f32 {
    const h: f16 = @bitCast(b);
    return @floatCast(h);
}

const VitLnParams = extern struct {
    rows: u32,
    dim: u32,
    eps: f32,
};

const VitLinearParams = extern struct {
    rows: u32,
    N: u32,
    K: u32,
    has_bias: u32,
    act: u32,
    add_res: u32,
};

const VitAddParams = extern struct {
    count: u32,
};

const VitRopeParams = extern struct {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
};

const VitAttnParams = extern struct {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
};

const VitPoolNormParams = extern struct {
    in_grid: u32,
    out_grid: u32,
    k_pool: u32,
    dim: u32,
    eps: f32,
};

const VitBlockWeights = struct {
    norm1_w: Buffer,
    norm1_b: Buffer,
    qkv_w: Buffer,
    qkv_b: Buffer,
    proj_w: Buffer,
    proj_b: Buffer,
    norm2_w: Buffer,
    norm2_b: Buffer,
    fc1_w: Buffer,
    fc1_b: Buffer,
    fc2_w: Buffer,
    fc2_b: Buffer,
};

pub const EncodedImage = struct {
    /// Output vision embeddings [num_tokens * hidden_size] BF16 (in LLM hidden space).
    embs: []u16,
    num_tokens: u32,
    hidden_size: u32,
    grid_h: u32,
    grid_w: u32,

    pub fn deinit(self: EncodedImage, a: std.mem.Allocator) void {
        a.free(self.embs);
    }
};

pub const VisionEncoder = struct {
    a: std.mem.Allocator,
    ctx: *mtl.Context,
    c: VisionConfig,
    lib: mtl.Library,
    p_ln: Pipeline,
    p_lin: Pipeline,
    p_add: Pipeline,
    p_rope: Pipeline,
    p_attn: Pipeline,
    p_pool_norm: Pipeline,

    bufs: std.ArrayList(Buffer) = .empty,
    blocks: []VitBlockWeights,
    patch_w: Buffer,
    patch_b: Buffer,
    patch_has_bias: bool = false,
    patch_k: usize,

    // Qwen3.5-VL specific
    qwen_pos_f32: []f32 = &.{},
    merger_norm_w: ?Buffer = null,
    merger_norm_b: ?Buffer = null,
    merger_postshuffle: bool = false,
    merger_fc1_w: ?Buffer = null,
    merger_fc1_b: ?Buffer = null,
    merger_fc2_w: ?Buffer = null,
    merger_fc2_b: ?Buffer = null,

    // Gemma 3-VL SigLIP specific
    siglip_pos_w: ?Buffer = null,
    post_ln_w: ?Buffer = null,
    post_ln_b: ?Buffer = null,
    gemma_norm_w_p1: ?Buffer = null,
    gemma_proj_w: ?Buffer = null,

    /// Attempt to load a vision tower from `model_dir`. Returns `null` cleanly for text-only models.
    pub fn tryInit(a: std.mem.Allocator, ctx: *mtl.Context, model_dir: []const u8) !?VisionEncoder {
        const vc = (try VisionConfig.load(a, model_dir)) orelse return null;

        var ckpt = st.Checkpoint.openDir(a, model_dir) catch return null;
        defer ckpt.close();

        // Detect weight prefix (MLX vs HuggingFace)
        const v_prefix: []const u8 = switch (vc.kind) {
            .qwen35_vl => blk: {
                if (ckpt.getOptional("vision_tower.patch_embed.proj.weight") != null) break :blk "vision_tower";
                if (ckpt.getOptional("model.visual.patch_embed.proj.weight") != null) break :blk "model.visual";
                if (ckpt.getOptional("visual.patch_embed.proj.weight") != null) break :blk "visual";
                return null;
            },
            .gemma3_siglip => blk: {
                if (ckpt.getOptional("vision_tower.vision_model.embeddings.patch_embedding.weight") != null) break :blk "vision_tower.vision_model";
                if (ckpt.getOptional("model.vision_tower.vision_model.embeddings.patch_embedding.weight") != null) break :blk "model.vision_tower.vision_model";
                return null;
            },
            .none => return null,
        };

        const m_prefix: []const u8 = if (ckpt.getOptional("multi_modal_projector.mm_soft_emb_norm.weight") != null)
            "multi_modal_projector"
        else
            "model.multi_modal_projector";

        const lib = try ctx.compileSource(shaders.vision, .fast);
        errdefer lib.deinit();

        var enc = VisionEncoder{
            .a = a,
            .ctx = ctx,
            .c = vc,
            .lib = lib,
            .p_ln = try ctx.makePipeline(lib, "vit_layer_norm"),
            .p_lin = try ctx.makePipeline(lib, "vit_linear_act"),
            .p_add = try ctx.makePipeline(lib, "vit_add_bf16"),
            .p_rope = try ctx.makePipeline(lib, "vit_apply_rope_qk"),
            .p_attn = try ctx.makePipeline(lib, "vit_bidir_attn"),
            .p_pool_norm = try ctx.makePipeline(lib, "vit_gemma3_pool_rmsnorm"),
            .blocks = try a.alloc(VitBlockWeights, vc.depth),
            .patch_w = undefined,
            .patch_b = undefined,
            .patch_k = 3 * vc.patch_size * vc.patch_size,
        };
        errdefer enc.deinit();

        if (vc.kind == .qwen35_vl) {
            try enc.loadQwenWeights(&ckpt, v_prefix);
        } else {
            try enc.loadGemma3Weights(&ckpt, v_prefix, m_prefix);
        }

        ctx.commitResidency();
        return enc;
    }

    pub fn deinit(self: *VisionEncoder) void {
        if (self.qwen_pos_f32.len > 0) self.a.free(self.qwen_pos_f32);
        self.a.free(self.blocks);
        for (self.bufs.items) |b| self.ctx.freeBuffer(b);
        self.bufs.deinit(self.a);
        self.p_ln.deinit();
        self.p_lin.deinit();
        self.p_add.deinit();
        self.p_rope.deinit();
        self.p_attn.deinit();
        self.p_pool_norm.deinit();
        self.lib.deinit();
    }

    fn allocUploadBf16(self: *VisionEncoder, data: []const u16) !Buffer {
        const b = try self.ctx.allocBuffer(data.len * 2);
        try self.bufs.append(self.a, b);
        @memcpy(b.slice(u16, data.len), data);
        return b;
    }

    fn allocZerosBf16(self: *VisionEncoder, count: usize) !Buffer {
        const b = try self.ctx.allocBuffer(@max(count, 1) * 2);
        try self.bufs.append(self.a, b);
        @memset(b.ptr[0..b.len], 0);
        return b;
    }

    fn loadTensorBf16(self: *VisionEncoder, ckpt: *const st.Checkpoint, name: []const u8) ![]u16 {
        const t = try ckpt.get(name);
        return dequantOrConvertToBf16(self.a, ckpt, name, t);
    }

    fn uploadTensorBf16(self: *VisionEncoder, ckpt: *const st.Checkpoint, name: []const u8) !Buffer {
        const vals = try self.loadTensorBf16(ckpt, name);
        defer self.a.free(vals);
        return self.allocUploadBf16(vals);
    }

    fn uploadOptionalBiasBf16(self: *VisionEncoder, ckpt: *const st.Checkpoint, name: []const u8, len: usize) !Buffer {
        if (ckpt.getOptional(name)) |t| {
            const vals = try dequantOrConvertToBf16(self.a, ckpt, name, t);
            defer self.a.free(vals);
            return self.allocUploadBf16(vals);
        }
        return self.allocZerosBf16(len);
    }

    fn loadQwenWeights(self: *VisionEncoder, ckpt: *const st.Checkpoint, vp: []const u8) !void {
        const c = self.c;
        const p2 = c.patch_size * c.patch_size;
        const k_dim = 3 * p2;
        var nb: [256]u8 = undefined;

        // 1. Patch embed Conv3d weight -> collapsed still-image [V_H, 3 * P * P] in [C, Ph, Pw] order
        const pw_name = try std.fmt.bufPrint(&nb, "{s}.patch_embed.proj.weight", .{vp});
        const pw_t = try ckpt.get(pw_name);
        const pw_raw = try dequantOrConvertToBf16(self.a, ckpt, pw_name, pw_t);
        defer self.a.free(pw_raw);

        const collapsed = try self.a.alloc(u16, c.hidden_size * k_dim);
        defer self.a.free(collapsed);

        if (pw_t.rank == 5) {
            // Detect HF [out, C, T, Ph, Pw] vs MLX [out, T, Ph, Pw, C]
            const is_hf = (pw_t.shape[1] == 3 and pw_t.shape[2] == c.temporal_patch_size);
            const t_dim = c.temporal_patch_size;
            const ps = c.patch_size;
            for (0..c.hidden_size) |o| {
                for (0..3) |ch| {
                    for (0..ps) |ph| {
                        for (0..ps) |pw| {
                            var sum: f32 = 0.0;
                            for (0..t_dim) |ti| {
                                const src_idx = if (is_hf)
                                    ((((o * 3 + ch) * t_dim + ti) * ps + ph) * ps + pw)
                                else
                                    ((((o * t_dim + ti) * ps + ph) * ps + pw) * 3 + ch);
                                sum += bf16ToF32(pw_raw[src_idx]);
                            }
                            collapsed[o * k_dim + ch * p2 + ph * ps + pw] = f32ToBf16(sum);
                        }
                    }
                }
            }
        } else if (pw_t.rank == 2) {
            @memcpy(collapsed, pw_raw[0 .. c.hidden_size * k_dim]);
        } else {
            return error.InvalidCheckpoint;
        }
        self.patch_w = try self.allocUploadBf16(collapsed);

        const pb_name = try std.fmt.bufPrint(&nb, "{s}.patch_embed.proj.bias", .{vp});
        self.patch_has_bias = (ckpt.getOptional(pb_name) != null);
        self.patch_b = try self.uploadOptionalBiasBf16(ckpt, pb_name, c.hidden_size);

        // 2. Learned 2D pos_embed [num_pos, V_H] kept on CPU in FP32 for exact bilinear interpolation
        const pos_name = try std.fmt.bufPrint(&nb, "{s}.pos_embed.weight", .{vp});
        const pos_bf16 = try self.loadTensorBf16(ckpt, pos_name);
        defer self.a.free(pos_bf16);
        self.qwen_pos_f32 = try self.a.alloc(f32, pos_bf16.len);
        for (pos_bf16, 0..) |v, idx| self.qwen_pos_f32[idx] = bf16ToF32(v);

        // 3. ViT blocks
        for (0..c.depth) |li| {
            self.blocks[li] = .{
                .norm1_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.norm1.weight", .{ vp, li })),
                .norm1_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.norm1.bias", .{ vp, li }), c.hidden_size),
                .qkv_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.attn.qkv.weight", .{ vp, li })),
                .qkv_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.attn.qkv.bias", .{ vp, li }), 3 * c.hidden_size),
                .proj_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.attn.proj.weight", .{ vp, li })),
                .proj_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.attn.proj.bias", .{ vp, li }), c.hidden_size),
                .norm2_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.norm2.weight", .{ vp, li })),
                .norm2_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.norm2.bias", .{ vp, li }), c.hidden_size),
                .fc1_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.mlp.linear_fc1.weight", .{ vp, li })),
                .fc1_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.mlp.linear_fc1.bias", .{ vp, li }), c.intermediate_size),
                .fc2_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.mlp.linear_fc2.weight", .{ vp, li })),
                .fc2_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.blocks.{d}.mlp.linear_fc2.bias", .{ vp, li }), c.hidden_size),
            };
        }

        // 4. 2x2 Spatial Patch Merger
        const merged_dim = c.hidden_size * c.spatial_merge_size * c.spatial_merge_size;
        const mn_w_name = try std.fmt.bufPrint(&nb, "{s}.merger.norm.weight", .{vp});
        const mn_w_vals = try self.loadTensorBf16(ckpt, mn_w_name);
        defer self.a.free(mn_w_vals);
        self.merger_postshuffle = (mn_w_vals.len == merged_dim);
        self.merger_norm_w = try self.allocUploadBf16(mn_w_vals);
        self.merger_norm_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.merger.norm.bias", .{vp}), mn_w_vals.len);
        self.merger_fc1_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.merger.linear_fc1.weight", .{vp}));
        self.merger_fc1_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.merger.linear_fc1.bias", .{vp}), merged_dim);
        self.merger_fc2_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.merger.linear_fc2.weight", .{vp}));
        self.merger_fc2_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.merger.linear_fc2.bias", .{vp}), c.out_hidden_size);
    }

    fn loadGemma3Weights(self: *VisionEncoder, ckpt: *const st.Checkpoint, vp: []const u8, mp: []const u8) !void {
        const c = self.c;
        const p2 = c.patch_size * c.patch_size;
        const k_dim = 3 * p2;
        var nb: [256]u8 = undefined;

        // 1. Patch embed Conv2d weight -> [V_H, 3 * P * P] in [C, Ph, Pw] order
        const pw_name = try std.fmt.bufPrint(&nb, "{s}.embeddings.patch_embedding.weight", .{vp});
        const pw_t = try ckpt.get(pw_name);
        const pw_raw = try dequantOrConvertToBf16(self.a, ckpt, pw_name, pw_t);
        defer self.a.free(pw_raw);

        const patch_out = try self.a.alloc(u16, c.hidden_size * k_dim);
        defer self.a.free(patch_out);
        if (pw_t.rank == 4) {
            // HF: [out, C, Ph, Pw]; MLX: [out, Ph, Pw, C]
            const is_hf = (pw_t.shape[1] == 3 and pw_t.shape[2] == c.patch_size);
            const ps = c.patch_size;
            for (0..c.hidden_size) |o| {
                for (0..3) |ch| {
                    for (0..ps) |ph| {
                        for (0..ps) |pw| {
                            const src_idx = if (is_hf)
                                (((o * 3 + ch) * ps + ph) * ps + pw)
                            else
                                (((o * ps + ph) * ps + pw) * 3 + ch);
                            patch_out[o * k_dim + ch * p2 + ph * ps + pw] = pw_raw[src_idx];
                        }
                    }
                }
            }
        } else {
            @memcpy(patch_out, pw_raw[0 .. c.hidden_size * k_dim]);
        }
        self.patch_w = try self.allocUploadBf16(patch_out);
        const pb_name = try std.fmt.bufPrint(&nb, "{s}.embeddings.patch_embedding.bias", .{vp});
        self.patch_has_bias = (ckpt.getOptional(pb_name) != null);
        self.patch_b = try self.uploadOptionalBiasBf16(ckpt, pb_name, c.hidden_size);

        // 2. Fixed square position_embedding [grid*grid, V_H]
        self.siglip_pos_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.embeddings.position_embedding.weight", .{vp}));

        // 3. SigLIP encoder layers (fuse q_proj, k_proj, v_proj vertically into qkv [3*V_H, V_H])
        const qkv_w_tmp = try self.a.alloc(u16, 3 * c.hidden_size * c.hidden_size);
        defer self.a.free(qkv_w_tmp);
        const qkv_b_tmp = try self.a.alloc(u16, 3 * c.hidden_size);
        defer self.a.free(qkv_b_tmp);
        const mat_elems = c.hidden_size * c.hidden_size;

        for (0..c.depth) |li| {
            const qw = try self.loadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.q_proj.weight", .{ vp, li }));
            defer self.a.free(qw);
            const kw = try self.loadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.k_proj.weight", .{ vp, li }));
            defer self.a.free(kw);
            const vw = try self.loadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.v_proj.weight", .{ vp, li }));
            defer self.a.free(vw);
            @memcpy(qkv_w_tmp[0..mat_elems], qw);
            @memcpy(qkv_w_tmp[mat_elems .. 2 * mat_elems], kw);
            @memcpy(qkv_w_tmp[2 * mat_elems .. 3 * mat_elems], vw);

            @memset(qkv_b_tmp, 0);
            inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |proj_n, pi| {
                const bn = try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.{s}.bias", .{ vp, li, proj_n });
                if (ckpt.getOptional(bn)) |bt| {
                    const bv = try dequantOrConvertToBf16(self.a, ckpt, bn, bt);
                    defer self.a.free(bv);
                    @memcpy(qkv_b_tmp[pi * c.hidden_size ..][0..c.hidden_size], bv);
                }
            }

            self.blocks[li] = .{
                .norm1_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.layer_norm1.weight", .{ vp, li })),
                .norm1_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.layer_norm1.bias", .{ vp, li }), c.hidden_size),
                .qkv_w = try self.allocUploadBf16(qkv_w_tmp),
                .qkv_b = try self.allocUploadBf16(qkv_b_tmp),
                .proj_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.out_proj.weight", .{ vp, li })),
                .proj_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.self_attn.out_proj.bias", .{ vp, li }), c.hidden_size),
                .norm2_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.layer_norm2.weight", .{ vp, li })),
                .norm2_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.layer_norm2.bias", .{ vp, li }), c.hidden_size),
                .fc1_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.mlp.fc1.weight", .{ vp, li })),
                .fc1_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.mlp.fc1.bias", .{ vp, li }), c.intermediate_size),
                .fc2_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.mlp.fc2.weight", .{ vp, li })),
                .fc2_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.encoder.layers.{d}.mlp.fc2.bias", .{ vp, li }), c.hidden_size),
            };
        }

        // 4. SigLIP post_layernorm
        self.post_ln_w = try self.uploadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.post_layernorm.weight", .{vp}));
        self.post_ln_b = try self.uploadOptionalBiasBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.post_layernorm.bias", .{vp}), c.hidden_size);

        // 5. Gemma3MultiModalProjector: (+1)-centered Gemma3RMSNorm + linear projection [out_H, V_H]
        const gn_raw = try self.loadTensorBf16(ckpt, try std.fmt.bufPrint(&nb, "{s}.mm_soft_emb_norm.weight", .{mp}));
        defer self.a.free(gn_raw);
        for (gn_raw) |*v| v.* = f32ToBf16(1.0 + bf16ToF32(v.*));
        self.gemma_norm_w_p1 = try self.allocUploadBf16(gn_raw);

        const mp_hf_name = try std.fmt.bufPrint(&nb, "{s}.mm_input_projection_weight", .{mp});
        var mp_alt_buf: [256]u8 = undefined;
        const mp_mlx_name = try std.fmt.bufPrint(&mp_alt_buf, "{s}.mm_input_projection.weight", .{mp});
        const mp_name = if (ckpt.getOptional(mp_hf_name) != null) mp_hf_name else mp_mlx_name;
        const mp_t = try ckpt.get(mp_name);
        const mp_vals = try dequantOrConvertToBf16(self.a, ckpt, mp_name, mp_t);
        defer self.a.free(mp_vals);

        // vit_linear_act expects W shaped [out_H, V_H]; HF mm_input_projection_weight is [V_H, out_H] -> transpose!
        if (mp_t.rank == 2 and mp_t.shape[0] == c.hidden_size and mp_t.shape[1] == c.out_hidden_size) {
            const transp = try self.a.alloc(u16, c.out_hidden_size * c.hidden_size);
            defer self.a.free(transp);
            for (0..c.hidden_size) |k| {
                for (0..c.out_hidden_size) |n| {
                    transp[n * c.hidden_size + k] = mp_vals[k * c.out_hidden_size + n];
                }
            }
            self.gemma_proj_w = try self.allocUploadBf16(transp);
        } else {
            self.gemma_proj_w = try self.allocUploadBf16(mp_vals);
        }
    }

    /// Decode, resize, patch-pack, and encode an image from a local file path, `file://`, or base64 data URI.
    pub fn encodeImage(self: *VisionEncoder, a: std.mem.Allocator, source: []const u8) !EncodedImage {
        const raw_bytes = try img.loadImageSource(a, source);
        defer a.free(raw_bytes);

        const c = self.c;
        if (c.kind == .qwen35_vl) {
            const orig = try img.probeImageSize(raw_bytes);
            const factor: u32 = @intCast(c.patch_size * c.spatial_merge_size);
            const hw = img.smartResizeQwen(orig[1], orig[0], factor, c.min_pixels, c.max_pixels);
            const rgb = try img.decodeAndResize(a, raw_bytes, hw[1], hw[0]);
            defer rgb.deinit(a);
            const patches = try img.packQwenPatches(a, rgb, @intCast(c.patch_size), @intCast(c.spatial_merge_size));
            defer a.free(patches);
            return self.encodePatches(a, patches, @intCast(hw[0] / c.patch_size), @intCast(hw[1] / c.patch_size));
        } else {
            const sz: u32 = @intCast(c.image_size);
            const rgb = try img.decodeAndResize(a, raw_bytes, sz, sz);
            defer rgb.deinit(a);
            const patches = try img.packSiglipPatches(a, rgb, @intCast(c.patch_size));
            defer a.free(patches);
            const g: u32 = @intCast(c.image_size / c.patch_size);
            return self.encodePatches(a, patches, g, g);
        }
    }

    /// Run the Metal ViT pipeline directly on pre-packed BF16 patches `[grid_h * grid_w, 3 * P * P]`.
    pub fn encodePatches(self: *VisionEncoder, a: std.mem.Allocator, patches_bf16: []const u16, grid_h: u32, grid_w: u32) !EncodedImage {
        const c = self.c;
        const n_patches: usize = @as(usize, grid_h) * grid_w;
        const vh = c.hidden_size;
        const vi = c.intermediate_size;
        const hd = c.headDim();
        const out_h = c.out_hidden_size;

        const out_tokens: usize = if (c.kind == .qwen35_vl)
            n_patches / (c.spatial_merge_size * c.spatial_merge_size)
        else
            c.mm_tokens_per_image;

        // Allocate temporary Metal work buffers for this image (removed from residency on exit)
        const patch_buf = try self.ctx.allocBuffer(patches_bf16.len * 2);
        defer self.ctx.freeBuffer(patch_buf);
        @memcpy(patch_buf.slice(u16, patches_bf16.len), patches_bf16);

        const x_buf = try self.ctx.allocBuffer(n_patches * vh * 2);
        defer self.ctx.freeBuffer(x_buf);
        const norm_buf = try self.ctx.allocBuffer(n_patches * vh * 2);
        defer self.ctx.freeBuffer(norm_buf);
        const qkv_buf = try self.ctx.allocBuffer(n_patches * 3 * vh * 2);
        defer self.ctx.freeBuffer(qkv_buf);
        const attn_buf = try self.ctx.allocBuffer(n_patches * vh * 2);
        defer self.ctx.freeBuffer(attn_buf);
        const mlp_buf = try self.ctx.allocBuffer(n_patches * vi * 2);
        defer self.ctx.freeBuffer(mlp_buf);
        const out_buf = try self.ctx.allocBuffer(out_tokens * out_h * 2);
        defer self.ctx.freeBuffer(out_buf);

        // Optional Qwen3.5-VL interpolated pos_embed + 2D Vision RoPE tables + merger scratch
        var pos_buf: ?Buffer = null;
        var cos_buf: ?Buffer = null;
        var sin_buf: ?Buffer = null;
        var merge_buf: ?Buffer = null;
        defer {
            if (pos_buf) |b| self.ctx.freeBuffer(b);
            if (cos_buf) |b| self.ctx.freeBuffer(b);
            if (sin_buf) |b| self.ctx.freeBuffer(b);
            if (merge_buf) |b| self.ctx.freeBuffer(b);
        }

        if (c.kind == .qwen35_vl) {
            const pb = try self.ctx.allocBuffer(n_patches * vh * 2);
            pos_buf = pb;
            self.computeQwenPosEmbed(pb.slice(u16, n_patches * vh), grid_h, grid_w);

            const half_d = hd / 2;
            const cb = try self.ctx.allocBuffer(n_patches * half_d * 4);
            const sb = try self.ctx.allocBuffer(n_patches * half_d * 4);
            cos_buf = cb;
            sin_buf = sb;
            self.computeQwenRopeTables(cb.slice(f32, n_patches * half_d), sb.slice(f32, n_patches * half_d), grid_h, grid_w);

            const merged_dim = vh * c.spatial_merge_size * c.spatial_merge_size;
            merge_buf = try self.ctx.allocBuffer(out_tokens * merged_dim * 2);
        }

        // Make temporary buffers visible to Metal 4 ResidencySet before encoding
        self.ctx.commitResidency();

        var cb = self.ctx.beginCommandBuffer();
        var enc = cb.beginCompute();

        // 1. Patch embedding linear projection -> x_buf [n_patches, vh]
        self.dispatchLinear(&enc, patch_buf, self.patch_w, self.patch_b, x_buf, n_patches, vh, self.patch_k, self.patch_has_bias, 0, false);
        enc.barrier();

        // 2. Add 2D positional embedding
        const pos_src = if (c.kind == .qwen35_vl) pos_buf.? else self.siglip_pos_w.?;
        const total_x: u32 = @intCast(n_patches * vh);
        enc.setPipeline(self.p_add);
        enc.setBuffer(x_buf, 0, 0);
        enc.setBuffer(pos_src, 0, 1);
        enc.setValue(VitAddParams{ .count = total_x }, 2);
        enc.dispatchThreads(.{ .width = total_x, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
        enc.barrier();

        // 3. ViT transformer blocks
        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        for (self.blocks) |blk| {
            // norm1(x) -> norm_buf
            self.dispatchLayerNorm(&enc, x_buf, blk.norm1_w, blk.norm1_b, norm_buf, n_patches, vh);
            enc.barrier();

            // qkv(norm_buf) -> qkv_buf [N, 3 * vh]
            self.dispatchLinear(&enc, norm_buf, blk.qkv_w, blk.qkv_b, qkv_buf, n_patches, 3 * vh, vh, true, 0, false);
            enc.barrier();

            // Optional 2D Vision RoPE (Qwen3.5-VL)
            if (c.kind == .qwen35_vl) {
                enc.setPipeline(self.p_rope);
                enc.setBuffer(qkv_buf, 0, 0);
                enc.setBuffer(cos_buf.?, 0, 1);
                enc.setBuffer(sin_buf.?, 0, 2);
                enc.setValue(VitRopeParams{
                    .seq_len = @intCast(n_patches),
                    .num_heads = @intCast(c.num_heads),
                    .head_dim = @intCast(hd),
                }, 3);
                enc.dispatchGroups(.{ .width = c.num_heads, .height = n_patches, .depth = 1 }, .{ .width = 32, .height = 1, .depth = 1 });
                enc.barrier();
            }

            // Bidirectional self-attention -> attn_buf [N, vh]
            enc.setPipeline(self.p_attn);
            enc.setBuffer(qkv_buf, 0, 0);
            enc.setBuffer(attn_buf, 0, 1);
            enc.setValue(VitAttnParams{
                .seq_len = @intCast(n_patches),
                .num_heads = @intCast(c.num_heads),
                .head_dim = @intCast(hd),
                .scale = attn_scale,
            }, 2);
            enc.dispatchGroups(.{ .width = c.num_heads, .height = n_patches, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
            enc.barrier();

            // proj(attn_buf) + residual -> x_buf
            self.dispatchLinear(&enc, attn_buf, blk.proj_w, blk.proj_b, x_buf, n_patches, vh, vh, true, 0, true);
            enc.barrier();

            // norm2(x) -> norm_buf
            self.dispatchLayerNorm(&enc, x_buf, blk.norm2_w, blk.norm2_b, norm_buf, n_patches, vh);
            enc.barrier();

            // fc1(norm_buf) + gelu_pytorch_tanh -> mlp_buf
            self.dispatchLinear(&enc, norm_buf, blk.fc1_w, blk.fc1_b, mlp_buf, n_patches, vi, vh, true, 1, false);
            enc.barrier();

            // fc2(mlp_buf) + residual -> x_buf
            self.dispatchLinear(&enc, mlp_buf, blk.fc2_w, blk.fc2_b, x_buf, n_patches, vh, vi, true, 0, true);
            enc.barrier();
        }

        // 4. Spatial merger / multimodal projector -> out_buf [out_tokens, out_h]
        if (c.kind == .qwen35_vl) {
            const merged_dim = vh * c.spatial_merge_size * c.spatial_merge_size;
            if (self.merger_postshuffle) {
                self.dispatchLayerNorm(&enc, x_buf, self.merger_norm_w.?, self.merger_norm_b.?, norm_buf, out_tokens, merged_dim);
            } else {
                self.dispatchLayerNorm(&enc, x_buf, self.merger_norm_w.?, self.merger_norm_b.?, norm_buf, n_patches, vh);
            }
            enc.barrier();

            // linear_fc1 + exact GELU (act = 2) -> merge_buf [out_tokens, merged_dim]
            self.dispatchLinear(&enc, norm_buf, self.merger_fc1_w.?, self.merger_fc1_b.?, merge_buf.?, out_tokens, merged_dim, merged_dim, true, 2, false);
            enc.barrier();

            // linear_fc2 -> out_buf [out_tokens, out_h]
            self.dispatchLinear(&enc, merge_buf.?, self.merger_fc2_w.?, self.merger_fc2_b.?, out_buf, out_tokens, out_h, merged_dim, true, 0, false);
            enc.barrier();
        } else {
            // SigLIP post_layernorm -> norm_buf [n_patches, vh]
            self.dispatchLayerNorm(&enc, x_buf, self.post_ln_w.?, self.post_ln_b.?, norm_buf, n_patches, vh);
            enc.barrier();

            // AvgPool2d + Gemma3RMSNorm (+1-centered) -> attn_buf [out_tokens, vh]
            const out_grid: u32 = @intFromFloat(@sqrt(@as(f64, @floatFromInt(out_tokens))));
            const k_pool: u32 = grid_h / out_grid;
            enc.setPipeline(self.p_pool_norm);
            enc.setBuffer(norm_buf, 0, 0);
            enc.setBuffer(self.gemma_norm_w_p1.?, 0, 1);
            enc.setBuffer(attn_buf, 0, 2);
            enc.setValue(VitPoolNormParams{
                .in_grid = grid_h,
                .out_grid = out_grid,
                .k_pool = k_pool,
                .dim = @intCast(vh),
                .eps = c.layer_norm_eps,
            }, 3);
            enc.dispatchGroups(.{ .width = out_tokens, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
            enc.barrier();

            // Multimodal linear projection -> out_buf [out_tokens, out_h] (no bias)
            self.dispatchLinear(&enc, attn_buf, self.gemma_proj_w.?, self.patch_b, out_buf, out_tokens, out_h, vh, false, 0, false);
            enc.barrier();
        }

        enc.end();
        cb.commit();
        try self.ctx.waitIdle();

        const embs = try a.alloc(u16, out_tokens * out_h);
        @memcpy(embs, out_buf.slice(u16, out_tokens * out_h));
        return .{
            .embs = embs,
            .num_tokens = @intCast(out_tokens),
            .hidden_size = @intCast(out_h),
            .grid_h = grid_h,
            .grid_w = grid_w,
        };
    }

    fn dispatchLayerNorm(self: *const VisionEncoder, enc: *mtl.Encoder, x: Buffer, w: Buffer, b: Buffer, out: Buffer, rows: usize, dim: usize) void {
        enc.setPipeline(self.p_ln);
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(w, 0, 1);
        enc.setBuffer(b, 0, 2);
        enc.setBuffer(out, 0, 3);
        enc.setValue(VitLnParams{ .rows = @intCast(rows), .dim = @intCast(dim), .eps = self.c.layer_norm_eps }, 4);
        enc.dispatchGroups(.{ .width = rows, .height = 1, .depth = 1 }, .{ .width = 256, .height = 1, .depth = 1 });
    }

    fn dispatchLinear(
        self: *const VisionEncoder,
        enc: *mtl.Encoder,
        x: Buffer,
        w: Buffer,
        b: Buffer,
        y: Buffer,
        rows: usize,
        n: usize,
        k: usize,
        has_bias: bool,
        act: u32,
        add_res: bool,
    ) void {
        enc.setPipeline(self.p_lin);
        enc.setBuffer(x, 0, 0);
        enc.setBuffer(w, 0, 1);
        enc.setBuffer(b, 0, 2);
        enc.setBuffer(y, 0, 3);
        enc.setValue(VitLinearParams{
            .rows = @intCast(rows),
            .N = @intCast(n),
            .K = @intCast(k),
            .has_bias = if (has_bias) 1 else 0,
            .act = act,
            .add_res = if (add_res) 1 else 0,
        }, 4);
        enc.dispatchGroups(.{ .width = (n + 3) / 4, .height = (rows + 3) / 4, .depth = 1 }, .{ .width = 128, .height = 1, .depth = 1 });
    }

    /// Exact Qwen3.5-VL `fast_pos_embed_interpolate` permuted into 2x2 merge-block order (bh, bw, ih, iw).
    fn computeQwenPosEmbed(self: *const VisionEncoder, dst: []u16, grid_h: u32, grid_w: u32) void {
        const c = self.c;
        const vh = c.hidden_size;
        const side: usize = @intFromFloat(@round(@sqrt(@as(f64, @floatFromInt(c.num_position_embeddings)))));
        const side_f: f64 = @floatFromInt(side - 1);
        const ms = c.spatial_merge_size;
        const mh = @as(usize, grid_h) / ms;
        const mw = @as(usize, grid_w) / ms;

        var p_idx: usize = 0;
        for (0..mh) |bh| {
            for (0..mw) |bw| {
                for (0..ms) |ih| {
                    for (0..ms) |iw| {
                        const py = bh * ms + ih;
                        const px = bw * ms + iw;
                        const hy: f64 = if (grid_h > 1) (@as(f64, @floatFromInt(py)) * side_f) / @as(f64, @floatFromInt(grid_h - 1)) else 0.0;
                        const wx: f64 = if (grid_w > 1) (@as(f64, @floatFromInt(px)) * side_f) / @as(f64, @floatFromInt(grid_w - 1)) else 0.0;
                        const h0: usize = @intFromFloat(@floor(hy));
                        const w0: usize = @intFromFloat(@floor(wx));
                        const h1: usize = @min(h0 + 1, side - 1);
                        const w1: usize = @min(w0 + 1, side - 1);
                        const dh: f32 = @floatCast(hy - @as(f64, @floatFromInt(h0)));
                        const dw: f32 = @floatCast(wx - @as(f64, @floatFromInt(w0)));

                        const w00 = (1.0 - dh) * (1.0 - dw);
                        const w01 = (1.0 - dh) * dw;
                        const w10 = dh * (1.0 - dw);
                        const w11 = dh * dw;

                        const p00 = self.qwen_pos_f32[(h0 * side + w0) * vh ..][0..vh];
                        const p01 = self.qwen_pos_f32[(h0 * side + w1) * vh ..][0..vh];
                        const p10 = self.qwen_pos_f32[(h1 * side + w0) * vh ..][0..vh];
                        const p11 = self.qwen_pos_f32[(h1 * side + w1) * vh ..][0..vh];
                        const out_row = dst[p_idx * vh ..][0..vh];

                        for (0..vh) |d| {
                            out_row[d] = f32ToBf16(w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d]);
                        }
                        p_idx += 1;
                    }
                }
            }
        }
    }

    /// Exact Qwen3.5-VL 2D Vision RoPE cos/sin tables `[N, head_dim / 2]` in 2x2 merge-block order.
    fn computeQwenRopeTables(self: *const VisionEncoder, cos_dst: []f32, sin_dst: []f32, grid_h: u32, grid_w: u32) void {
        const c = self.c;
        const hd = c.headDim();
        const half_d = hd / 2;
        const q_d = hd / 4;
        var inv_freq: [64]f64 = @splat(0.0);
        for (0..q_d) |i| {
            inv_freq[i] = 1.0 / std.math.pow(f64, 10000.0, (2.0 * @as(f64, @floatFromInt(i))) / @as(f64, @floatFromInt(half_d)));
        }

        const ms = c.spatial_merge_size;
        const mh = @as(usize, grid_h) / ms;
        const mw = @as(usize, grid_w) / ms;
        var p_idx: usize = 0;
        for (0..mh) |bh| {
            for (0..mw) |bw| {
                for (0..ms) |ih| {
                    for (0..ms) |iw| {
                        const py: f64 = @floatFromInt(bh * ms + ih);
                        const px: f64 = @floatFromInt(bw * ms + iw);
                        const c_row = cos_dst[p_idx * half_d ..][0..half_d];
                        const s_row = sin_dst[p_idx * half_d ..][0..half_d];
                        for (0..q_d) |i| {
                            const ay = py * inv_freq[i];
                            const ax = px * inv_freq[i];
                            c_row[i] = @floatCast(@cos(ay));
                            s_row[i] = @floatCast(@sin(ay));
                            c_row[q_d + i] = @floatCast(@cos(ax));
                            s_row[q_d + i] = @floatCast(@sin(ax));
                        }
                        p_idx += 1;
                    }
                }
            }
        }
    }
};

/// Convert BF16/F16/F32 or MLX affine-quantized U32 tensor to flat BF16 `[]u16`.
fn dequantOrConvertToBf16(a: std.mem.Allocator, ckpt: *const st.Checkpoint, name: []const u8, t: st.Tensor) ![]u16 {
    if (t.dtype == .bf16) {
        const src = t.asSlice(u16);
        const out = try a.alloc(u16, src.len);
        @memcpy(out, src);
        return out;
    }
    if (t.dtype == .f16) {
        const src = t.asSlice(u16);
        const out = try a.alloc(u16, src.len);
        for (src, 0..) |v, i| out[i] = f32ToBf16(f16BitsToF32(v));
        return out;
    }
    if (t.dtype == .f32) {
        const src = t.asSlice(f32);
        const out = try a.alloc(u16, src.len);
        for (src, 0..) |v, i| out[i] = f32ToBf16(v);
        return out;
    }
    if (t.dtype == .u32 and std.mem.endsWith(u8, name, ".weight")) {
        // Fallback dequantize for MLX affine-quantized vision layers
        const stem = name[0 .. name.len - ".weight".len];
        var sb_buf: [256]u8 = undefined;
        const sc_t = try ckpt.get(try std.fmt.bufPrint(&sb_buf, "{s}.scales", .{stem}));
        const bi_t = try ckpt.get(try std.fmt.bufPrint(&sb_buf, "{s}.biases", .{stem}));

        const rows = t.shape[0];
        const packed_cols = t.shape[1];
        const g_cols = sc_t.shape[1];
        // Infer (bits, gs) from packed_cols & g_cols
        var bits: usize = 4;
        var gs: usize = 64;
        const cand_bits = [_]usize{ 4, 8, 6, 3, 2, 5 };
        const cand_gs = [_]usize{ 64, 32, 128 };
        outer: for (cand_gs) |cg| {
            const k_cand = g_cols * cg;
            for (cand_bits) |cb| {
                if ((k_cand * cb + 31) / 32 == packed_cols) {
                    bits = cb;
                    gs = cg;
                    break :outer;
                }
            }
        }
        const k_dim = g_cols * gs;
        const out = try a.alloc(u16, rows * k_dim);
        const words = t.asSlice(u32);
        const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
        for (0..rows) |r| {
            for (0..g_cols) |g| {
                const sc_idx = r * g_cols + g;
                const sc: f32 = switch (sc_t.dtype) {
                    .bf16 => bf16ToF32(sc_t.asSlice(u16)[sc_idx]),
                    .f16 => f16BitsToF32(sc_t.asSlice(u16)[sc_idx]),
                    .f32 => sc_t.asSlice(f32)[sc_idx],
                    else => 1.0,
                };
                const bi: f32 = switch (bi_t.dtype) {
                    .bf16 => bf16ToF32(bi_t.asSlice(u16)[sc_idx]),
                    .f16 => f16BitsToF32(bi_t.asSlice(u16)[sc_idx]),
                    .f32 => bi_t.asSlice(f32)[sc_idx],
                    else => 0.0,
                };
                for (0..gs) |j| {
                    const col = g * gs + j;
                    const bit_off = col * bits;
                    const wi = bit_off >> 5;
                    const sh: u5 = @intCast(bit_off & 31);
                    var raw_q: u32 = words[r * packed_cols + wi] >> sh;
                    if (@as(usize, sh) + bits > 32 and wi + 1 < packed_cols) {
                        const lo_bits: u5 = @intCast(32 - @as(usize, sh));
                        raw_q |= words[r * packed_cols + wi + 1] << lo_bits;
                    }
                    const q = raw_q & mask;
                    out[r * k_dim + col] = f32ToBf16(@as(f32, @floatFromInt(q)) * sc + bi);
                }
            }
        }
        return out;
    }
    return error.InvalidCheckpoint;
}

/// Build exact multimodal prompt token IDs with vision start/pad/end markers for Qwen3.5-VL & Gemma 3-VL.
pub fn buildMultimodalPrompt(
    a: std.mem.Allocator,
    tok: *const tok_mod.Tokenizer,
    vc: VisionConfig,
    user_text: []const u8,
    img_token_counts: []const u32,
    use_chat: bool,
    enable_thinking: bool,
) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);

    if (use_chat) {
        const prefix = switch (vc.kind) {
            .gemma3_siglip => "<bos><start_of_turn>user\n",
            else => "<|im_start|>user\n",
        };
        const p_ids = try tok.encode(a, prefix);
        defer a.free(p_ids);
        try ids.appendSlice(a, p_ids);
    }

    for (img_token_counts) |cnt| {
        try ids.append(a, vc.vision_start_token_id);
        for (0..cnt) |_| try ids.append(a, vc.image_token_id);
        try ids.append(a, vc.vision_end_token_id);
    }

    if (user_text.len > 0) {
        const t_ids = try tok.encode(a, user_text);
        defer a.free(t_ids);
        try ids.appendSlice(a, t_ids);
    }

    if (use_chat) {
        const suffix = switch (vc.kind) {
            .gemma3_siglip => "<end_of_turn>\n<start_of_turn>model\n",
            else => if (!tok.has_think_token)
                "<|im_end|>\n<|im_start|>assistant\n"
            else if (enable_thinking)
                "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            else
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        };
        const s_ids = try tok.encode(a, suffix);
        defer a.free(s_ids);
        try ids.appendSlice(a, s_ids);
    }

    return ids.toOwnedSlice(a);
}
