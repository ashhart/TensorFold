//! H3 video VAE decoder on Metal: ViT-only decoder (36 blocks, 32 heads x 64,
//! RoPE theta 100, RMSNorm qk, register tokens, patch head 4x16x16 per token,
//! post-quant 1x1x1 conv before the ViT, tile 256/overlap 64 with cross-fade).
//! Weights: video_vae/source/model.safetensors, tensor names decoder.* and
//! post_quant_conv.*. Reference for parity:
//! zig/src/families/h3/vae/reference/vae_video.py. Kernels: vae.metal.

const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");

pub const Config = struct {
    latent_channels: usize = 24,
    out_channels: usize = 3,
    layers: usize = 36,
    heads: usize = 32,
    head_dim: usize = 64,
    ffn_mult: usize = 4,
    register_tokens: usize = 4,
    rope_theta: f32 = 100.0,
    rope_dim_ratio: f32 = 0.75,
    eps: f32 = 1e-5,
    spatial_ratio: usize = 16,
    temporal_ratio: usize = 4,
    clip_length: usize = 17,
    token_drop: usize = 3,
    tile: usize = 256,
    tile_overlap: usize = 64,
    latents_mean: [24]f32,
    latents_std: [24]f32,

    pub fn dim(self: *const Config) usize {
        return self.heads * self.head_dim;
    }
    pub fn ffn(self: *const Config) usize {
        return self.dim() * self.ffn_mult;
    }
};

pub const Block = struct {
    norm1_w: []const f32,
    attn_qkv_w: []const f32,
    attn_qkv_b: []const f32,
    attn_out_w: []const f32,
    attn_out_b: []const f32,
    scale1: []const f32,
    norm2_w: []const f32,
    ff_w1: []const f32,
    ff_w1_b: []const f32,
    ff_w2: []const f32,
    ff_w2_b: []const f32,
    scale2: []const f32,
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: Config,
    // ViT weights as f32 host buffers (one per tensor), shapes verified at load.
    x_embedder_w: []f32,
    x_embedder_b: []f32,
    register_tokens: []f32,
    blocks: []Block,
    norm_out_w: []f32,
    norm_out_b: []f32,
    proj_out_w: []f32,
    proj_out_b: []f32,
    post_quant_w: []f32,
    post_quant_b: []f32,
    vit_checkpoint: ck.Checkpoint,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, pipeline_dir: []const u8) !*Decoder {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const wrapper_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ pipeline_dir, "video_vae", "config.json" }), arena, .limited(1 << 20));
        const source_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ pipeline_dir, "video_vae", "source", "config.json" }), arena, .limited(1 << 20));

        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .device = device,
            .config = try parseConfig(wrapper_bytes, source_bytes),
            .x_embedder_w = &.{},
            .x_embedder_b = &.{},
            .register_tokens = &.{},
            .blocks = &.{},
            .norm_out_w = &.{},
            .norm_out_b = &.{},
            .proj_out_w = &.{},
            .proj_out_b = &.{},
            .post_quant_w = &.{},
            .post_quant_b = &.{},
            .vit_checkpoint = ck.Checkpoint.init(gpa),
        };
        try checkGeometry(d.config, wrapper_bytes, source_bytes);

        d.queue = try device.queue();
        errdefer d.queue.deinit();

        var vit = ck.Checkpoint.init(gpa);
        try vit.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/video_vae/source/model.safetensors", .{pipeline_dir}, 0), "");
        try d.unpackVit(gpa, &vit);
        d.vit_checkpoint = vit;
        return d;
    }

    fn parseConfig(wrapper_json: []const u8, source_json: []const u8) !Config {
        var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const wrapper = try std.json.parseFromSlice(std.json.Value, arena, wrapper_json, .{});
        const source = try std.json.parseFromSlice(std.json.Value, arena, source_json, .{});
        var c: Config = .{ .latents_mean = undefined, .latents_std = undefined };
        if (source.value.object.get("z_channels")) |v| if (v == .integer) {
            if (v.integer != 24) return error.UnsupportedGeometry;
        };
        if (source.value.object.get("use_vit_decoder")) |v| if (v == .bool) {
            if (!v.bool) return error.NotVitDecoder;
        };
        if (source.value.object.get("vit_decoder_kwargs")) |v| {
            const o = v.object;
            const layers = o.get("num_layers") orelse return error.BadConfig;
            if (layers != .integer or layers.integer != 36) return error.UnsupportedGeometry;
            const heads = o.get("heads") orelse return error.BadConfig;
            if (heads != .integer or heads.integer != 32) return error.UnsupportedGeometry;
            const hd = o.get("dim_head") orelse return error.BadConfig;
            if (hd != .integer or hd.integer != 64) return error.UnsupportedGeometry;
        }
        if (wrapper.value.object.get("vae_clip_length")) |v| {
            if (v == .integer) c.clip_length = @intCast(v.integer);
        }
        if (wrapper.value.object.get("vae_token_drop")) |v| {
            if (v == .integer) c.token_drop = @intCast(v.integer);
        }
        return c;
    }

    fn checkGeometry(config: Config, wrapper_json: []const u8, source_json: []const u8) !void {
        _ = config;
        _ = wrapper_json;
        _ = source_json;
    }

    fn unpackVit(d: *Decoder, gpa: std.mem.Allocator, c: *const ck.Checkpoint) !void {
        const dim = d.config.dim();
        const ffn = d.config.ffn();
        d.x_embedder_w = try readF32(c, "decoder.x_embedder.weight", dim * d.config.latent_channels);
        d.x_embedder_b = try readF32(c, "decoder.x_embedder.bias", dim);
        d.register_tokens = try readF32(c, "decoder.register_tokens", d.config.register_tokens * dim);
        d.norm_out_w = try readF32(c, "decoder.norm_out.weight", dim);
        d.norm_out_b = try readF32(c, "decoder.norm_out.bias", dim);
        const head_block = d.config.out_channels * d.config.temporal_ratio * d.config.spatial_ratio * d.config.spatial_ratio;
        d.proj_out_w = try readF32(c, "decoder.proj_out.weight", head_block * dim);
        d.proj_out_b = try readF32(c, "decoder.proj_out.bias", head_block);
        d.post_quant_w = try readF32(c, "post_quant_conv.weight", d.config.latent_channels * d.config.latent_channels);
        d.post_quant_b = try readF32(c, "post_quant_conv.bias", d.config.latent_channels);

        const blocks = try gpa.alloc(Block, d.config.layers);
        const qkv = 3 * dim;
        const ww = 2 * ffn;
        var name: [128]u8 = undefined;
        for (blocks, 0..) |*b, i| {
            b.* = .{
                .norm1_w = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.norm1.weight", .{i}), dim),
                .attn_qkv_w = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_qkv.weight", .{i}), qkv * dim),
                .attn_qkv_b = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_qkv.bias", .{i}), qkv),
                .attn_out_w = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_out.weight", .{i}), dim * dim),
                .attn_out_b = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_out.bias", .{i}), dim),
                .scale1 = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.scale1", .{i}), dim),
                .norm2_w = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.norm2.weight", .{i}), dim),
                .ff_w1 = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w1.weight", .{i}), ww * dim),
                .ff_w1_b = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w1.bias", .{i}), ww),
                .ff_w2 = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w2.weight", .{i}), dim * ffn),
                .ff_w2_b = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w2.bias", .{i}), dim),
                .scale2 = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.scale2", .{i}), dim),
            };
        }
        d.blocks = blocks;
    }

    /// Full decode: (C=24, F, H, W) latents to RGB uint8 frames.
    /// The ViT forward and the tile logic land with the kernels; see the plan.
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize) ![]u8 {
        _ = self;
        _ = gpa;
        _ = latents;
        _ = depth;
        _ = height;
        _ = width;
        return error.NotPortedYet;
    }
};

fn readF32(c: *const ck.Checkpoint, name: []const u8, count: usize) ![]f32 {
    const tensor = c.tensors.get(name) orelse return error.MissingTensor;
    if (tensor.count() != count) return error.BadShape;
    const host = tensor.host(f32);
    return @constCast(host);
}
