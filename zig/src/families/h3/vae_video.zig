//! H3 video VAE decoder on Metal: the ViT decoder (36 blocks, 32 heads x 64,
//! RoPE theta 100, RMSNorm qk) running the released MiniMax-H3 video_vae
//! geometry. Weights: video_vae/source/model.safetensors, tensor names
//! decoder.* (x_embedder, register_tokens, transformer_blocks.N.*, norm_out,
//! proj_out) plus the CNN tail (post_quant_conv, decoder conv stack) of
//! AutoencoderKLLegacy. Reference for parity:
//! zig/src/families/h3/vae/reference/vae_video.py (Python/MLX).
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;

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

    pub fn dim(self: *const Config) usize {
        return self.heads * self.head_dim;
    }
    pub fn ffn(self: *const Config) usize {
        return self.dim() * self.ffn_mult;
    }
};

pub const Weights = struct {
    checkpoint: ck.Checkpoint,
    config: Config,
    shards: usize,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, source_dir: []const u8, shards: usize) !Weights {
        _ = gpa;
        _ = device;
        var self = Weights{ .checkpoint = undefined, .config = .{}, .shards = shards };
        self.checkpoint = try ck.Checkpoint.open(source_dir, shards);
        return self;
    }

    /// decoder.transformer_blocks.N.<part> for 0..layers, verified against the
    /// released geometry before any buffer is made.
    pub fn block(self: *const Weights, index: usize, part: []const u8, count: usize) !Ref {
        var name: [128]u8 = undefined;
        const text = try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.{s}", .{ index, part });
        return self.checkpoint.ref(text, count);
    }
};

pub const Decoder = struct {
    weights: Weights,
    config: Config,
    device: mtl.Device,

    /// Full decode: (C=24, F, H, W) latents to RGB frames (F*4, H*16, W*16, 3) uint8.
    /// Steps mirror vae/reference/vae_video.py VideoDecoder: clip slicing with
    /// token_drop cross-fade, ViT forward, CNN tail, denormalize, uint8.
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
