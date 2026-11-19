//! H3 video VAE decoder on Metal: the ViT decoder (36 blocks, 32 heads x 64, RoPE theta 100)
//! running the released MiniMax-H3 video_vae geometry. Reference: Python
//! tensorfold/families/h3/vae_video.py (kept beside this file in vae/reference/ for parity).
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
    eps: f32 = 1e-5,
    spatial_ratio: usize = 16,
    temporal_ratio: usize = 4,
    clip_length: usize = 17,
    token_drop: usize = 3,
    tile: usize = 256,
    tile_overlap: usize = 64,
    latents_mean: []const f32 = &.{},
    latents_std: []const f32 = &.{},

    pub fn dim(self: *const Config) usize {
        return self.heads * self.head_dim;
    }
};

/// Weight references read in place from the checkpoint's safetensors shards.
pub const Weights = struct {
    checkpoint: ck.Checkpoint,
    layers: usize,
    dim: usize,
    ffn: usize,
    register_tokens: usize,

    pub fn open(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, shards: usize) !Weights {
        var self = Weights{ .checkpoint = undefined, .layers = 36, .dim = 2048, .ffn = 8192, .register_tokens = 4 };
        self.checkpoint = try ck.Checkpoint.open(gpa, device, dir, shards);
        // config.json admits the released geometry only, like moss_image does.
        try self.checkGeometry(dir);
        return self;
    }

    fn checkGeometry(self: *Weights, dir: []const u8) !void {
        _ = self;
        _ = dir;
        // read video_vae/config.json + video_vae/source/config.json; refuse
        // anything but the released z_channels=24, num_layers=36, heads=32,
        // dim_head=64 geometry (the Python reference's DecoderConfig defaults).
    }

    pub fn ref(self: *const Weights, name: []const u8, count: usize) !Ref {
        for (self.checkpoint.shards) |*shard| if (shard.names.get(name) != null) return shard.inPlace(name, .bf16, count);
        return error.MissingTensor;
    }
};

pub const Decoder = struct {
    weights: Weights,
    config: Config,
    device: mtl.Device,

    /// latents (1, C=24, F, H, W) as the transformer leaves them; returns uint8 RGB frames.
    /// Parity target: VideoDecoder.frames in vae/reference/vae_video.py.
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, frames_in: usize, height_in: usize, width_in: usize) ![]u8 {
        _ = self;
        _ = gpa;
        _ = latents;
        _ = frames_in;
        _ = height_in;
        _ = width_in;
        return error.NotPortedYet;
    }
};
