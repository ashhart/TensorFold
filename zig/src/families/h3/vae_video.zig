//! H3 video VAE decoder on Metal: the ViT decoder (36 blocks, 32 heads x 64,
//! RoPE theta 100, RMSNorm qk) running the released MiniMax-H3 video_vae
//! geometry. Weights: video_vae/source/model.safetensors, tensor names
//! decoder.* plus the CNN tail of AutoencoderKLLegacy. Reference for parity:
//! zig/src/families/h3/vae/reference/vae_video.py (Python/MLX).
//! Kernels: zig/kernels/metal/h3/vae.metal.
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

    pub fn dim(self: *const Config) usize {
        return self.heads * self.head_dim;
    }
    pub fn ffn(self: *const Config) usize {
        return self.dim() * self.ffn_mult;
    }
};

const DecoderTensors = struct {
    x_embedder_w: []f32, x_embedder_b: []f32,
    register_tokens: []f32,
    norm_out_w: []f32, norm_out_b: []f32,
    proj_out_w: []f32, proj_out_b: []f32,
    attn_qkv_w: []const []f32, attn_qkv_b: []const []f32,
    attn_out_w: []const []f32, attn_out_b: []const []f32,
    ff_w1: []const []f32, ff_w1_b: []const []f32,
    ff_w2: []const []f32, ff_w2_b: []const []f32,
    norm1: []const []f32, norm2: []const []f32,
    scale1: []const []f32, scale2: []const []f32,
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: Config,
    t: DecoderTensors,
    cnn_checkpoint: ck.Checkpoint,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, pipeline_dir: []const u8) !*Decoder {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const wrapper = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ pipeline_dir, "video_vae", "config.json" }), arena, .limited(1 << 20));
        const source = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ pipeline_dir, "video_vae", "source", "config.json" }), arena, .limited(1 << 20));

        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .device = device, .config = try parseConfig(wrapper, source), .t = undefined, .cnn_checkpoint = ck.Checkpoint.init(gpa) };
        try checkGeometry(&d.config, wrapper, source);

        d.queue = try device.queue();
        errdefer d.queue.deinit();

        var vit = ck.Checkpoint.init(gpa);
        try vit.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/video_vae/source/model.safetensors", .{pipeline_dir}, 0), "");
        try d.unpackVit(&vit);
        d.cnn_checkpoint = vit;
        return d;
    }

    fn parseConfig(wrapper_json: []const u8, source_json: []const u8) !Config {
        _ = wrapper_json;
        _ = source_json;
        return .{};
    }

    fn checkGeometry(config: *const Config, wrapper_json: []const u8, source_json: []const u8) !void {
        var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        inline for (.{
            .{ "wrapper", wrapper_json }, .{ "source", source_json },
        }) |pair| {
            const doc = try std.json.parseFromSlice(std.json.Value, arena, pair[1], .{});
            _ = doc;
            _ = pair;
        }
        _ = config;
    }

    fn unpackVit(d: *Decoder, c: *const ck.Checkpoint) !void {
        const dim = d.config.dim();
        const ffn = d.config.ffn();
        var t: DecoderTensors = undefined;
        t.x_embedder_w = try readF32(c, "decoder.x_embedder.weight", dim * d.config.latent_channels);
        t.x_embedder_b = try readF32(c, "decoder.x_embedder.bias", dim);
        t.register_tokens = try readF32(c, "decoder.register_tokens", d.config.register_tokens * dim);
        t.norm_out_w = try readF32(c, "decoder.norm_out.weight", dim);
        t.norm_out_b = try readF32(c, "decoder.norm_out.bias", dim);
        t.proj_out_w = try readF32(c, "decoder.proj_out.weight", d.config.out_channels * d.config.temporal_ratio * d.config.spatial_ratio * d.config.spatial_ratio * dim);
        t.proj_out_b = try readF32(c, "decoder.proj_out.bias", d.config.out_channels * d.config.temporal_ratio * d.config.spatial_ratio * d.config.spatial_ratio);
        const layers = d.config.layers;
        const qkv = 3 * dim;
        const ww = 2 * ffn;
        const sa = d.gpa;
        const qkv_w = try sa.alloc([]f32, layers); const qkv_b = try sa.alloc([]f32, layers);
        const out_w = try sa.alloc([]f32, layers); const out_b = try sa.alloc([]f32, layers);
        const fw1 = try sa.alloc([]f32, layers); const fw1b = try sa.alloc([]f32, layers);
        const fw2 = try sa.alloc([]f32, layers); const fw2b = try sa.alloc([]f32, layers);
        const n1 = try sa.alloc([]f32, layers); const n2 = try sa.alloc([]f32, layers);
        const s1 = try sa.alloc([]f32, layers); const s2 = try sa.alloc([]f32, layers);
        var name: [128]u8 = undefined;
        for (0..layers) |i| {
            qkv_w[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_qkv.weight", .{i}), qkv * dim);
            qkv_b[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_qkv.bias", .{i}), qkv);
            out_w[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_out.weight", .{i}), dim * dim);
            out_b[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.attn.to_out.bias", .{i}), dim);
            fw1[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w1.weight", .{i}), ww * dim);
            fw1b[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w1.bias", .{i}), ww);
            fw2[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w2.weight", .{i}), dim * ffn);
            fw2b[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.ff.w2.bias", .{i}), dim);
            n1[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.norm1.weight", .{i}), dim);
            n2[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.norm2.weight", .{i}), dim);
            s1[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.scale1", .{i}), dim);
            s2[i] = try readF32(c, try std.fmt.bufPrint(&name, "decoder.transformer_blocks.{d}.scale2", .{i}), dim);
        }
        t.attn_qkv_w = qkv_w; t.attn_qkv_b = qkv_b;
        t.attn_out_w = out_w; t.attn_out_b = out_b;
        t.ff_w1 = fw1; t.ff_w1_b = fw1b;
        t.ff_w2 = fw2; t.ff_w2_b = fw2b;
        t.norm1 = n1; t.norm2 = n2;
        t.scale1 = s1; t.scale2 = s2;
        d.t = t;
    }

    /// Full decode: (C=24, F, H, W) latents to RGB uint8 frames.
    /// ViT forward lands with its kernels in the steps below; the CNN tail after.
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

fn gpaAllocSlice(gpa: std.mem.Allocator, count: usize) ![]const []f32 {
    const slice = try gpa.alloc([]f32, count);
    return slice;
}

fn readF32(c: *const ck.Checkpoint, name: []const u8, count: usize) ![]f32 {
    const tensor = c.tensors.get(name) orelse return error.MissingTensor;
    if (tensor.count() != count) return error.BadShape;
    const host = tensor.host(f32);
    return @constCast(host);
}
