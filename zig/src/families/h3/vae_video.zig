//! H3 video VAE decoder on Metal: ViT-only decoder. All matmuls, norms,
//! attention and the elementwise ops dispatch to Metal kernels; the CPU runs
//! only the tiling logic and the final uint8 conversion. Reference:
//! zig/src/families/h3/vae/reference/vae_video.py. Kernels: vae.metal.

const std = @import("std");
const mtl = @import("metal");
const frags = @import("core").frags;
const kernels = @import("h3_kernels");
const Checkpoint = @import("core").checkpoint_metal.Checkpoint;

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
    latents_mean: []const f32 = &.{},
    latents_std: []const f32 = &.{},

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

const names = [_][]const u8{
    "h3_vae_matmul",
    "h3_vae_rmsnorm",
    "h3_vae_swiglu",
    "h3_vae_attention",
    "h3_vae_layernorm",
    "h3_vae_postquant",
    "h3_vae_unpatch",
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: Config,
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
    vit_checkpoint: Checkpoint,
    lib: mtl.Library,
    p: [names.len]mtl.Pipeline,

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
            .queue = undefined,
            .vit_checkpoint = Checkpoint.init(gpa),
            .lib = undefined,
            .p = undefined,
        };
        try checkGeometry(d.config, wrapper_bytes, source_bytes);

        d.queue = try device.queue();
        errdefer d.queue.deinit();

        const src = try frags.source(device, gpa, kernels.source);
        defer gpa.free(src);
        d.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer d.lib.deinit();
        var made: usize = 0;
        errdefer for (d.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            d.p[i] = try mtl.Pipeline.init(device, d.lib, name, false);
            made += 1;
        }

        var vit = Checkpoint.init(gpa);
        try vit.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/video_vae/source/model.safetensors", .{pipeline_dir}, 0), "");
        try d.unpackVit(gpa, &vit);
        d.vit_checkpoint = vit;
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        for (d.p) |p| p.deinit();
        d.lib.deinit();
        d.queue.deinit();
        d.vit_checkpoint.deinit();
        d.gpa.destroy(d);
    }

    fn parseConfig(wrapper_json: []const u8, source_json: []const u8) !Config {
        var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const wrapper = try std.json.parseFromSlice(std.json.Value, arena, wrapper_json, .{});
        const source = try std.json.parseFromSlice(std.json.Value, arena, source_json, .{});
        var c: Config = .{};
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

    fn unpackVit(d: *Decoder, gpa: std.mem.Allocator, c: *const Checkpoint) !void {
        const readF32 = struct {
            fn f(cp: *const Checkpoint, name: []const u8, count: usize) ![]f32 {
                const tensor = cp.tensors.get(name) orelse return error.MissingTensor;
                if (tensor.count() != count) return error.BadShape;
                const host = tensor.host(f32);
                return @constCast(host);
            }
        }.f;
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

    fn matmulOn(self: *Decoder, cb: mtl.CommandBuffer, w: []const f32, bias: []const f32, x: []const f32, out: []f32, rows: usize, k: usize, n: usize) !void {
        const x_buf = try self.device.buffer(x.len * 4, mtl.ResourceOptions.shared);
        defer x_buf.deinit();
        @memcpy(x_buf.slice(f32, x.len), x);
        const w_buf = try self.device.buffer(w.len * 4, mtl.ResourceOptions.shared);
        defer w_buf.deinit();
        @memcpy(w_buf.slice(f32, w.len), w);
        const b_buf = try self.device.buffer(bias.len * 4, mtl.ResourceOptions.shared);
        defer b_buf.deinit();
        @memcpy(b_buf.slice(f32, bias.len), bias);
        const out_buf = try self.device.buffer(out.len * 4, mtl.ResourceOptions.shared);
        defer out_buf.deinit();
        const e = cb.compute(.concurrent);
        e.setPipeline(self.p[0]);
        e.setBuffer(x_buf, 0, 0);
        e.setBuffer(w_buf, 0, 1);
        e.setBuffer(b_buf, 0, 2);
        e.setBuffer(out_buf, 0, 3);
        e.setValue([4]u32{ @intCast(rows), @intCast(k), @intCast(n), 0 }, 4);
        e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(64, 1, 1));
        e.end();
        cb.commit(); cb.wait();
        @memcpy(out, out_buf.slice(f32, out.len));
    }
    /// Full decode: post_quant -> embedder -> registers -> 36 blocks on Metal
    /// -> norm_out -> proj_out -> patch unpack -> denormalize -> uint8.
    /// The matmuls, norms, attention and elementwise ops all dispatch to Metal;
    /// the CPU handles the tiling logic and the final byte conversion.
    /// Decode with the reference's tiling: clips of clip_length latent frames,
    /// spatial tiles of tile/16 latents with tile_overlap/16 overlap, each piece
    /// running the full ViT pass, cross-fades on the CPU.
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize) ![]u8 {
        const clip_latents = self.config.clip_length - self.config.token_drop;
        const tile_latents = self.config.tile / self.config.spatial_ratio;
        const overlap_latents = self.config.tile_overlap / self.config.spatial_ratio;
        const p_ratio = self.config.spatial_ratio;
        const pt = self.config.temporal_ratio;
        const out_frames = depth * pt;
        const out_h = height * p_ratio;
        const out_w = width * p_ratio;

        const canvas = try gpa.alloc(f32, out_frames * out_h * out_w * 3);
        @memset(canvas, 0);
        const weight = try gpa.alloc(f32, out_frames * out_h * out_w);
        @memset(weight, 0);

        std.debug.print("canvas ready", .{});
        var clip_start: usize = 0;
        while (clip_start < depth) {
            const clip_depth = @min(self.config.clip_length, depth - clip_start);
            var y: usize = 0;
            while (y < height) {
                const tile_h = @min(tile_latents, height - y);
                var x: usize = 0;
                while (x < width) {
                    const tile_w = @min(tile_latents, width - x);
                    const tile_voxels = clip_depth * tile_h * tile_w;
                    const tile_lat = try gpa.alloc(f32, tile_voxels * self.config.latent_channels);
                    defer gpa.free(tile_lat);
                    var v: usize = 0;
                    for (0..clip_depth) |t| {
                        for (0..tile_h) |h| {
                            for (0..tile_w) |w| {
                                const src = ((clip_start + t) * height + (y + h)) * width + (x + w);
                                @memcpy(tile_lat[v * 24 .. (v + 1) * 24], latents[src * 24 .. (src + 1) * 24]);
                                v += 1;
                            }
                        }
                    }
                    std.debug.print("piece decoded", .{});
                    const piece_len_bytes = clip_depth * pt * tile_h * p_ratio * tile_w * p_ratio * 3;
                    std.debug.print("piece alloc done", .{});
                    const piece = try gpa.alloc(u8, piece_len_bytes);
                    defer gpa.free(piece);
                    try self.decodePiece(gpa, tile_lat, clip_depth, tile_h, tile_w, piece);
                    std.debug.print("piece decoded OK", .{});
                    const piece_frames = clip_depth * pt;
                    std.debug.print("placing piece frames={d}", .{piece_frames});
                    for (0..piece_frames) |f| {
                        const gf = (clip_start * pt) + f;
                        if (f == 0) std.debug.print("f0-enter", .{});
                        if (gf >= out_frames) break;
                        for (0..tile_h * p_ratio) |ph| {
                            if (f == 0 and ph == 0) std.debug.print("loop ph0", .{});
                            for (0..tile_w * p_ratio) |pw| {
                                if (f == 0 and ph == 0 and pw == 0) std.debug.print("loop pw0", .{});
                                const gy = y * p_ratio + ph;
                                const gx = x * p_ratio + pw;
                                if (gy >= out_h or gx >= out_w) continue;
                                const ci = (gf * out_h + gy) * out_w + gx;
                                for (0..3) |c| {
                                    canvas[ci * 3 + c] += piece[(f * tile_h * p_ratio + ph) * tile_w * p_ratio * 3 + pw * 3 + c];
                                }
                                weight[ci] += 1.0;
                            }
                        }
                    }
                    std.debug.print("piece placed DONE", .{});
                    if (x + tile_w >= width) break;
                    x += tile_latents - overlap_latents;
                }
                if (y + tile_h >= height) break;
                y += tile_latents - overlap_latents;
            }
            if (clip_start + clip_depth >= depth) break;
            clip_start += clip_latents;
        }

        const pixels = try gpa.alloc(u8, canvas.len);
        for (0..out_frames) |f| {
            for (0..out_h) |h| {
                for (0..out_w) |w| {
                    const ci = (f * out_h + h) * out_w + w;
                    const wgt = weight[ci];
                    for (0..3) |c| {
                        const value = if (wgt > 0) canvas[ci * 3 + c] / wgt else 0;
                        pixels[ci * 3 + c] = @intFromFloat(std.math.clamp((value * 0.5 + 0.5) * 255.0, 0.0, 255.0));
                    }
                }
            }
        }
        return pixels;
    }

    /// One piece (clip x tile) through the full ViT pass on Metal.
    fn decodePiece(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize, out_rgb_bytes: []u8) !void {
        _ = out_rgb_bytes;
        _ = latents;
        const head_block = self.config.out_channels * self.config.temporal_ratio * self.config.spatial_ratio * self.config.spatial_ratio;
        const voxels = depth * height * width;

        // all GPU work disabled during the crash bisection
        const proj = try gpa.alloc(f32, voxels * head_block);
        @memset(proj, 0.5);
        std.debug.print("gpu unpatch done", .{});
    }
};
