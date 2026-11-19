//! H3 video VAE decoder on Metal: ViT-only decoder (36 blocks, 32 heads x 64,
//! RoPE theta 100, RMSNorm qk, register tokens, patch head 4x16x16 per token,
//! post-quant 1x1x1 conv before the ViT, tile 256/overlap 64 with cross-fade).
//! Weights: video_vae/source/model.safetensors, tensor names decoder.* and
//! post_quant_conv.*. Reference for parity:
//! zig/src/families/h3/vae/reference/vae_video.py. Kernels: vae.metal, joined
//! into the family source through h3_kernels.

const std = @import("std");
const mtl = @import("metal");
const ck = @import("core").checkpoint_metal;
const frags = @import("core").frags;
const kernels = @import("h3_kernels");

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
    vit_checkpoint: ck.Checkpoint,
    lib: mtl.Library,
    p_rmsnorm: mtl.Pipeline,
    p_swiglu: mtl.Pipeline,
    p_attention: mtl.Pipeline,

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
            .vit_checkpoint = ck.Checkpoint.init(gpa),
            .lib = undefined,
            .p_rmsnorm = undefined,
            .p_swiglu = undefined,
            .p_attention = undefined,
        };
        try checkGeometry(d.config, wrapper_bytes, source_bytes);

        d.queue = try device.queue();
        errdefer d.queue.deinit();

        const src = try frags.source(device, gpa, @import("h3_kernels").source);
        defer gpa.free(src);
        d.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer d.lib.deinit();
        d.p_rmsnorm = try mtl.Pipeline.init(device, d.lib, "h3_vae_rmsnorm", false);
        errdefer d.p_rmsnorm.deinit();
        d.p_swiglu = try mtl.Pipeline.init(device, d.lib, "h3_vae_swiglu", false);
        errdefer d.p_swiglu.deinit();
        d.p_attention = try mtl.Pipeline.init(device, d.lib, "h3_vae_attention", false);
        errdefer d.p_attention.deinit();

        var vit = ck.Checkpoint.init(gpa);
        try vit.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/video_vae/source/model.safetensors", .{pipeline_dir}, 0), "");
        try d.unpackVit(gpa, &vit);
        d.vit_checkpoint = vit;
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        d.p_attention.deinit();
        d.p_swiglu.deinit();
        d.p_rmsnorm.deinit();
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

    fn unpackVit(d: *Decoder, gpa: std.mem.Allocator, c: *const ck.Checkpoint) !void {
        const readF32 = struct {
            fn f(cp: *const ck.Checkpoint, name: []const u8, count: usize) ![]f32 {
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

    /// Decode: (C=24, F, H, W) latents to RGB uint8 frames.
    /// post_quant and x_embedder run on the CPU; the 36 blocks run through the
    /// Metal kernels; proj_out and the patch unpack back on the CPU. Tiling
    /// 256/64 with cross-fade wraps the whole flow, as the reference does.
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize) ![]u8 {
        const dim = self.config.dim();
        const channels = self.config.latent_channels;
        const voxels = depth * height * width;

        const pq = try gpa.alloc(f32, voxels * channels);
        defer gpa.free(pq);
        for (0..voxels) |v| {
            for (0..channels) |o| {
                var acc = self.post_quant_b[o];
                for (0..channels) |i| acc += self.post_quant_w[o * channels + i] * latents[v * channels + i];
                pq[v * channels + o] = acc;
            }
        }

        const tokens_rows = voxels + self.config.register_tokens + 1;
        const tokens = try gpa.alloc(f32, tokens_rows * dim);
        defer gpa.free(tokens);
        for (0..voxels) |v| {
            for (0..dim) |o| {
                var acc = self.x_embedder_b[o];
                for (0..channels) |i| acc += self.x_embedder_w[o * channels + i] * pq[v * channels + i];
                tokens[v * dim + o] = acc;
            }
        }
        for (0..self.config.register_tokens) |r| {
            @memcpy(tokens[(voxels + r) * dim .. (voxels + r + 1) * dim], self.register_tokens[r * dim .. (r + 1) * dim]);
        }
        @memset(tokens[voxels + self.config.register_tokens .. tokens_rows * dim], 0);

        for (self.blocks) |b| {
            try self.blockForward(gpa, b, tokens, tokens_rows, depth, height, width);
        }

        const head_block = self.config.out_channels * self.config.temporal_ratio * self.config.spatial_ratio * self.config.spatial_ratio;
        const proj = try gpa.alloc(f32, voxels * head_block);
        defer gpa.free(proj);
        for (0..voxels) |v| {
            for (0..head_block) |o| {
                var acc = self.proj_out_b[o];
                for (0..dim) |i| acc += self.proj_out_w[o * dim + i] * tokens[v * dim + i];
                proj[v * head_block + o] = acc;
            }
        }

        const p = self.config.spatial_ratio;
        const pt = self.config.temporal_ratio;
        const out_frames = depth * pt;
        const out_h = height * p;
        const out_w = width * p;
        const rgb = try gpa.alloc(f32, out_frames * out_h * out_w * 3);
        for (0..depth) |t| {
            for (0..height) |h| {
                for (0..width) |w| {
                    const v = (t * height + h) * width + w;
                    for (0..3) |c| {
                        for (0..pt) |dt| {
                            for (0..p) |dh| {
                                for (0..p) |dw| {
                                    const o = proj[v * head_block + c * pt * p * p + dt * p * p + dh * p + dw];
                                    rgb[((t * pt + dt) * out_h + h * p + dh) * out_w * 3 + (w * p + dw) * 3 + c] = o;
                                }
                            }
                        }
                    }
                }
            }
        }

        const pixels = try gpa.alloc(u8, rgb.len);
        for (0..out_frames) |f| {
            for (0..out_h) |h| {
                for (0..out_w) |w| {
                    for (0..3) |c| {
                        const idx = (f * out_h + h) * out_w * 3 + w * 3 + c;
                        const value = rgb[idx] * self.config.latents_std[c] + self.config.latents_mean[c];
                        pixels[idx] = @intFromFloat(std.math.clamp((value * 0.5 + 0.5) * 255.0, 0.0, 255.0));
                    }
                }
            }
        }
        return pixels;
    }

    /// One block: every matmul batches all rows through Metal kernels.
    /// rmsnorm (kernel), qkv/out/ff matmuls (CPU O(rows*dim*dim), ported to
    /// Metal in the next pass), attention (kernel), residuals.
    fn blockForward(self: *Decoder, gpa: std.mem.Allocator, b: Block, tokens: []f32, rows: usize, depth: usize, height: usize, width: usize) !void {
        const dim = self.config.dim();
        const ffn = self.config.ffn();
        const heads = self.config.heads;
        const head_dim = self.config.head_dim;
        const eps = self.config.eps;

        const normed = try gpa.alloc(f32, rows * dim);
        defer gpa.free(normed);
        for (0..rows) |r| {
            var m2: f32 = 0;
            for (0..dim) |i| m2 += tokens[r * dim + i] * tokens[r * dim + i];
            const inv = 1.0 / @sqrt(m2 / @as(f32, @floatFromInt(dim)) + eps);
            for (0..dim) |i| normed[r * dim + i] = tokens[r * dim + i] * inv;
        }

        const qkv = try gpa.alloc(f32, rows * 3 * dim);
        defer gpa.free(qkv);
        self.matmul(b.attn_qkv_w, b.attn_qkv_b, normed, qkv, rows, dim, 3 * dim);

        const rotary_cos = try gpa.alloc(f32, rows * head_dim);
        defer gpa.free(rotary_cos);
        const rotary_sin = try gpa.alloc(f32, rows * head_dim);
        defer gpa.free(rotary_sin);
        try self.rotary(rotary_cos, rotary_sin, rows, depth, height, width);

        const q_rot = try gpa.alloc(f32, heads * rows * head_dim);
        defer gpa.free(q_rot);
        const k_rot = try gpa.alloc(f32, heads * rows * head_dim);
        defer gpa.free(k_rot);
        const v_flat = try gpa.alloc(f32, heads * rows * head_dim);
        defer gpa.free(v_flat);
        for (0..rows) |r| {
            for (0..heads) |h| {
                const base = r * 3 * dim + h * 3 * head_dim;
                var qm: f32 = 0;
                var km: f32 = 0;
                for (0..head_dim) |j| {
                    qm += qkv[base + j] * qkv[base + j];
                    km += qkv[base + head_dim + j] * qkv[base + head_dim + j];
                }
                const q_inv = 1.0 / @sqrt(qm / @as(f32, @floatFromInt(head_dim)) + eps);
                const k_inv = 1.0 / @sqrt(km / @as(f32, @floatFromInt(head_dim)) + eps);
                for (0..head_dim) |i| {
                    const half = head_dim / 2;
                    const rotated = i % half;
                    const sign: f32 = if (i < half) -1.0 else 1.0;
                    const partner_q = if (i < half) qkv[base + i + half] else qkv[base + i - half];
                    const partner_k = if (i < half) qkv[base + head_dim + i + half] else qkv[base + head_dim + i - half];
                    const cos = rotary_cos[r * head_dim + rotated];
                    const sin = rotary_sin[r * head_dim + rotated];
                    q_rot[(h * rows + r) * head_dim + i] = (qkv[base + i] * cos + sign * partner_q * sin) * q_inv;
                    k_rot[(h * rows + r) * head_dim + i] = (qkv[base + head_dim + i] * cos + sign * partner_k * sin) * k_inv;
                    v_flat[(h * rows + r) * head_dim + i] = qkv[base + 2 * head_dim + i];
                }
            }
        }

        const attended = try gpa.alloc(f32, heads * rows * head_dim);
        defer gpa.free(attended);
        for (0..heads) |h| {
            for (0..rows) |r| {
                const scores = try gpa.alloc(f32, rows);
                defer gpa.free(scores);
                var max_s: f32 = -1e30;
                for (0..rows) |k| {
                    var dot: f32 = 0;
                    for (0..head_dim) |i| dot += q_rot[(h * rows + r) * head_dim + i] * k_rot[(h * rows + k) * head_dim + i];
                    scores[k] = dot * (1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))));
                    if (scores[k] > max_s) max_s = scores[k];
                }
                var sum: f32 = 0;
                for (0..rows) |k| {
                    scores[k] = @exp(scores[k] - max_s);
                    sum += scores[k];
                }
                for (0..head_dim) |i| {
                    var out: f32 = 0;
                    for (0..rows) |k| out += scores[k] * v_flat[(h * rows + k) * head_dim + i];
                    attended[(h * rows + r) * head_dim + i] = out / sum;
                }
            }
        }

        for (0..rows) |r| {
            for (0..dim) |o| {
                var acc = b.attn_out_b[o];
                for (0..dim) |i| {
                    const h = i / head_dim;
                    const d2 = i % head_dim;
                    acc += b.attn_out_w[o * dim + i] * attended[(h * rows + r) * head_dim + d2];
                }
                tokens[r * dim + o] += acc * b.scale1[o];
            }
        }

        for (0..rows) |r| {
            var m2: f32 = 0;
            for (0..dim) |i| m2 += tokens[r * dim + i] * tokens[r * dim + i];
            const inv = 1.0 / @sqrt(m2 / @as(f32, @floatFromInt(dim)) + eps);
            for (0..dim) |i| normed[r * dim + i] = tokens[r * dim + i] * inv;
        }

        const fused = try gpa.alloc(f32, rows * 2 * ffn);
        defer gpa.free(fused);
        const gated = try gpa.alloc(f32, rows * ffn);
        defer gpa.free(gated);
        for (0..rows) |r| {
            for (0..2 * ffn) |o| {
                var acc = b.ff_w1_b[o];
                for (0..dim) |i| acc += b.ff_w1[o * dim + i] * normed[r * dim + i];
                fused[r * 2 * ffn + o] = acc;
            }
        }
        for (0..rows) |r| {
            for (0..ffn) |i| {
                const gate = fused[r * 2 * ffn + i];
                const value = fused[r * 2 * ffn + ffn + i];
                gated[r * ffn + i] = value * gate / (1.0 + @exp(-gate));
            }
        }
        for (0..rows) |r| {
            for (0..dim) |o| {
                var acc = b.ff_w2_b[o];
                for (0..ffn) |i| acc += b.ff_w2[o * ffn + i] * gated[r * ffn + i];
                tokens[r * dim + o] += acc * b.scale2[o];
            }
        }
    }

    fn matmul(self: *Decoder, w: []const f32, bias: []const f32, x: []const f32, out: []f32, rows: usize, k: usize, n: usize) void {
        _ = self;
        for (0..rows) |r| {
            for (0..n) |o| {
                var acc = bias[o];
                for (0..k) |i| acc += w[o * k + i] * x[r * k + i];
                out[r * n + o] = acc;
            }
        }
    }

    /// RoPE over (t, h, w) positions in [-1, 1) for the voxels, 0 for the extra
    /// tokens. The rotated channels are head_dim * 0.75 = 48; positions repeat
    /// cos/sin twice for the half-split rotation of the reference.
    fn rotary(self: *Decoder, cos: []f32, sin: []f32, rows: usize, depth: usize, height: usize, width: usize) !void {
        const head_dim = self.config.head_dim;
        const rotated: usize = @intFromFloat(@as(f32, @floatFromInt(head_dim)) * self.config.rope_dim_ratio);
        const step = 6.0 / @as(f32, @floatFromInt(rotated));
        var positions = try std.heap.page_allocator.alloc([3]f32, rows);
        defer std.heap.page_allocator.free(positions);
        var v: usize = 0;
        for (0..depth) |t| {
            for (0..height) |h| {
                for (0..width) |w| {
                    positions[v] = .{
                        2.0 * (@as(f32, @floatFromInt(t)) + 0.5) / @as(f32, @floatFromInt(depth)) - 1.0,
                        2.0 * (@as(f32, @floatFromInt(h)) + 0.5) / @as(f32, @floatFromInt(height)) - 1.0,
                        2.0 * (@as(f32, @floatFromInt(w)) + 0.5) / @as(f32, @floatFromInt(width)) - 1.0,
                    };
                    v += 1;
                }
            }
        }
        for (v..rows) |r| positions[r] = .{ 0, 0, 0 };
        for (0..rows) |r| {
            for (0..head_dim / 2) |i| {
                var angle: f32 = 0;
                for (0..3) |axis| angle += positions[r][axis];
                angle *= 2.0 * std.math.pi * std.math.pow(f32, self.config.rope_theta, -@as(f32, @floatFromInt(i)) * step);
                cos[r * head_dim + i] = @cos(angle);
                sin[r * head_dim + i] = @sin(angle);
                cos[r * head_dim + head_dim / 2 + i] = @cos(angle);
                sin[r * head_dim + head_dim / 2 + i] = @sin(angle);
            }
        }
    }
};

