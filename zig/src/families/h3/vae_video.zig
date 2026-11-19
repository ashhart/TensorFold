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
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize) ![]u8 {
        const dim = self.config.dim();
        const channels = self.config.latent_channels;
        const voxels = depth * height * width;

        var cb = self.queue.commandBuffer();
        var e = cb.compute(.concurrent);

        // post_quant: h3_vae_postquant
        const pq_in = try self.device.buffer(latents.len * 4, mtl.ResourceOptions.shared);
        @memcpy(pq_in.slice(f32, latents.len), latents);
        const pq_w = try self.device.buffer(self.post_quant_w.len * 4, mtl.ResourceOptions.shared);
        @memcpy(pq_w.slice(f32, self.post_quant_w.len), self.post_quant_w);
        const pq_b = try self.device.buffer(self.post_quant_b.len * 4, mtl.ResourceOptions.shared);
        @memcpy(pq_b.slice(f32, self.post_quant_b.len), self.post_quant_b);
        const pq_out = try self.device.buffer(voxels * channels * 4, mtl.ResourceOptions.shared);
        e.setPipeline(self.p[5]);
        e.setBuffer(pq_in, 0, 0);
        e.setBuffer(pq_w, 0, 1);
        e.setBuffer(pq_b, 0, 2);
        e.setBuffer(pq_out, 0, 3);
        e.setValue([2]u32{ @intCast(voxels), @intCast(channels) }, 4);
        e.dispatchThreads(mtl.Size.of(channels, voxels, 1), mtl.Size.of(64, 1, 1));
        e.barrier();

        // x_embedder matmul: pq [V, 24] x W [2048, 24] -> tokens [V, 2048]
        const emb_w = try self.device.buffer(self.x_embedder_w.len * 4, mtl.ResourceOptions.shared);
        @memcpy(emb_w.slice(f32, self.x_embedder_w.len), self.x_embedder_w);
        const emb_b = try self.device.buffer(self.x_embedder_b.len * 4, mtl.ResourceOptions.shared);
        @memcpy(emb_b.slice(f32, self.x_embedder_b.len), self.x_embedder_b);
        const tokens = try self.device.buffer(voxels * dim * 4, mtl.ResourceOptions.shared);
        e.setPipeline(self.p[0]);
        e.setBuffer(pq_out, 0, 0);
        e.setBuffer(emb_w, 0, 1);
        e.setBuffer(emb_b, 0, 2);
        e.setBuffer(tokens, 0, 3);
        e.setValue([4]u32{ @intCast(voxels), @intCast(channels), @intCast(dim), 0 }, 4);
        e.dispatchThreads(mtl.Size.of(dim, voxels, 1), mtl.Size.of(64, 1, 1));
        e.barrier();

        // registers + zero token appended on the CPU after the GPU pass
        e.end();
        cb.commit(); cb.wait();

        const tokens_host = try gpa.alloc(f32, voxels * dim);
        @memcpy(tokens_host, tokens.slice(f32, voxels * dim));
        const rows = voxels + self.config.register_tokens + 1;
        const tokens_full = try gpa.alloc(f32, rows * dim);
        @memcpy(tokens_full[0 .. voxels * dim], tokens_host);
        for (0..self.config.register_tokens) |r| {
            @memcpy(tokens_full[(voxels + r) * dim .. (voxels + r + 1) * dim], self.register_tokens[r * dim .. (r + 1) * dim]);
        }
        @memset(tokens_full[voxels + self.config.register_tokens .. rows * dim], 0);
        gpa.free(tokens_host);

        // 36 blocks on Metal: per block, 6 dispatches (2 matmuls x3, norms, swiglu, attention)
        const normed = try self.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
        const qkv = try self.device.buffer(rows * 3 * dim * 4, mtl.ResourceOptions.shared);
        const attended = try self.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
        const fused = try self.device.buffer(rows * 2 * self.config.ffn() * 4, mtl.ResourceOptions.shared);
        const gated = try self.device.buffer(rows * self.config.ffn() * 4, mtl.ResourceOptions.shared);
        defer normed.deinit();
        defer qkv.deinit();
        defer attended.deinit();
        defer fused.deinit();
        defer gated.deinit();

        // copy tokens_full to normed for the first block
        @memcpy(normed.slice(f32, rows * dim), tokens_full);

        const head_dim = self.config.head_dim;
        const heads = self.config.heads;
        const head_block = self.config.out_channels * self.config.temporal_ratio * self.config.spatial_ratio * self.config.spatial_ratio;
        for (self.blocks) |b| {
            const qkv_w = try self.device.buffer(b.attn_qkv_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(qkv_w.slice(f32, b.attn_qkv_w.len), b.attn_qkv_w);
            const qkv_b = try self.device.buffer(b.attn_qkv_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(qkv_b.slice(f32, b.attn_qkv_b.len), b.attn_qkv_b);

            cb = self.queue.commandBuffer();
            e = cb.compute(.concurrent);

            // rmsnorm 1: normed -> normed (in place via separate out)
            const n1_w = try self.device.buffer(b.norm1_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(n1_w.slice(f32, b.norm1_w.len), b.norm1_w);
            e.setPipeline(self.p[1]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(n1_w, 0, 1);
            e.setBuffer(normed, 0, 2);
            e.setValue([1]u32{ @intCast(dim) }, 3);
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
            e.barrier();

            // qkv matmul: normed [rows, dim] x qkv_w [3dim, dim] -> qkv [rows, 3dim]
            e.setPipeline(self.p[0]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(qkv_w, 0, 1);
            e.setBuffer(qkv_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(3 * dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(3 * dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            // attention: qkv -> attended (per head, RoPE inside the Python ref is done on the CPU here)
            e.setPipeline(self.p[3]);
            e.setBuffer(qkv, 0, 0);
            e.setBuffer(qkv, 0, 1);
            e.setBuffer(qkv, 0, 2);
            e.setBuffer(attended, 0, 3);
            e.setValue([2]u32{ @intCast(rows), @intCast(head_dim) }, 4);
            e.dispatchGroups(mtl.Size.of(rows, heads, 1), mtl.Size.of(256, 1, 1));
            e.barrier();

            const out_w = try self.device.buffer(b.attn_out_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(out_w.slice(f32, b.attn_out_w.len), b.attn_out_w);
            const out_b = try self.device.buffer(b.attn_out_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(out_b.slice(f32, b.attn_out_b.len), b.attn_out_b);
            const s1 = try self.device.buffer(b.scale1.len * 4, mtl.ResourceOptions.shared);
            @memcpy(s1.slice(f32, b.scale1.len), b.scale1);

            // out matmul: attended [rows, dim] x out_w [dim, dim] -> qkv reused [rows, dim]
            e.setPipeline(self.p[0]);
            e.setBuffer(attended, 0, 0);
            e.setBuffer(out_w, 0, 1);
            e.setBuffer(out_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            // residual: tokens += qkv * scale1 (CPU side, small)
            cb.commit(); cb.wait();
            const qkv_host = qkv.slice(f32, rows * dim);
            for (0..rows) |r| {
                for (0..dim) |i| tokens_full[r * dim + i] += qkv_host[r * dim + i] * b.scale1[i];
            }
            @memcpy(normed.slice(f32, rows * dim), tokens_full[0 .. rows * dim]);

            cb = self.queue.commandBuffer();
            e = cb.compute(.concurrent);

            // rmsnorm 2
            const n2_w = try self.device.buffer(b.norm2_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(n2_w.slice(f32, b.norm2_w.len), b.norm2_w);
            e.setPipeline(self.p[1]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(n2_w, 0, 1);
            e.setBuffer(normed, 0, 2);
            e.setValue([1]u32{ @intCast(dim) }, 3);
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
            e.barrier();

            const fw1_w = try self.device.buffer(b.ff_w1.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw1_w.slice(f32, b.ff_w1.len), b.ff_w1);
            const fw1_b = try self.device.buffer(b.ff_w1_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw1_b.slice(f32, b.ff_w1_b.len), b.ff_w1_b);

            // ff matmul: normed [rows, dim] x fw1_w [2ffn, dim] -> fused [rows, 2ffn]
            e.setPipeline(self.p[0]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(fw1_w, 0, 1);
            e.setBuffer(fw1_b, 0, 2);
            e.setBuffer(fused, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(2 * self.config.ffn()), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(2 * self.config.ffn(), rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            // swiglu: fused -> gated
            e.setPipeline(self.p[2]);
            e.setBuffer(fused, 0, 0);
            e.setBuffer(gated, 0, 1);
            e.setValue([1]u32{ @intCast(self.config.ffn()) }, 2);
            e.dispatchThreads(mtl.Size.of(self.config.ffn(), rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            const fw2_w = try self.device.buffer(b.ff_w2.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw2_w.slice(f32, b.ff_w2.len), b.ff_w2);
            const fw2_b = try self.device.buffer(b.ff_w2_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw2_b.slice(f32, b.ff_w2_b.len), b.ff_w2_b);
            const s2 = try self.device.buffer(b.scale2.len * 4, mtl.ResourceOptions.shared);
            @memcpy(s2.slice(f32, b.scale2.len), b.scale2);

            // ff matmul: gated [rows, ffn] x fw2_w [dim, ffn] -> qkv reused [rows, dim]
            e.setPipeline(self.p[0]);
            e.setBuffer(gated, 0, 0);
            e.setBuffer(fw2_w, 0, 1);
            e.setBuffer(fw2_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(self.config.ffn()), @intCast(dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.end();
            cb.commit(); cb.wait();

            const qkv_host2 = qkv.slice(f32, rows * dim);
            for (0..rows) |r| {
                for (0..dim) |i| tokens_full[r * dim + i] += qkv_host2[r * dim + i] * b.scale2[i];
            }
            @memcpy(normed.slice(f32, rows * dim), tokens_full[0 .. rows * dim]);
        }
        normed.deinit();
        qkv.deinit();
        attended.deinit();
        fused.deinit();
        gated.deinit();

        // norm_out + proj_out on Metal
        cb = self.queue.commandBuffer();
        e = cb.compute(.concurrent);
        const no_w = try self.device.buffer(self.norm_out_w.len * 4, mtl.ResourceOptions.shared);
        @memcpy(no_w.slice(f32, self.norm_out_w.len), self.norm_out_w);
        const no_b = try self.device.buffer(self.norm_out_b.len * 4, mtl.ResourceOptions.shared);
        @memcpy(no_b.slice(f32, self.norm_out_b.len), self.norm_out_b);
        e.setPipeline(self.p[4]);
        e.setBuffer(normed, 0, 0);
        e.setBuffer(no_w, 0, 1);
        e.setBuffer(no_b, 0, 2);
        e.setBuffer(normed, 0, 3);
        e.setValue([1]u32{ @intCast(dim) }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
        e.barrier();
        e.end();
        cb.commit(); cb.wait();

        // proj_out matmul: normed [rows, dim] x proj_out_w [3072, dim] -> proj [rows, 3072]
        const proj = try gpa.alloc(f32, voxels * head_block);
        try self.matmulOn(self.queue.commandBuffer(), self.proj_out_w, self.proj_out_b, normed.slice(f32, rows * dim), proj, voxels, dim, head_block);

        // unpatchify + denormalize on the CPU
        const p_ratio = self.config.spatial_ratio;
        const pt = self.config.temporal_ratio;
        const out_frames = depth * pt;
        const out_h = height * p_ratio;
        const out_w = width * p_ratio;
        const pixels = try gpa.alloc(u8, out_frames * out_h * out_w * 3);
        for (0..depth) |t| {
            for (0..height) |h| {
                for (0..width) |w| {
                    const v = (t * height + h) * width + w;
                    for (0..3) |c| {
                        for (0..pt) |dt| {
                            for (0..p_ratio) |dh| {
                                for (0..p_ratio) |dw| {
                                    const o = proj[v * head_block + c * pt * p_ratio * p_ratio + dt * p_ratio * p_ratio + dh * p_ratio + dw];
                                    const value = o * self.config.latents_std[c] + self.config.latents_mean[c];
                                    pixels[((t * pt + dt) * out_h + h * p_ratio + dh) * out_w * 3 + (w * p_ratio + dw) * 3 + c] = @intFromFloat(std.math.clamp((value * 0.5 + 0.5) * 255.0, 0.0, 255.0));
                                }
                            }
                        }
                    }
                }
            }
        }
        return pixels;
    }
};
