//! H3 video VAE decoder on Metal: ViT-only decoder. All matmuls, norms, ... ... ...

const std = @import("std");
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fclose(file: *anyopaque) c_int;
const tiling = @import("vae_video_tiling.zig");
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fread(ptr: *anyopaque, size: usize, count: usize, file: *anyopaque) usize;
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
    "h3_vae_layernorm",
    "h3_vae_postquant",
    "h3_vae_unpatch",
    "h3_vae_vextract",
    "h3_vae_attention_il",
    "h3_vae_enc_conv",
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
        if (wrapper.value.object.get("latents_mean")) |v| {
            if (v == .array) {
                const arr = v.array;
                const mean = try std.heap.page_allocator.alloc(f32, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    if (item == .float) mean[i] = @floatCast(item.float);
                    if (item == .integer) mean[i] = @floatFromInt(item.integer);
                }
                c.latents_mean = mean;
            }
        }
        if (wrapper.value.object.get("latents_std")) |v| {
            if (v == .array) {
                const arr = v.array;
                const std_arr = try std.heap.page_allocator.alloc(f32, arr.items.len);
                for (arr.items, 0..) |item, i| {
                    if (item == .float) std_arr[i] = @floatCast(item.float);
                    if (item == .integer) std_arr[i] = @floatFromInt(item.integer);
                }
                c.latents_std = std_arr;
            }
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
        cb.commit();
        cb.wait();
        @memcpy(out, out_buf.slice(f32, out.len));
    }
    /// Full decode: post_quant -> embedder -> registers -> 36 blocks on Metal ... ... ... ... ... ...
    pub fn frames(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize) ![]u8 {
        const p_ratio = self.config.spatial_ratio;
        const pt = self.config.temporal_ratio;
        const chunk = (self.config.clip_length + self.config.temporal_ratio - 1) / self.config.temporal_ratio;
        const lead = (self.config.temporal_ratio - (self.config.clip_length % self.config.temporal_ratio)) % self.config.temporal_ratio;
        const token_overlap = (chunk - (self.config.token_drop % chunk)) % chunk;
        const clips = (depth + self.config.token_drop) / chunk - @intFromBool(self.config.token_drop > 0);
        const out_frames = clips * (chunk * pt - lead) + (token_overlap * pt - lead);
        const out_h = height * p_ratio;
        const out_w = width * p_ratio;

        const canvas = try gpa.alloc(f32, out_frames * out_h * out_w * 3);
        @memset(canvas, 0);

        std.debug.print("canvas ready", .{});
        var clip_start: usize = 0;
        var clip_index: usize = 0;
        var overlap_clip: ?[]f32 = null; // previous clip's part1 pixel (frames, out_h, out_w, 3)
        defer if (overlap_clip) |o| gpa.free(o);
        var last_gbase: usize = 0;
        var last_clip_frames: usize = 0;
        const frame_overlap_px: usize = if (token_overlap * pt > lead) token_overlap * pt - lead else 0;
        while (clip_start < depth) : (clip_start += chunk) {
            const clip_depth = @min(chunk + token_overlap, depth - clip_start);
            const clip_px_frames = clip_depth * pt; // raw decoded pixel frames per tile
            const grid = try tileStarts(gpa, height * p_ratio, self.config.tile, self.config.tile_overlap, p_ratio);
            defer gpa.free(grid.starts);
            defer gpa.free(grid.overlaps);
            const gridx = try tileStarts(gpa, width * p_ratio, self.config.tile, self.config.tile_overlap, p_ratio);
            defer gpa.free(gridx.starts);
            defer gpa.free(gridx.overlaps);
            const ny = grid.starts.len;
            const nx = gridx.starts.len;
            const tile_h_lat = @min(self.config.tile, height * p_ratio) / p_ratio;
            const tile_w_lat = @min(self.config.tile, width * p_ratio) / p_ratio;
            const raw = try gpa.alloc([]u8, ny * nx);
            defer {
                for (raw) |r| gpa.free(r);
                gpa.free(raw);
            }
            for (grid.starts, 0..) |yp, i| {
                for (gridx.starts, 0..) |xp, j| {
                    const tile_h = @min(tile_h_lat, height - yp / p_ratio);
                    const tile_w = @min(tile_w_lat, width - xp / p_ratio);
                    const tile_voxels = clip_depth * tile_h * tile_w;
                    const tile_lat = try gpa.alloc(f32, tile_voxels * self.config.latent_channels);
                    defer gpa.free(tile_lat);
                    var v: usize = 0;
                    for (0..clip_depth) |t| {
                        for (0..tile_h) |h| {
                            for (0..tile_w) |w| {
                                for (0..24) |c| {
                                    const src = ((c * depth) + (clip_start + t)) * height * width + (yp / p_ratio + h) * width + (xp / p_ratio + w);
                                    tile_lat[v * 24 + c] = latents[src];
                                }
                                v += 1;
                            }
                        }
                    }
                    const piece_len_bytes = clip_px_frames * tile_h * p_ratio * tile_w * p_ratio * 3;
                    const piece = try gpa.alloc(u8, piece_len_bytes);
                    defer gpa.free(piece);
                    try self.decodePiece(gpa, tile_lat, clip_depth, tile_h, tile_w, piece);
                    const rawu = try gpa.alloc(u8, piece_len_bytes);
                    @memcpy(rawu, piece);
                    raw[i * nx + j] = rawu;
                }
            }
            const clip_pix = try gpa.alloc(u8, clip_px_frames * out_h * out_w * 3);
            defer gpa.free(clip_pix);
            @memset(clip_pix, 0);
            for (grid.starts, 0..) |yp, i| {
                for (gridx.starts, 0..) |xp, j| {
                    const tile_h = @min(tile_h_lat, height - yp / p_ratio);
                    const tile_w = @min(tile_w_lat, width - xp / p_ratio);
                    const th_px = tile_h * p_ratio;
                    const tw_px = tile_w * p_ratio;
                    const npx = clip_px_frames * th_px * tw_px * 3;
                    const piece = try gpa.alloc(f32, npx);
                    defer gpa.free(piece);
                    for (raw[i * nx + j], 0..) |b2, k2| piece[k2] = @floatFromInt(b2);
                    if (i > 0) {
                        const up_raw = raw[(i - 1) * nx + j];
                        const up_th = @min(tile_h_lat, height - grid.starts[i - 1] / p_ratio) * p_ratio;
                        const up_tw = @min(tile_w_lat, width - gridx.starts[j] / p_ratio) * p_ratio;
                        const up = try gpa.alloc(f32, up_raw.len);
                        defer gpa.free(up);
                        for (up_raw, 0..) |b2, k2| up[k2] = @floatFromInt(b2);
                        const extent = @min(grid.overlaps[i - 1], @min(up_th, th_px));
                        blendTail(up, clip_px_frames, up_th, up_tw, piece, th_px, tw_px, extent, 0);
                    }
                    if (j > 0) {
                        const left_raw = raw[i * nx + j - 1];
                        const left_th = @min(tile_h_lat, height - grid.starts[i] / p_ratio) * p_ratio;
                        const left_tw = @min(tile_w_lat, width - gridx.starts[j - 1] / p_ratio) * p_ratio;
                        const left = try gpa.alloc(f32, left_raw.len);
                        defer gpa.free(left);
                        for (left_raw, 0..) |b2, k2| left[k2] = @floatFromInt(b2);
                        const extent = @min(gridx.overlaps[j - 1], @min(left_tw, tw_px));
                        blendTail(left, clip_px_frames, left_th, left_tw, piece, th_px, tw_px, extent, 1);
                    }
                    const crop_b: usize = if (i < ny - 1) grid.overlaps[i] else 0;
                    const crop_r: usize = if (j < nx - 1) gridx.overlaps[j] else 0;
                    const write_h = th_px - crop_b;
                    const write_w = tw_px - crop_r;
                    for (0..clip_px_frames) |f| {
                        for (0..write_h) |ph| {
                            for (0..write_w) |pw| {
                                const gy = yp + ph;
                                const gx = xp + pw;
                                const ci2 = (f * out_h + gy) * out_w + gx;
                                const src = (f * th_px + ph) * tw_px * 3 + pw * 3;
                                clip_pix[ci2 * 3 + 0] = @intFromFloat(@round(piece[src + 0]));
                                clip_pix[ci2 * 3 + 1] = @intFromFloat(@round(piece[src + 1]));
                                clip_pix[ci2 * 3 + 2] = @intFromFloat(@round(piece[src + 2]));
                            }
                        }
                    }
                }
            }
            const frame_n = out_h * out_w * 3;
            {
                var sum2: u64 = 0;
                for (clip_pix) |v| sum2 += v;
                std.debug.print("DBG4: clip_pix sum={d} raw0={d}", .{ sum2, raw[0][0] });
            }
            const part0_frames_n = chunk * pt - lead;
            const extent: usize = if (overlap_clip != null) @min(frame_overlap_px, part0_frames_n) else 0;
            for (0..part0_frames_n) |f| {
                const gf = clip_index * part0_frames_n + f;
                if (gf >= out_frames) break;
                if (f < extent) {
                    const ov = overlap_clip.?;
                    const ov_frames = ov.len / frame_n;
                    const ovf = ov_frames - extent + f;
                    const ramp = @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(extent));
                    for (0..frame_n) |k| {
                        const b = @as(f32, @floatFromInt(clip_pix[(lead + f) * frame_n + k]));
                        canvas[gf * frame_n + k] = ov[ovf * frame_n + k] * (1.0 - ramp) + b * ramp;
                    }
                } else {
                    for (0..frame_n) |k| {
                        canvas[gf * frame_n + k] = @as(f32, @floatFromInt(clip_pix[(lead + f) * frame_n + k]));
                    }
                }
            }
            const p1_start = chunk * pt + lead;
            const p1_frames_n = if (clip_px_frames > p1_start) clip_px_frames - p1_start else 0;
            if (overlap_clip) |o| gpa.free(o);
            overlap_clip = try gpa.alloc(f32, p1_frames_n * frame_n);
            for (0..p1_frames_n) |f| {
                for (0..frame_n) |k| {
                    overlap_clip.?[f * frame_n + k] = @as(f32, @floatFromInt(clip_pix[(p1_start + f) * frame_n + k]));
                }
            }
            last_gbase = clip_index * part0_frames_n + part0_frames_n;
            last_clip_frames = p1_frames_n;
            clip_index += 1;
        }
        if (overlap_clip) |ov| {
            const frame_n = out_h * out_w * 3;
            for (0..last_clip_frames) |f| {
                const gf = last_gbase + f;
                if (gf >= out_frames) break;
                for (0..frame_n) |k| {
                    canvas[gf * frame_n + k] = ov[f * frame_n + k];
                }
            }
        }
        const pixels = try gpa.alloc(u8, canvas.len);
        for (0..out_frames) |f| {
            for (0..out_h) |h| {
                for (0..out_w) |w| {
                    const ci = (f * out_h + h) * out_w + w;
                    for (0..3) |c| {
                        const value = canvas[ci * 3 + c];
                        pixels[ci * 3 + c] = @intFromFloat(std.math.clamp(value, 0.0, 255.0));
                    }
                }
            }
        }
        return pixels;
    }

    /// One piece (clip x tile) through the full ViT pass on Metal.
    pub fn decodePiece(self: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize, out_rgb_bytes: []u8) !void {
        return tiling.decodePiece(self, gpa, latents, depth, height, width, out_rgb_bytes);
    }
};

fn tileStarts(gpa: std.mem.Allocator, length: usize, tile: usize, min_overlap: usize, ratio: usize) !struct { starts: []usize, overlaps: []usize } {
    if (tile >= length) {
        const st = try gpa.alloc(usize, 1);
        st[0] = 0;
        const ov = try gpa.alloc(usize, 0);
        return .{ .starts = st, .overlaps = ov };
    }
    var count = (length + tile - 1) / tile;
    while (tile * count - min_overlap * (count - 1) < length) count += 1;
    const overlaps = try gpa.alloc(usize, count - 1);
    @memset(overlaps, min_overlap);
    var remaining = tile * count - min_overlap * (count - 1) - length;
    var i: usize = 0;
    while (remaining >= ratio) : (remaining -= ratio) {
        overlaps[i % (count - 1)] += ratio;
        i += 1;
    }
    const starts = try gpa.alloc(usize, count);
    starts[0] = 0;
    for (0..count - 1) |k| starts[k + 1] = starts[k] + tile - overlaps[k];
    return .{ .starts = starts, .overlaps = overlaps };
}

fn blendTail(a: []const f32, frames: usize, a_h: usize, a_w: usize, piece: []f32, p_h: usize, p_w: usize, extent: usize, axis: usize) void {
    for (0..frames) |f| {
        for (0..extent) |k| {
            const ramp = @as(f32, @floatFromInt(k)) / @as(f32, @floatFromInt(extent));
            if (axis == 0) {
                const a_row = (f * a_h + (a_h - extent + k)) * a_w * 3;
                const p_row = (f * p_h + k) * p_w * 3;
                for (0..a_w * 3) |c| {
                    piece[p_row + c] = a[a_row + c] * (1.0 - ramp) + piece[p_row + c] * ramp;
                }
            } else {
                for (0..p_h) |r| {
                    const a_base = (f * a_h + r) * a_w * 3 + (a_w - extent) * 3;
                    const p_base = (f * p_h + r) * p_w * 3;
                    for (0..extent * 3) |c| {
                        piece[p_base + c] = a[a_base + c] * (1.0 - ramp) + piece[p_base + c] * ramp;
                    }
                }
            }
        }
    }
}
