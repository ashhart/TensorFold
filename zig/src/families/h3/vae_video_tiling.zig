//! decodePiece: one ViT pass for a clip x tile piece of the H3 video VAE decoder.
const std = @import("std");
const mtl = @import("metal");
const Decoder = @import("vae_video.zig").Decoder;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fclose(file: *anyopaque) c_int;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fread(ptr: *anyopaque, size: usize, count: usize, file: *anyopaque) usize;

/// One piece (clip x tile) through the full ViT pass on Metal.
pub fn decodePiece(d: *Decoder, gpa: std.mem.Allocator, latents: []const f32, depth: usize, height: usize, width: usize, out_rgb_bytes: []u8) !void {
    const dim = d.config.dim();
    const voxels = depth * height * width;
    const rows = voxels + d.config.register_tokens + 1;

    var cb = d.queue.commandBuffer();
    var e = cb.compute(.concurrent);

    const pq_in = try d.device.buffer(latents.len * 4, mtl.ResourceOptions.shared);
    @memcpy(pq_in.slice(f32, latents.len), latents);
    const pq_w = try d.device.buffer(d.post_quant_w.len * 4, mtl.ResourceOptions.shared);
    @memcpy(pq_w.slice(f32, d.post_quant_w.len), d.post_quant_w);
    const pq_b = try d.device.buffer(d.post_quant_b.len * 4, mtl.ResourceOptions.shared);
    @memcpy(pq_b.slice(f32, d.post_quant_b.len), d.post_quant_b);
    const pq_out = try d.device.buffer(voxels * 24 * 4, mtl.ResourceOptions.shared);
    e.setPipeline(d.p[4]);
    e.setBuffer(pq_in, 0, 0);
    e.setBuffer(pq_w, 0, 1);
    e.setBuffer(pq_b, 0, 2);
    e.setBuffer(pq_out, 0, 3);
    e.setValue([2]u32{ @intCast(voxels), @intCast(24) }, 4);
    e.dispatchThreads(mtl.Size.of(24 * voxels, 1, 1), mtl.Size.of(256, 1, 1));
    e.barrier();

    const emb_w = try d.device.buffer(d.x_embedder_w.len * 4, mtl.ResourceOptions.shared);
    @memcpy(emb_w.slice(f32, d.x_embedder_w.len), d.x_embedder_w);
    const emb_b = try d.device.buffer(d.x_embedder_b.len * 4, mtl.ResourceOptions.shared);
    @memcpy(emb_b.slice(f32, d.x_embedder_b.len), d.x_embedder_b);
    const tokens = try d.device.buffer(voxels * dim * 4, mtl.ResourceOptions.shared);
    e.setPipeline(d.p[0]);
    e.setBuffer(pq_out, 0, 0);
    e.setBuffer(emb_w, 0, 1);
    e.setBuffer(emb_b, 0, 2);
    e.setBuffer(tokens, 0, 3);
    e.setValue([4]u32{ @intCast(voxels), @intCast(24), @intCast(dim), 0 }, 4);
    e.dispatchThreads(mtl.Size.of(dim, voxels, 1), mtl.Size.of(64, 1, 1));
    e.end();
    cb.commit();
    cb.wait();

    const tokens_host = try gpa.alloc(f32, voxels * dim);
    @memcpy(tokens_host, tokens.slice(f32, voxels * dim));
    const tokens_full = try gpa.alloc(f32, rows * dim);
    @memcpy(tokens_full[0 .. voxels * dim], tokens_host);
    for (0..d.config.register_tokens) |r| {
        @memcpy(tokens_full[(voxels + r) * dim .. (voxels + r + 1) * dim], d.register_tokens[r * dim .. (r + 1) * dim]);
    }
    @memset(tokens_full[(voxels + d.config.register_tokens) * dim .. rows * dim], 0);
    gpa.free(tokens_host);
    {
        const th = try gpa.alloc(f32, voxels * dim);
        defer gpa.free(th);
        @memcpy(th, tokens.slice(f32, voxels * dim));
        std.debug.print("EMB row0[0]={d:.5} row1[0]={d:.5} row2[0]={d:.5}", .{ th[0], th[2048], th[4096] });
    }
    std.debug.print("stage: pq+embedder+registers done", .{});
    pq_in.deinit();
    pq_w.deinit();
    pq_b.deinit();
    pq_out.deinit();
    emb_w.deinit();
    emb_b.deinit();
    tokens.deinit();
    if (getenv("TOKENS_INPUT") != null) {
        const tip = "tokens-dump.f32";
        const tipz: [*:0]const u8 = tip;
        const ft = fopen(tipz, "rb") orelse return error.CannotOpen;
        _ = fread(tokens_full.ptr, 4, tokens_full.len, ft);
        _ = fclose(ft);
        std.debug.print("TOKENS_INPUT loaded", .{});
    }

    // Stage 2: one block on Metal
    const normed = try d.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
    const qkv = try d.device.buffer(rows * 3 * dim * 4, mtl.ResourceOptions.shared);
    const attended = try d.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
    defer normed.deinit();
    defer qkv.deinit();
    defer attended.deinit();
    @memcpy(normed.slice(f32, rows * dim), tokens_full);

    cb = d.queue.commandBuffer();
    e = cb.compute(.concurrent);
    e.end();
    cb.commit();
    cb.wait();
    const fused = try d.device.buffer(rows * 2 * d.config.ffn() * 4, mtl.ResourceOptions.shared);
    const gated = try d.device.buffer(rows * d.config.ffn() * 4, mtl.ResourceOptions.shared);
    const rope = try d.device.buffer(rows * 48 * 2 * 4, mtl.ResourceOptions.shared);
    const rope_h = rope.slice(f32, rows * 48);
    const rope_s = rope.slice(f32, rows * 48)[rows * 48 ..];
    var pos: usize = 0;
    for (0..depth) |t| {
        for (0..height) |h| {
            for (0..width) |w| {
                const pt = 2.0 * (@as(f32, @floatFromInt(t)) + 0.5) / @as(f32, @floatFromInt(depth)) - 1.0;
                const ph = 2.0 * (@as(f32, @floatFromInt(h)) + 0.5) / @as(f32, @floatFromInt(height)) - 1.0;
                const pw = 2.0 * (@as(f32, @floatFromInt(w)) + 0.5) / @as(f32, @floatFromInt(width)) - 1.0;
                const pos_val = [3]f32{ pt, ph, pw };
                var k2: usize = 0;
                for (0..3) |axis| {
                    for (0..8) |f| {
                        const inv = std.math.pow(f32, 100.0, -@as(f32, @floatFromInt(f)) * 0.125);
                        const a = 2.0 * std.math.pi * pos_val[axis] * inv;
                        rope_h[pos * 48 + k2] = @cos(a);
                        rope_s[pos * 48 + k2] = @sin(a);
                        rope_h[pos * 48 + 24 + k2] = @cos(a);
                        rope_s[pos * 48 + 24 + k2] = @sin(a);
                        k2 += 1;
                    }
                }
                pos += 1;
            }
        }
    }
    for (pos..rows) |rr| {
        for (0..48) |k2| {
            rope_h[rr * 48 + k2] = 1.0;
            rope_s[rr * 48 + k2] = 0.0;
        }
    }
    if (getenv("SKIP_BLOCKS") == null) {
        for (d.blocks) |b| {
            @memcpy(normed.slice(f32, rows * dim), tokens_full);
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);

            const n1_w = try d.device.buffer(b.norm1_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(n1_w.slice(f32, b.norm1_w.len), b.norm1_w);
            const normed_out = try d.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
            e.setPipeline(d.p[1]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(n1_w, 0, 1);
            e.setBuffer(normed_out, 0, 2);
            e.setValue([1]u32{@intCast(dim)}, 3);
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);

            const qkv_w = try d.device.buffer(b.attn_qkv_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(qkv_w.slice(f32, b.attn_qkv_w.len), b.attn_qkv_w);
            const qkv_b = try d.device.buffer(b.attn_qkv_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(qkv_b.slice(f32, b.attn_qkv_b.len), b.attn_qkv_b);
            e.setPipeline(d.p[0]);
            e.setBuffer(normed_out, 0, 0);
            e.setBuffer(qkv_w, 0, 1);
            e.setBuffer(qkv_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(3 * dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(3 * dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);
            e.barrier();

            // V extraction in the interleaved layout: attended[h][r][i] = qkv[r][h*3*hd + 2*hd + i] ... ... ...
            const rope_buf = try d.device.buffer(rows * 48 * 2 * 4, mtl.ResourceOptions.shared);
            const rope_cos = rope_buf.slice(f32, rows * 48);
            const rope_sin = rope_buf.slice(f32, rows * 48)[rows * 48 ..];
            var rpos: usize = 0;
            for (0..depth) |t| {
                for (0..height) |h| {
                    for (0..width) |w| {
                        const pt = 2.0 * (@as(f32, @floatFromInt(t)) + 0.5) / @as(f32, @floatFromInt(depth)) - 1.0;
                        const ph = 2.0 * (@as(f32, @floatFromInt(h)) + 0.5) / @as(f32, @floatFromInt(height)) - 1.0;
                        const pw = 2.0 * (@as(f32, @floatFromInt(w)) + 0.5) / @as(f32, @floatFromInt(width)) - 1.0;
                        const pvals = [3]f32{ pt, ph, pw };
                        var k2: usize = 0;
                        for (0..3) |axis| {
                            for (0..8) |f| {
                                const inv = std.math.pow(f32, 100.0, -@as(f32, @floatFromInt(f)) * 0.125);
                                const a = 2.0 * std.math.pi * pvals[axis] * inv;
                                rope_cos[rpos * 48 + k2] = @cos(a);
                                rope_sin[rpos * 48 + k2] = @sin(a);
                                rope_cos[rpos * 48 + 24 + k2] = @cos(a);
                                rope_sin[rpos * 48 + 24 + k2] = @sin(a);
                                k2 += 1;
                            }
                        }
                        rpos += 1;
                    }
                }
            }
            for (rpos..rows) |rr| {
                for (0..48) |k2| {
                    rope_cos[rr * 48 + k2] = 1.0;
                    rope_sin[rr * 48 + k2] = 0.0;
                }
            }
            e.setPipeline(d.p[7]);
            e.setBuffer(qkv, 0, 0);
            e.setBuffer(attended, 0, 1);
            e.setBuffer(rope_buf, 0, 2);
            e.setValue([2]u32{ @intCast(rows), 0 }, 3);
            e.dispatchThreads(mtl.Size.of(rows * 32, 1, 1), mtl.Size.of(1, 1, 1));
            e.barrier();
            e.end();
            cb.commit();
            cb.wait();
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);

            const out_w = try d.device.buffer(b.attn_out_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(out_w.slice(f32, b.attn_out_w.len), b.attn_out_w);
            const out_b = try d.device.buffer(b.attn_out_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(out_b.slice(f32, b.attn_out_b.len), b.attn_out_b);
            e.setPipeline(d.p[0]);
            e.setBuffer(attended, 0, 0);
            e.setBuffer(out_w, 0, 1);
            e.setBuffer(out_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.end();
            cb.commit();
            cb.wait();

            const qkv_host = qkv.slice(f32, rows * dim);
            for (0..rows) |r| {
                for (0..dim) |i| tokens_full[r * dim + i] += qkv_host[r * dim + i] * b.scale1[i];
            }
            n1_w.deinit();
            qkv_w.deinit();
            qkv_b.deinit();
            out_w.deinit();
            out_b.deinit();
            normed_out.deinit();
            rope_buf.deinit();
            std.debug.print("stage: block1 attn done", .{});

            // Stage 3: ff half
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);
            @memcpy(normed.slice(f32, rows * dim), tokens_full);

            const n2_w = try d.device.buffer(b.norm2_w.len * 4, mtl.ResourceOptions.shared);
            @memcpy(n2_w.slice(f32, b.norm2_w.len), b.norm2_w);
            e.setPipeline(d.p[1]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(n2_w, 0, 1);
            e.setBuffer(normed, 0, 2);
            e.setValue([1]u32{@intCast(dim)}, 3);
            e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
            e.end();
            cb.commit();
            cb.wait();
            cb = d.queue.commandBuffer();
            e = cb.compute(.concurrent);

            const fw1_w = try d.device.buffer(b.ff_w1.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw1_w.slice(f32, b.ff_w1.len), b.ff_w1);
            const fw1_b = try d.device.buffer(b.ff_w1_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw1_b.slice(f32, b.ff_w1_b.len), b.ff_w1_b);
            e.setPipeline(d.p[0]);
            e.setBuffer(normed, 0, 0);
            e.setBuffer(fw1_w, 0, 1);
            e.setBuffer(fw1_b, 0, 2);
            e.setBuffer(fused, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(dim), @intCast(2 * d.config.ffn()), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(2 * d.config.ffn(), rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            e.setPipeline(d.p[2]);
            e.setBuffer(fused, 0, 0);
            e.setBuffer(gated, 0, 1);
            e.setValue([1]u32{@intCast(d.config.ffn())}, 2);
            e.dispatchThreads(mtl.Size.of(d.config.ffn(), rows, 1), mtl.Size.of(64, 1, 1));
            e.barrier();

            const fw2_w = try d.device.buffer(b.ff_w2.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw2_w.slice(f32, b.ff_w2.len), b.ff_w2);
            const fw2_b = try d.device.buffer(b.ff_w2_b.len * 4, mtl.ResourceOptions.shared);
            @memcpy(fw2_b.slice(f32, b.ff_w2_b.len), b.ff_w2_b);
            e.setPipeline(d.p[0]);
            e.setBuffer(gated, 0, 0);
            e.setBuffer(fw2_w, 0, 1);
            e.setBuffer(fw2_b, 0, 2);
            e.setBuffer(qkv, 0, 3);
            e.setValue([4]u32{ @intCast(rows), @intCast(d.config.ffn()), @intCast(dim), 0 }, 4);
            e.dispatchThreads(mtl.Size.of(dim, rows, 1), mtl.Size.of(64, 1, 1));
            e.end();
            cb.commit();
            cb.wait();

            const qkv_host2 = qkv.slice(f32, rows * dim);
            for (0..rows) |r| {
                for (0..dim) |i| tokens_full[r * dim + i] += qkv_host2[r * dim + i] * b.scale2[i];
            }
            n2_w.deinit();
            fw1_w.deinit();
            fw1_b.deinit();
            fw2_w.deinit();
            fw2_b.deinit();
        }
    }
    fused.deinit();
    gated.deinit();
    std.debug.print("stage: 36 blocks done", .{});
    const head_block = d.config.out_channels * d.config.temporal_ratio * d.config.spatial_ratio * d.config.spatial_ratio;

    const normed2 = try d.device.buffer(rows * dim * 4, mtl.ResourceOptions.shared);
    @memcpy(normed.slice(f32, rows * dim), tokens_full);
    // Stage 5: layernorm + proj_out on Metal
    cb = d.queue.commandBuffer();
    e = cb.compute(.concurrent);
    e.end();
    cb.commit();
    cb.wait();
    const no_w = try d.device.buffer(d.norm_out_w.len * 4, mtl.ResourceOptions.shared);
    @memcpy(no_w.slice(f32, d.norm_out_w.len), d.norm_out_w);
    const no_b = try d.device.buffer(d.norm_out_b.len * 4, mtl.ResourceOptions.shared);
    @memcpy(no_b.slice(f32, d.norm_out_b.len), d.norm_out_b);
    // CPU layernorm fallback (bisection): tokens_full -> normed2 host slice
    {
        const nh = normed2.slice(f32, rows * dim);
        const nw = d.norm_out_w;
        const nb = d.norm_out_b;
        for (0..rows) |r| {
            var mean: f32 = 0;
            for (0..dim) |i| mean += tokens_full[r * dim + i];
            mean /= @floatFromInt(dim);
            var m2: f32 = 0;
            for (0..dim) |i| {
                const t = tokens_full[r * dim + i] - mean;
                m2 += t * t;
            }
            const rstd = 1.0 / @sqrt(m2 / @as(f32, @floatFromInt(dim)) + 1e-5);
            for (0..dim) |i| nh[r * dim + i] = (tokens_full[r * dim + i] - mean) * rstd * nw[i] + nb[i];
        }
    }
    no_w.deinit();
    no_b.deinit();
    std.debug.print("stage: norm_out done", .{});

    // Stage 7: proj_out on device, then the GPU unpatch kernel to RGB bytes.
    const proj_dev = try d.device.buffer(voxels * head_block * 4, mtl.ResourceOptions.shared);
    const w_buf = try d.device.buffer(d.proj_out_w.len * 4, mtl.ResourceOptions.shared);
    @memcpy(w_buf.slice(f32, d.proj_out_w.len), d.proj_out_w);
    const b_buf = try d.device.buffer(d.proj_out_b.len * 4, mtl.ResourceOptions.shared);
    @memcpy(b_buf.slice(f32, d.proj_out_b.len), d.proj_out_b);

    const q7 = try d.device.queue();
    {
        const cb7 = q7.commandBuffer();
        const e7 = cb7.compute(.concurrent);
        e7.setPipeline(d.p[0]);
        e7.setBuffer(normed2, 0, 0);
        e7.setBuffer(w_buf, 0, 1);
        e7.setBuffer(b_buf, 0, 2);
        e7.setBuffer(proj_dev, 0, 3);
        e7.setValue([4]u32{ @intCast(voxels), @intCast(dim), @intCast(head_block), 0 }, 4);
        e7.dispatchThreads(mtl.Size.of(head_block, voxels, 1), mtl.Size.of(64, 1, 1));
        e7.end();
        cb7.commit();
        cb7.wait();
    }

    const mean_dev = try d.device.buffer(24 * 4, mtl.ResourceOptions.shared);
    const std_dev = try d.device.buffer(24 * 4, mtl.ResourceOptions.shared);
    const mean_h = mean_dev.slice(f32, 24);
    const std_h = std_dev.slice(f32, 24);
    // per-channel ImageNet normalisation of the PIXELS (the latents were ...
    const pixel_mean = [_]f32{ 0.485, 0.456, 0.406 };
    const pixel_std = [_]f32{ 0.229, 0.224, 0.225 };
    for (0..24) |i| {
        mean_h[i] = pixel_mean[i % 3];
        std_h[i] = pixel_std[i % 3];
    }
    const out_frames = depth * 4;
    const out_h = height * 16;
    const out_w = width * 16;
    const rgb_dev = try d.device.buffer(out_frames * out_h * out_w * 3, mtl.ResourceOptions.shared);
    const cb8 = q7.commandBuffer();
    const e8 = cb8.compute(.concurrent);
    e8.setPipeline(d.p[5]);
    e8.setBuffer(proj_dev, 0, 0);
    e8.setBuffer(mean_dev, 0, 1);
    e8.setBuffer(std_dev, 0, 2);
    e8.setBuffer(rgb_dev, 0, 3);
    e8.setValue([5]u32{ @intCast(depth), @intCast(height), @intCast(width), @intCast(out_w * 3), @intCast(head_block) }, 4);
    e8.dispatchThreads(mtl.Size.of(out_w * 3 * out_h * out_frames, 1, 1), mtl.Size.of(256, 1, 1));
    e8.end();
    cb8.commit();
    cb8.wait();
    mean_dev.deinit();
    std_dev.deinit();
    const rgb_bytes = rgb_dev.slice(u8, out_frames * out_h * out_w * 3);
    @memcpy(out_rgb_bytes, rgb_bytes);
    rgb_dev.deinit();
    proj_dev.deinit();
    w_buf.deinit();
    b_buf.deinit();
    std.debug.print("gpu unpatch done", .{});
}
