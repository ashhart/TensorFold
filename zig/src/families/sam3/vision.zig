// SAM 3.1 architecture, adapted from facebookresearch/sam3.
// Copyright (c) Meta Platforms, Inc. and affiliates. See LICENSES/SAM.txt.
const std = @import("std");
const gpu = @import("tensor_metal");
const ops = @import("ops.zig");
const Runtime = ops.Runtime;
const Tensor = ops.Tensor;
const Feature = ops.Feature;
const feature = ops.feature;
const view = ops.view;
const Size = gpu.mtl.Size;

pub fn trunk(r: *Runtime, image: Feature) !Feature {
    const start = r.mark();
    var h = try r.conv("detector.backbone.vision_backbone.trunk.patch_embed.proj", image, 14, 0, false, false);
    if (h.height != 72 or h.width != 72 or h.channels != 1024) return error.SamImageShape;
    const tiled = r.constant("trunk.positions") orelse positions: {
        const pos = try r.weight("detector.backbone.vision_backbone.trunk.pos_embed");
        const tiled = try r.alloc(h.tensor.count);
        for (0..72) |y| for (0..72) |x| @memcpy(tiled.slice(f32, tiled.count)[(y * 72 + x) * 1024 ..][0..1024], pos.slice(f32, pos.count)[(1 + (y % 24) * 24 + x % 24) * 1024 ..][0..1024]);
        break :positions try r.remember("trunk.positions", tiled);
    };
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    h.tensor = try r.norm("detector.backbone.vision_backbone.trunk.ln_pre", try r.add(h.tensor, tiled), 1024, 1e-5);
    r.trim(start, &.{h.tensor});
    for (0..32) |i| {
        const key = try r.name("detector.backbone.vision_backbone.trunk.blocks.{d}", .{i});
        const global = i % 8 == 7;
        const input = try r.visionNorm(try r.name("{s}.norm1", .{key}), h.tensor, !global);
        const qkv = try r.visionLinear(try r.name("{s}.attn.qkv", .{key}), input, 1024, false);
        const q, const k, const v = try r.visionQkv(qkv, global);
        const attended = try r.visionAttention(q, k, v, global);
        const residual = try r.visionResidual(try r.name("{s}.attn.proj", .{key}), attended, 1024, h.tensor, !global);
        const normalized = try r.visionNorm(try r.name("{s}.norm2", .{key}), residual, false);
        const up = try r.visionLinear(try r.name("{s}.mlp.fc1", .{key}), normalized, 1024, true);
        const next = try r.visionResidual(try r.name("{s}.mlp.fc2", .{key}), up, 4736, residual, false);
        r.trim(start, &.{next});
        h.tensor = next;
        if (!r.vision_ready and i % 4 == 3 and i != 31) {
            try r.g.endBatch();
            try r.g.beginBatch();
        }
    }
    try r.g.endBatch();
    r.vision_ready = true;
    return h;
}

pub fn pyramid(r: *Runtime, x: Feature, kind: []const u8) ![3]Feature {
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    var out: [3]Feature = undefined;
    for (&out, 0..) |*dst, i| {
        const mark = r.mark();
        const key = try r.name("detector.backbone.vision_backbone.{s}.{d}", .{ kind, i });
        var h = x;
        if (i == 0) {
            h = try r.conv(try r.name("{s}.dconv_2x2_0", .{key}), h, 2, 0, true, false);
            h.tensor = try r.gelu(h.tensor);
            h = try r.conv(try r.name("{s}.dconv_2x2_1", .{key}), h, 2, 0, true, false);
        }
        if (i == 1) h = try r.conv(try r.name("{s}.dconv_2x2", .{key}), h, 2, 0, true, false);
        h = try r.conv(try r.name("{s}.conv_1x1", .{key}), h, 1, 0, false, false);
        dst.* = try r.conv(try r.name("{s}.conv_3x3", .{key}), h, 1, 1, false, false);
        r.trim(mark, &.{dst.tensor});
    }
    try r.g.endBatch();
    return out;
}

pub fn sine(r: *Runtime, h: usize, w: usize) !Tensor {
    return sineWidth(r, h, w, 256);
}

pub fn sineWidth(r: *Runtime, h: usize, w: usize, width: usize) !Tensor {
    const key = try r.name("sine.{d}.{d}.{d}", .{ h, w, width });
    if (r.constant(key)) |cached| return cached;
    const out = try r.alloc(h * w * width);
    const data = out.slice(f32, out.count);
    for (0..h) |y| for (0..w) |x| for (0..width) |c| {
        const pos = if (c < width / 2) @as(f32, @floatFromInt(y)) / (@as(f32, @floatFromInt(h - 1)) + 1e-6) else @as(f32, @floatFromInt(x)) / (@as(f32, @floatFromInt(w - 1)) + 1e-6);
        const angle = pos * (2 * std.math.pi) / std.math.pow(f32, 10000, @as(f32, @floatFromInt((c % (width / 2)) / 2 * 2)) / @as(f32, @floatFromInt(width / 2)));
        data[(y * w + x) * width + c] = if (c % 2 == 0) @sin(angle) else @cos(angle);
    };
    return r.remember(key, out);
}

pub fn fourier(r: *Runtime, key: []const u8, coords: []const [2]f32) !Tensor {
    const matrix = try r.weight(key);
    if (matrix.count != 256) return error.SamFourierShape;
    const out = try r.alloc(coords.len * 256);
    const data = out.slice(f32, out.count);
    for (coords, 0..) |p, i| for (0..128) |c| {
        const v = 2 * std.math.pi * ((2 * p[0] - 1) * matrix.get(c) + (2 * p[1] - 1) * matrix.get(128 + c));
        data[i * 256 + c] = @sin(v);
        data[i * 256 + 128 + c] = @cos(v);
    };
    return out;
}

pub fn imagePe(r: *Runtime, key: []const u8) !Tensor {
    if (r.constant(key)) |cached| return cached;
    const coords = try r.a.alloc([2]f32, 5184);
    for (coords, 0..) |*p, i| p.* = .{ (@as(f32, @floatFromInt(i % 72)) + 0.5) / 72, (@as(f32, @floatFromInt(i / 72)) + 0.5) / 72 };
    return r.remember(key, try fourier(r, key, coords));
}

pub fn text(r: *Runtime, tokens: []const u32) !Tensor {
    const start = r.mark();
    const root = "detector.backbone.language_backbone.encoder";
    const embedding = try r.weight(root ++ ".token_embedding.weight");
    const positions = try r.weight(root ++ ".positional_embedding");
    if (tokens.len < 2 or tokens.len > positions.count / 1024) return error.InvalidTextLength;
    for (tokens) |id| if (id >= embedding.count / 1024) return error.InvalidTextToken;
    var h = try r.alloc(tokens.len * 1024);
    for (tokens, 0..) |id, i| for (0..1024) |c| {
        h.slice(f32, h.count)[i * 1024 + c] = embedding.get(id * 1024 + c) + positions.get(i * 1024 + c);
    };
    r.trim(start, &.{h});
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    for (0..24) |i| {
        const key = try r.name(root ++ ".transformer.resblocks.{d}", .{i});
        const n = try r.norm(try r.name("{s}.ln_1", .{key}), h, 1024, 1e-5);
        const attn = try r.name("{s}.attn", .{key});
        const q = try r.project(attn, n, 1024, 0);
        const k = try r.project(attn, n, 1024, 1);
        const v = try r.project(attn, n, 1024, 2);
        const attended = try r.attend(q, k, v, 16, 1024, null, true);
        const residual = try r.add(h, try r.linear(try r.name("{s}.attn.out_proj", .{key}), attended, 1024));
        const up = try r.gelu(try r.linear(try r.name("{s}.mlp.c_fc", .{key}), try r.norm(try r.name("{s}.ln_2", .{key}), residual, 1024, 1e-5), 1024));
        const next = try r.add(residual, try r.linear(try r.name("{s}.mlp.c_proj", .{key}), up, 4096));
        r.trim(start, &.{next});
        h = next;
    }
    const encoded = try r.linear("detector.backbone.language_backbone.resizer", try r.norm(root ++ ".ln_final", h, 1024, 1e-5), 1024);
    r.trim(start, &.{encoded});
    try r.g.endBatch();
    return encoded;
}

fn ffn(r: *Runtime, key: []const u8, h: Tensor, gelu: bool) !Tensor {
    const up = try r.linear(try r.name("{s}.linear1", .{key}), h, 256);
    return r.linear(try r.name("{s}.linear2", .{key}), if (gelu) try r.gelu(up) else try r.unary(up, 0), 2048);
}

fn geoClass(r: *Runtime, image: Tensor, pos: Tensor) !Tensor {
    const root = "detector.geometry_encoder";
    var h = try r.norm(root ++ ".norm", try r.linear(root ++ ".final_proj", try r.weight(root ++ ".cls_embed.weight"), 256), 256, 1e-5);
    const image_pos = try r.add(image, pos);
    for (0..3) |i| {
        const key = try r.name(root ++ ".encode.{d}", .{i});
        const n = try r.norm(try r.name("{s}.norm1", .{key}), h, 256, 1e-5);
        h = try r.add(h, try r.mha(try r.name("{s}.self_attn", .{key}), n, n, n, 8, 256, null));
        const cross = try r.mha(try r.name("{s}.cross_attn_image", .{key}), try r.norm(try r.name("{s}.norm2", .{key}), h, 256, 1e-5), image_pos, image, 8, 256, null);
        h = try r.add(h, cross);
        h = try r.add(h, try ffn(r, key, try r.norm(try r.name("{s}.norm3", .{key}), h, 256, 1e-5), false));
    }
    return r.norm(root ++ ".encode_norm", h, 256, 1e-5);
}

fn boxPosition(r: *Runtime, boxes: Tensor) !Tensor {
    const out = try r.alloc(200 * 512);
    const dst = out.slice(f32, out.count);
    for (0..200) |q| for (0..4) |c| for (0..128) |d| {
        const angle = boxes.get(q * 4 + c) * 2 * std.math.pi / std.math.pow(f32, 10000, @as(f32, @floatFromInt(d / 2 * 2)) / 128);
        dst[q * 512 + c * 128 + d] = if (d % 2 == 0) @sin(angle) else @cos(angle);
    };
    return out;
}

fn boxBias(r: *Runtime, boxes: Tensor) !Tensor {
    const delta = try r.alloc(200 * 72 * 2);
    var components: [2]Tensor = undefined;
    for (&components, 0..) |*out, axis| {
        for (0..200) |q| for (0..72) |i| for (0..2) |edge| {
            const corner = boxes.get(q * 4 + axis) + boxes.get(q * 4 + axis + 2) * @as(f32, if (edge == 0) -0.5 else 0.5);
            const d = (@as(f32, @floatFromInt(i)) / 72 - corner) * 8;
            delta.slice(f32, delta.count)[(q * 72 + i) * 2 + edge] = std.math.sign(d) * @log2(@abs(d) + 1) / 3;
        };
        out.* = try r.mlp(if (axis == 0) "detector.transformer.decoder.boxRPB_embed_x" else "detector.transformer.decoder.boxRPB_embed_y", delta, 2, 2);
    }
    const bias = try r.alloc(8 * 201 * 5184);
    const dst = bias.slice(f32, bias.count);
    for (0..8) |head| {
        @memset(dst[head * 201 * 5184 ..][0..5184], 0);
        for (0..200) |q| for (0..72) |y| for (0..72) |x| {
            dst[(head * 201 + q + 1) * 5184 + y * 72 + x] = components[0].get((q * 72 + x) * 8 + head) + components[1].get((q * 72 + y) * 8 + head);
        };
    }
    return bias;
}

pub const Detection = struct { mask: Feature, box: [4]f32, score: f32 };

pub const DetectionOptions = struct {
    threshold: f32 = 0.5,
    max_results: usize = 200,
};

pub fn detect(r: *Runtime, features: [3]Feature, text_features: Tensor, options: DetectionOptions) ![]Detection {
    if (!std.math.isFinite(options.threshold) or options.threshold < 0 or options.threshold > 1 or
        options.max_results == 0 or options.max_results > 200) return error.InvalidDetectionOptions;
    const start = r.mark();
    const image = features[2].tensor;
    const pos = try sine(r, 72, 72);
    const prompt = try r.concat(&.{ text_features, try geoClass(r, image, pos) });
    var memory = image;
    for (0..6) |i| {
        const mark = r.mark();
        const key = try r.name("detector.transformer.encoder.layers.{d}", .{i});
        const norm = try r.norm(try r.name("{s}.norm1", .{key}), memory, 256, 1e-5);
        const qk = try r.add(norm, pos);
        var h = try r.add(memory, try r.mha(try r.name("{s}.self_attn", .{key}), qk, qk, norm, 8, 256, null));
        h = try r.add(h, try r.mha(try r.name("{s}.cross_attn_image", .{key}), try r.norm(try r.name("{s}.norm2", .{key}), h, 256, 1e-5), prompt, prompt, 8, 256, null));
        memory = try r.add(h, try ffn(r, key, try r.norm(try r.name("{s}.norm3", .{key}), h, 256, 1e-5), false));
        r.trim(mark, &.{memory});
    }
    const root = "detector.transformer.decoder";
    var queries = try r.concat(&.{ try r.weight(root ++ ".presence_token.weight"), try r.weight(root ++ ".query_embed.weight") });
    var boxes = try r.unary(try r.weight(root ++ ".reference_points.weight"), 2);
    for (0..6) |i| {
        const mark = r.mark();
        const key = try r.name(root ++ ".layers.{d}", .{i});
        const zero = try r.alloc(256);
        @memset(zero.slice(f32, 256), 0);
        const qp = try r.concat(&.{ zero, try r.mlp(root ++ ".ref_point_head", try boxPosition(r, boxes), 512, 2) });
        const qk = try r.add(queries, qp);
        var h = try r.norm(try r.name("{s}.norm2", .{key}), try r.add(queries, try r.mha(try r.name("{s}.self_attn", .{key}), qk, qk, queries, 8, 256, null)), 256, 1e-5);
        h = try r.norm(try r.name("{s}.catext_norm", .{key}), try r.add(h, try r.mha(try r.name("{s}.ca_text", .{key}), try r.add(h, qp), prompt, prompt, 8, 256, null)), 256, 1e-5);
        h = try r.norm(try r.name("{s}.norm1", .{key}), try r.add(h, try r.mha(try r.name("{s}.cross_attn", .{key}), try r.add(h, qp), try r.add(memory, pos), memory, 8, 256, try boxBias(r, boxes))), 256, 1e-5);
        queries = try r.norm(try r.name("{s}.norm3", .{key}), try r.add(h, try ffn(r, key, h, false)), 256, 1e-5);
        const offset = try r.mlp(root ++ ".bbox_embed", try r.norm(root ++ ".norm", view(queries, 256, 200 * 256), 256, 1e-5), 256, 3);
        const next = try r.alloc(800);
        for (next.slice(f32, 800), 0..) |*v, j| {
            const b = boxes.get(j);
            v.* = 1 / (1 + @exp(-(@log(b / (1 - b + 1e-6) + 1e-6) + offset.get(j))));
        }
        boxes = next;
        r.trim(mark, &.{ queries, boxes });
    }
    const query = try r.norm(root ++ ".norm", view(queries, 256, 200 * 256), 256, 1e-5);
    const scoring = "detector.dot_prod_scoring";
    const prompt_mlp = try r.norm(scoring ++ ".prompt_mlp.out_norm", try r.add(prompt, try r.mlp(scoring ++ ".prompt_mlp", prompt, 256, 2)), 256, 1e-5);
    const pooled = try r.alloc(256);
    @memset(pooled.slice(f32, 256), 0);
    for (0..prompt.count / 256) |row| for (0..256) |c| {
        pooled.slice(f32, 256)[c] += prompt_mlp.get(row * 256 + c) / @as(f32, @floatFromInt(prompt.count / 256));
    };
    const hs = try r.linear(scoring ++ ".hs_proj", query, 256);
    const pp = try r.linear(scoring ++ ".prompt_proj", pooled, 256);
    const presence = try r.mlp(root ++ ".presence_token_head", try r.norm(root ++ ".presence_token_out_norm", view(queries, 0, 256), 256, 1e-5), 256, 3);
    const presence_score = 1 / (1 + @exp(-presence.get(0)));
    const Candidate = struct { index: usize, score: f32 };
    var candidates: [200]Candidate = undefined;
    var count: usize = 0;
    for (0..200) |q| {
        var dot: f32 = 0;
        for (0..256) |c| dot += hs.get(q * 256 + c) * pp.get(c);
        const score = 1 / (1 + @exp(-std.math.clamp(dot / 16, -12, 12))) * presence_score;
        if (score >= options.threshold) {
            candidates[count] = .{ .index = q, .score = score };
            count += 1;
        }
    }
    std.mem.sort(Candidate, candidates[0..count], {}, struct {
        fn less(_: void, a: Candidate, b: Candidate) bool {
            return a.score > b.score or (a.score == b.score and a.index < b.index);
        }
    }.less);
    count = @min(count, options.max_results);
    const results = try r.a.alloc(Detection, count);
    if (count == 0) {
        r.trim(start, &.{});
        return results;
    }
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    const seg = "detector.segmentation_head";
    var visual = try r.add(memory, try r.mha(seg ++ ".cross_attend_prompt", try r.norm(seg ++ ".cross_attn_norm", memory, 256, 1e-5), prompt, prompt, 8, 256, null));
    var f = feature(visual, 72, 72, 256);
    for (0..2) |i| {
        const high = features[1 - i];
        visual = try r.add(high.tensor, (try r.resize(f, high.height, high.width, false)).tensor);
        f = try r.conv(try r.name(seg ++ ".pixel_decoder.conv_layers.{d}", .{i}), feature(visual, high.height, high.width, 256), 1, 1, false, false);
        f = try r.groupNorm(try r.name(seg ++ ".pixel_decoder.norms.{d}", .{i}), f, 8);
        f.tensor = try r.unary(f.tensor, 0);
    }
    f = try r.conv(seg ++ ".instance_seg_head", f, 1, 0, false, false);
    const keep = try r.a.alloc(Tensor, count);
    for (candidates[0..count], results, keep) |candidate, *result, *kept| {
        const mark = r.mark();
        const selected = candidate.index;
        const embed = try r.mlp(seg ++ ".mask_predictor.mask_embed", view(query, selected * 256, 256), 256, 3);
        const mask = try r.matmul(f.tensor, embed, f.height * f.width, 1, 256);
        const cx = boxes.get(selected * 4);
        const cy = boxes.get(selected * 4 + 1);
        const w = boxes.get(selected * 4 + 2);
        const h = boxes.get(selected * 4 + 3);
        result.* = .{ .mask = feature(mask, 288, 288, 1), .box = .{ cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2 }, .score = candidate.score };
        kept.* = mask;
        r.trim(mark, &.{mask});
    }
    try r.g.endBatch();
    r.trim(start, keep);
    return results;
}

pub fn feedForward(r: *Runtime, key: []const u8, h: Tensor) !Tensor {
    return ffn(r, key, h, true);
}
