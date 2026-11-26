//! Host tests for Inkling 2 Mini affine4 pinned pack metadata, SHA-256 verification, and pre-allocation admission.
const std = @import("std");
const st = @import("core").safetensors;
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const pack = @import("affine4_pack.zig");
const t = std.testing;

fn miniAffineConfig() !cfg.Config {
    const mini_json = @embedFile("../fixtures/mini_config.json");
    var why: cfg.Why = .{};
    var c = try cfg.parse(t.allocator, mini_json, &why);
    c.quant = .mlx_affine4;
    return c;
}

fn miniPlan(c: *const cfg.Config) !layer_plan.Plan {
    var precisions: [24]layer_plan.Precision = undefined;
    for (&precisions, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
    return layer_plan.build(&c.text, &precisions);
}

fn putTensor(idx: *pack.HeaderIndex, name: []const u8, dtype: st.DType, dims: []const usize) !void {
    var count: usize = 1;
    for (dims) |d| count *= d;
    var ht: pack.HeaderTensor = .{ .dtype = dtype, .rank = dims.len, .bytes = count * dtype.size() };
    for (dims, 0..) |d, i| ht.shape[i] = d;
    try idx.put(name, ht);
}

fn putFmt(idx: *pack.HeaderIndex, comptime fmt: []const u8, args: anytype, dtype: st.DType, dims: []const usize) !void {
    var buf: [280]u8 = undefined;
    try putTensor(idx, try std.fmt.bufPrint(&buf, fmt, args), dtype, dims);
}

fn putAffine4(idx: *pack.HeaderIndex, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !void {
    var mbuf: [256]u8 = undefined;
    const mod = try std.fmt.bufPrint(&mbuf, fmt, args);
    try putFmt(idx, "{s}.weight", .{mod}, .u32, &.{ rows, cols / 8 });
    try putFmt(idx, "{s}.scales", .{mod}, .bf16, &.{ rows, cols / 64 });
    try putFmt(idx, "{s}.biases", .{mod}, .bf16, &.{ rows, cols / 64 });
}

fn populateDeclaredPack(idx: *pack.HeaderIndex, c: *const cfg.Config, wp: *const layer_plan.Plan, include_vision: bool) !void {
    try putAffine4(idx, "model.language_model.embed_tokens", .{}, 201024, 2304);
    try putAffine4(idx, "lm_head", .{}, 201024, 2304);
    try putTensor(idx, "model.language_model.embed_norm.weight", .bf16, &.{2304});
    try putTensor(idx, "model.language_model.norm.weight", .bf16, &.{2304});
    try putTensor(idx, "model.language_model.final_input.query", .f32, &.{2304});
    try putTensor(idx, "model.language_model.message_input.query", .f32, &.{ 12, 4, 2304 });
    for (1..24) |layer| {
        const dims: []const usize = if (cfg.isGlobal(@intCast(layer))) &.{ 3, 2304 } else &.{2304};
        try putFmt(idx, "model.language_model.layers.{d}.attention_input.query", .{layer}, .f32, dims);
    }
    for (0..24) |layer| try putFmt(idx, "model.language_model.layers.{d}.mlp_input.query", .{layer}, .f32, &.{2304});

    for (0..24) |i| {
        const layer: u32 = @intCast(i);
        const p = wp.at(layer);
        const global = p.kind == .global;
        const q_width: usize = p.heads.q * p.heads.dim;
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.q_proj", .{layer}, q_width, 2304);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.k_proj", .{layer}, 1024, 2304);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.v_proj", .{layer}, 1024, 2304);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.r_proj", .{layer}, 384, 2304);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.o_proj", .{layer}, 2304, q_width);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.gate_proj", .{layer}, q_width, 2304);
        try putAffine4(idx, "model.language_model.layers.{d}.self_attn.sink_proj", .{layer}, 24, 2304);
        try putFmt(idx, "model.language_model.layers.{d}.self_attn.q_l2_clip.scale", .{layer}, .bf16, &.{ 24, 128 });
        try putFmt(idx, "model.language_model.layers.{d}.self_attn.rel_l2_clip.scale", .{layer}, .bf16, &.{ 24, 16 });
        try putFmt(idx, "model.language_model.layers.{d}.self_attn.rel_logits_proj.proj", .{layer}, .bf16, &.{ 16, p.extent });
        if (global) {
            inline for (.{ "k", "v" }) |role| {
                try putFmt(idx, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.static_weight", .{layer}, .bf16, &.{ 4, 1024 });
                try putAffine4(idx, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.weight_proj", .{layer}, 32, 2304);
            }
            try putFmt(idx, "model.language_model.layers.{d}.q_input_layernorm.weight", .{layer}, .bf16, &.{2304});
            try putFmt(idx, "model.language_model.layers.{d}.k_input_layernorm.weight", .{layer}, .bf16, &.{2304});
            try putFmt(idx, "model.language_model.layers.{d}.v_input_layernorm.weight", .{layer}, .bf16, &.{2304});
        } else {
            inline for (.{ "k", "v" }) |role| {
                try putFmt(idx, "model.language_model.layers.{d}.self_attn." ++ role ++ "_sconv.conv1d.weight", .{layer}, .bf16, &.{ 1024, 4, 1 });
            }
            try putFmt(idx, "model.language_model.layers.{d}.input_layernorm.weight", .{layer}, .bf16, &.{2304});
        }
        try putFmt(idx, "model.language_model.layers.{d}.post_attention_convolution.conv1d.weight", .{layer}, .bf16, &.{ 2304, 4, 1 });
        try putFmt(idx, "model.language_model.layers.{d}.post_attention_layernorm.weight", .{layer}, .bf16, &.{2304});
        try putFmt(idx, "model.language_model.layers.{d}.post_mlp_convolution.conv1d.weight", .{layer}, .bf16, &.{ 2304, 4, 1 });
        try putFmt(idx, "model.language_model.layers.{d}.mlp.gate.weight", .{layer}, .f32, &.{ 128, 2304 });
        try putFmt(idx, "model.language_model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}, .f32, &.{128});
        if (p.precision == .nvfp4) {
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.gate_up_proj.weight", .{layer}, .u32, &.{ 128, 2304, 288 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.gate_up_proj.scales", .{layer}, .u8, &.{ 128, 2304, 144 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.gate_up_proj.global_scale", .{layer}, .f32, &.{128});
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.down_proj.weight", .{layer}, .u32, &.{ 128, 2304, 144 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.down_proj.scales", .{layer}, .u8, &.{ 128, 2304, 72 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.down_proj.global_scale", .{layer}, .f32, &.{128});
        } else {
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.gate_up_proj.weight", .{layer}, .u32, &.{ 128, 2304, 576 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.gate_up_proj.scales", .{layer}, .u8, &.{ 128, 2304, 72 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.down_proj.weight", .{layer}, .u32, &.{ 128, 2304, 288 });
            try putFmt(idx, "model.language_model.layers.{d}.mlp.experts.down_proj.scales", .{layer}, .u8, &.{ 128, 2304, 36 });
        }
        if (layer >= 6 and layer < 18) {
            inline for (.{ "q", "k", "v", "mlp" }) |role| {
                try putFmt(idx, "model.language_model.layers.{d}." ++ role ++ "_message_gate.down_proj.weight", .{layer}, .bf16, &.{ 64, 4608 });
                try putFmt(idx, "model.language_model.layers.{d}." ++ role ++ "_message_gate.forget_gate_proj.weight", .{layer}, .f32, &.{ 2304, 64 });
                try putFmt(idx, "model.language_model.layers.{d}." ++ role ++ "_message_gate.message_gate_proj.weight", .{layer}, .f32, &.{ 2304, 64 });
            }
        }
    }

    if (include_vision) {
        const v = c.vision;
        const h: usize = v.hidden;
        const ff: usize = h * v.mlp_ratio;
        try putTensor(idx, "model.vision_tower.embeddings.patch_embedding.weight", .bf16, &.{ h, 3 * v.patch * v.patch });
        try putTensor(idx, "model.vision_tower.embeddings.position_embedding.weight", .bf16, &.{ @as(usize, v.grid) * v.grid, h });
        try putTensor(idx, "model.vision_tower.embeddings.class_embedding", .bf16, &.{h});
        try putTensor(idx, "model.vision_tower.embeddings.cls_position_embedding", .bf16, &.{h});
        try putTensor(idx, "model.vision_tower.pre_layernorm.weight", .bf16, &.{h});
        try putTensor(idx, "model.vision_tower.pre_layernorm.bias", .bf16, &.{h});
        try putTensor(idx, "model.vision_tower.merger.linear_1.weight", .bf16, &.{ v.proj_hidden, h * v.merge * v.merge });
        try putTensor(idx, "model.vision_tower.merger.linear_2.weight", .bf16, &.{ v.text_hidden, v.proj_hidden });
        for (0..v.layers) |i| {
            try putFmt(idx, "model.vision_tower.layers.{d}.layernorm_before.weight", .{i}, .bf16, &.{h});
            try putFmt(idx, "model.vision_tower.layers.{d}.layernorm_before.bias", .{i}, .bf16, &.{h});
            try putFmt(idx, "model.vision_tower.layers.{d}.layernorm_after.weight", .{i}, .bf16, &.{h});
            try putFmt(idx, "model.vision_tower.layers.{d}.layernorm_after.bias", .{i}, .bf16, &.{h});
            try putFmt(idx, "model.vision_tower.layers.{d}.lambda_1", .{i}, .bf16, &.{h});
            try putFmt(idx, "model.vision_tower.layers.{d}.lambda_2", .{i}, .bf16, &.{h});
            inline for (.{ "q_proj", "k_proj", "v_proj", "out_proj" }) |proj| {
                try putFmt(idx, "model.vision_tower.layers.{d}.self_attn." ++ proj ++ ".weight", .{i}, .bf16, &.{ h, h });
                try putFmt(idx, "model.vision_tower.layers.{d}.self_attn." ++ proj ++ ".bias", .{i}, .bf16, &.{h});
            }
            try putFmt(idx, "model.vision_tower.layers.{d}.mlp.fc1.weight", .{i}, .bf16, &.{ ff, h });
            try putFmt(idx, "model.vision_tower.layers.{d}.mlp.fc1.bias", .{i}, .bf16, &.{ff});
            try putFmt(idx, "model.vision_tower.layers.{d}.mlp.fc2.weight", .{i}, .bf16, &.{ h, ff });
            try putFmt(idx, "model.vision_tower.layers.{d}.mlp.fc2.bias", .{i}, .bf16, &.{h});
        }
    }
}

test "pinned_pack metadata locks target revision, source revision, converter, and 6 SHA-256 digests" {
    try t.expectEqualStrings("mlx-community/Inkling-2-Mini-4bit", pack.pinned_pack.target_repo);
    try t.expectEqualStrings("9856dc53edead71405f8348930937c3638ff21a2", pack.pinned_pack.target_revision);
    try t.expectEqualStrings("thinkingmachines/Inkling-2-Mini", pack.pinned_pack.source_repo);
    try t.expectEqualStrings("ed40e7da625f594fc9c4990f5f66be24873c22d7", pack.pinned_pack.source_revision);
    try t.expectEqualStrings("mlx-lm PR ml-explore/mlx-lm#2086", pack.pinned_pack.converter);
    try t.expectEqual(@as(usize, 4), pack.pinned_pack.shards.len);
    try t.expectEqual(@as(usize, 6), pack.pinned_pack.files.len);
    try t.expectEqualStrings("2e51f82d978af1ee86e899f6a20ab1e13ea16256d2f7c435cc200680d688d8ec", &pack.pinned_pack.shards[0].sha256);
    try t.expectEqualStrings("760cbed0fa51b69be20427da7cb1953043b664da717cc4413836468c8dbe3685", &pack.pinned_pack.shards[3].sha256);
}

test "pre-allocation header admission accepts declared pack and refuses missing, mixed, or stray tensors" {
    const c = try miniAffineConfig();
    const wp = try miniPlan(&c);

    // Full declared pack including vision tower passes without any GPU device or buffer allocation.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, true);
        try pack.admitHeaders(&idx, &c, &wp);
    }

    // Missing companion on lm_head refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        _ = idx.map.remove("lm_head.biases");
        try t.expectError(error.MissingTensor, pack.admitHeaders(&idx, &c, &wp));
    }

    // Missing global_scale on NVFP4 layer 19 refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        _ = idx.map.remove("model.language_model.layers.19.mlp.experts.down_proj.global_scale");
        try t.expectError(error.MissingTensor, pack.admitHeaders(&idx, &c, &wp));
    }

    // Stray global_scale on MXFP8 layer 20 refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        try putTensor(&idx, "model.language_model.layers.20.mlp.experts.gate_up_proj.global_scale", .f32, &.{128});
        try t.expectError(error.UnexpectedInkling2Tensor, pack.admitHeaders(&idx, &c, &wp));
    }

    // Stray affine companion on BF16/FP32 router refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        try putTensor(&idx, "model.language_model.layers.0.mlp.gate.scales", .bf16, &.{ 128, 36 });
        try t.expectError(error.UnexpectedAffineCompanion, pack.admitHeaders(&idx, &c, &wp));
    }

    // Stray affine companion on vision tower refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, true);
        try putTensor(&idx, "model.vision_tower.layers.0.mlp.fc1.scales", .bf16, &.{ 4096, 16 });
        try t.expectError(error.UnexpectedAffineCompanion, pack.admitHeaders(&idx, &c, &wp));
    }

    // Mixed official model.llm.* layout refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        try putTensor(&idx, "model.llm.embed.weight", .bf16, &.{ 201024, 2304 });
        try t.expectError(error.MixedCheckpointLayout, pack.admitHeaders(&idx, &c, &wp));
    }

    // Wrong bits (8-bit packed) on q_proj refused.
    {
        var idx = pack.HeaderIndex.init(t.allocator);
        defer idx.deinit();
        try populateDeclaredPack(&idx, &c, &wp, false);
        _ = idx.map.remove("model.language_model.layers.0.self_attn.q_proj.weight");
        try putTensor(&idx, "model.language_model.layers.0.self_attn.q_proj.weight", .u32, &.{ 3072, 2304 / 4 });
        try t.expectError(error.UnexpectedAffineTensor, pack.admitHeaders(&idx, &c, &wp));
    }
}

test "admitPackFiles and verifyFileDigest check on-disk files before GPU allocation" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    const header_json = "{\"model.language_model.embed_norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2304],\"data_offsets\":[0,4608]}}";
    var shard_bytes = try t.allocator.alloc(u8, 8 + header_json.len + 4608);
    defer t.allocator.free(shard_bytes);
    @memset(shard_bytes, 0);
    std.mem.writeInt(u64, shard_bytes[0..8], header_json.len, .little);
    @memcpy(shard_bytes[8..][0..header_json.len], header_json);
    try tmp.dir.writeFile(io, .{ .sub_path = "partial.safetensors", .data = shard_bytes });

    const dir_path = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(dir_path);
    const shard_path = try std.fs.path.joinZ(t.allocator, &.{ dir_path, "partial.safetensors" });
    defer t.allocator.free(shard_path);

    const c = try miniAffineConfig();
    const wp = try miniPlan(&c);
    try t.expectError(error.MissingTensor, pack.admitPackFiles(t.allocator, io, &.{shard_path}, &c, &wp));

    const digest = try pack.fileSha256Hex(io, shard_path);
    try pack.verifyFileDigest(io, shard_path, digest);
    var bad = digest;
    bad[0] = if (bad[0] == 'a') 'b' else 'a';
    try t.expectError(error.PinnedDigestMismatch, pack.verifyFileDigest(io, shard_path, bad));

    const missing_path = try std.fs.path.join(t.allocator, &.{ dir_path, "missing.safetensors" });
    defer t.allocator.free(missing_path);
    try t.expectError(error.MissingPinnedFile, pack.verifyFileDigest(io, missing_path, digest));
}
