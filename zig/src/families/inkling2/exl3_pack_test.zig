//! Host validation tests for Inkling 2 Small EXL3 packs: layouts, widths, codebooks, refusals, and negative controls.
const std = @import("std");
const core = @import("core");
const st = core.safetensors;
const exl3 = core.exl3_format;
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const exl3_cfg = @import("exl3_config.zig");
const names = @import("names.zig");
const plan = @import("plan.zig");
const pack = @import("exl3_pack.zig");

const environment = @import("sys").environment;
const t = std.testing;

const tiny_exl3_config =
    \\{
    \\  "architectures": ["Inkling2ForConditionalGeneration"],
    \\  "model_type": "inkling2",
    \\  "eos_token_id": 2,
    \\  "quantization_config": {"quant_method": "exl3", "format": "kortexa-inkling2-exl3-v1"},
    \\  "text_config": {
    \\    "model_max_length": 4096,
    \\    "hidden_size": 32,
    \\    "num_hidden_layers": 6,
    \\    "vocab_size": 64,
    \\    "unpadded_vocab_size": 60,
    \\    "num_attention_heads": 2,
    \\    "num_key_value_heads": 1,
    \\    "head_dim": 128,
    \\    "swa_num_attention_heads": 2,
    \\    "swa_num_key_value_heads": 1,
    \\    "swa_head_dim": 128,
    \\    "sliding_window_size": 16,
    \\    "d_rel": 16,
    \\    "rel_extent": 16,
    \\    "log_scaling_n_floor": 16,
    \\    "log_scaling_alpha": 0.1,
    \\    "rms_norm_eps": 1e-6,
    \\    "qk_l2_clip_threshold": 22.627416997969522,
    \\    "rel_l2_clip_threshold": 8.0,
    \\    "local_layer_ids": [0, 1, 2, 3, 4],
    \\    "sconv_kernel_size": 4,
    \\    "n_layers_per_attn_res_block": 6,
    \\    "n_routed_experts": 4,
    \\    "num_experts_per_tok": 2,
    \\    "n_shared_experts": 0,
    \\    "intermediate_size": 32,
    \\    "mlp_gate_softcap": 4.0,
    \\    "mlp_up_softcap": 25.0,
    \\    "route_scale": 8.0,
    \\    "final_logit_softcapping": 30.0,
    \\    "use_gate_bias": true
    \\  },
    \\  "vision_config": {
    \\    "hidden_size": 32,
    \\    "num_hidden_layers": 1,
    \\    "num_attention_heads": 2,
    \\    "native_patch_size": 2,
    \\    "adapter_merge_size": 2,
    \\    "posemb_grid_size": 2,
    \\    "mlp_ratio": 2.0,
    \\    "projection_hidden_dim": 32,
    \\    "projector_bias": false
    \\  },
    \\  "mtp_config": {
    \\    "hidden_size": 32,
    \\    "num_hidden_layers": 1,
    \\    "num_attention_heads": 2,
    \\    "num_key_value_heads": 1,
    \\    "head_dim": 128,
    \\    "intermediate_size": 64,
    \\    "rope_theta": 1000000.0,
    \\    "sliding_window_size": 16,
    \\    "local_layer_ids": [0],
    \\    "block_size": 8,
    \\    "markov_rank": 16,
    \\    "markov_head_type": "rnn_hidden",
    \\    "target_layer_ids": [5, 6]
    \\  }
    \\}
;

const TensorSpec = struct {
    name: []const u8,
    dtype: st.DType,
    dims: []const u64,
    bytes: ?[]const u8 = null,
};

fn dtypeName(d: st.DType) []const u8 {
    return switch (d) {
        .bf16 => "BF16",
        .f16 => "F16",
        .f32 => "F32",
        .i16 => "I16",
        .i32 => "I32",
        .u8 => "U8",
        else => unreachable,
    };
}

fn dtypeBytes(d: st.DType) usize {
    return switch (d) {
        .u8 => 1,
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        else => unreachable,
    };
}

fn writeSafetensors(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, specs: []const TensorSpec) !u64 {
    if (std.fs.path.dirname(sub_path)) |parent| try dir.createDirPath(io, parent);
    var hdr: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try hdr.append(a, '{');
    for (specs, 0..) |sp, idx| {
        if (idx > 0) try hdr.append(a, ',');
        var elems: usize = 1;
        for (sp.dims) |d| elems *= @intCast(d);
        const nbytes = elems * dtypeBytes(sp.dtype);
        const begin = data.items.len;
        if (sp.bytes) |raw| {
            try t.expectEqual(nbytes, raw.len);
            try data.appendSlice(a, raw);
        } else {
            try data.appendNTimes(a, 0, nbytes);
        }
        const end = data.items.len;
        try hdr.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ sp.name, dtypeName(sp.dtype) });
        for (sp.dims, 0..) |d, di| {
            if (di > 0) try hdr.append(a, ',');
            try hdr.print(a, "{d}", .{d});
        }
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ begin, end });
    }
    try hdr.append(a, '}');
    var out: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, hdr.items.len, .little);
    try out.appendSlice(a, &len_buf);
    try out.appendSlice(a, hdr.items);
    try out.appendSlice(a, data.items);
    try dir.writeFile(io, .{ .sub_path = sub_path, .data = out.items });
    return out.items.len;
}

const Sink = struct {
    a: std.mem.Allocator,
    specs: std.ArrayList(TensorSpec) = .empty,

    pub fn put(s: *@This(), name: []const u8, dtype: st.DType, dims: []const u64) !void {
        try s.specs.append(s.a, .{
            .name = try s.a.dupe(u8, name),
            .dtype = dtype,
            .dims = try s.a.dupe(u64, dims),
        });
    }
};

const Fault = enum {
    none,
    modelopt_config,
    incomplete_manifest,
    wrong_format,
    missing_expert,
    duplicate_expert,
    unsafe_path,
    shard_size_mismatch,
    wrong_marker_value,
    mcg_codebook,
    bad_bits_five,
    half_bits,
    k_mismatch,
    extra_shard_tensor,
    extra_dense_expert,
    missing_dense_tensor,
    ambiguous_layout,
};

fn buildTinyPack(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, layout: pack.Layout, fault: Fault) !void {
    const cfg_text: []const u8 = if (fault == .modelopt_config)
        try std.mem.replaceOwned(
            u8,
            a,
            tiny_exl3_config,
            "{\"quant_method\": \"exl3\", \"format\": \"kortexa-inkling2-exl3-v1\"}",
            "{\"quant_method\": \"modelopt\", \"quant_algo\": \"NVFP4\"}",
        )
    else
        tiny_exl3_config;
    var why: config.Why = .{};
    const cfg = try config.parse(a, cfg_text, &why);
    if (layout == .single_dense) {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_text });
    } else {
        try dir.createDirPath(io, "auxiliary");
        try dir.writeFile(io, .{ .sub_path = "auxiliary/config.json", .data = cfg_text });
    }

    var dense_sink: Sink = .{ .a = a };
    try names.all(&cfg, &.{}, .dense_text, &dense_sink);
    if (fault == .missing_dense_tensor) _ = dense_sink.specs.pop();
    if (fault == .extra_dense_expert) {
        try dense_sink.specs.append(a, .{
            .name = "model.llm.layers.0.mlp.experts.w13_weight",
            .dtype = .u8,
            .dims = &.{1},
        });
    }
    if (layout == .single_dense) {
        _ = try writeSafetensors(a, io, dir, "model-dense.safetensors", dense_sink.specs.items);
    } else {
        var global: std.ArrayList(TensorSpec) = .empty;
        var per_layer: [6]std.ArrayList(TensorSpec) = @splat(.empty);
        for (dense_sink.specs.items) |sp| {
            const pref = "model.llm.layers.";
            if (std.mem.startsWith(u8, sp.name, pref)) {
                const rest = sp.name[pref.len..];
                const dot = std.mem.indexOfScalar(u8, rest, '.').?;
                const li = try std.fmt.parseInt(usize, rest[0..dot], 10);
                try per_layer[li].append(a, sp);
            } else {
                try global.append(a, sp);
            }
        }
        _ = try writeSafetensors(a, io, dir, "dense/global.safetensors", global.items);
        for (0..6) |li| {
            const sub = try std.fmt.allocPrint(a, "dense/layer-{d:0>2}.safetensors", .{li});
            _ = try writeSafetensors(a, io, dir, sub, per_layer[li].items);
        }
    }

    var mtp_sink: Sink = .{ .a = a };
    try names.all(&cfg, &.{}, .mtp, &mtp_sink);
    const mtp_sub = if (layout == .single_dense) "mtp.safetensors" else "auxiliary/mtp.safetensors";
    _ = try writeSafetensors(a, io, dir, mtp_sub, mtp_sink.specs.items);

    const bits_by_expert = [4]u8{ 2, 3, 3, 4 };
    const expert_prefix = if (layout == .single_dense) "experts/" else "";
    var manifest: std.ArrayList(u8) = .empty;
    const fmt_str = if (fault == .wrong_format) "other-exl3-v1" else pack.format_id;
    const status_str = if (fault == .incomplete_manifest) "in_progress" else "complete";
    try manifest.print(a, "{{\"format\":\"{s}\",\"status\":\"{s}\",\"layers\":{{", .{ fmt_str, status_str });

    var mul1_ok: [4]u8 = undefined;
    std.mem.writeInt(u32, &mul1_ok, exl3.Mul1Mul, .little);
    var mul1_bad: [4]u8 = undefined;
    std.mem.writeInt(u32, &mul1_bad, 0x12345678, .little);

    for (0..6) |li| {
        if (li > 0) try manifest.append(a, ',');
        var b0_specs: std.ArrayList(TensorSpec) = .empty;
        var b1_specs: std.ArrayList(TensorSpec) = .empty;
        for (0..4) |e| {
            const dst = if (e < 2) &b0_specs else &b1_specs;
            var w_dim: u64 = @as(u64, bits_by_expert[e]) * 16;
            if (li == 0 and e == 0 and fault == .bad_bits_five) w_dim = 5 * 16;
            if (li == 0 and e == 0 and fault == .half_bits) w_dim = 40;
            const proj_names = [2][]const u8{ "w13", "w2" };
            const proj_k = [2]u64{ 32, 32 };
            const proj_n = [2]u64{ 64, 32 };
            for (proj_names, proj_k, proj_n) |pn, pk, pn_dim| {
                const t_dims = try a.dupe(u64, &.{ pk / 16, pn_dim / 16, w_dim });
                try dst.append(a, .{
                    .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.trellis", .{ e, pn }),
                    .dtype = .i16,
                    .dims = t_dims,
                });
                try dst.append(a, .{
                    .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.suh", .{ e, pn }),
                    .dtype = .f16,
                    .dims = try a.dupe(u64, &.{pk}),
                });
                try dst.append(a, .{
                    .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.svh", .{ e, pn }),
                    .dtype = .f16,
                    .dims = try a.dupe(u64, &.{pn_dim}),
                });
                const m_kind = if (li == 0 and e == 0 and fault == .mcg_codebook) "mcg" else "mul1";
                const m_bytes: []const u8 = if (li == 0 and e == 0 and fault == .wrong_marker_value) &mul1_bad else &mul1_ok;
                try dst.append(a, .{
                    .name = try std.fmt.allocPrint(a, "experts.{d}.{s}.{s}", .{ e, pn, m_kind }),
                    .dtype = .i32,
                    .dims = &.{},
                    .bytes = try a.dupe(u8, m_bytes),
                });
            }
        }
        if (li == 0 and fault == .extra_shard_tensor) {
            try b0_specs.append(a, .{ .name = "experts.0.w13.extra", .dtype = .u8, .dims = &.{1} });
        }
        const rel0 = try std.fmt.allocPrint(a, "layers/{d:0>2}/experts-000-001.safetensors", .{li});
        const rel1 = try std.fmt.allocPrint(a, "layers/{d:0>2}/experts-002-003.safetensors", .{li});
        const sz0 = try writeSafetensors(a, io, dir, try std.fmt.allocPrint(a, "{s}{s}", .{ expert_prefix, rel0 }), b0_specs.items);
        const sz1 = try writeSafetensors(a, io, dir, try std.fmt.allocPrint(a, "{s}{s}", .{ expert_prefix, rel1 }), b1_specs.items);

        const b1_start: u32 = if (li == 0 and fault == .missing_expert) 3 else if (li == 0 and fault == .duplicate_expert) 1 else 2;
        const rel0_out = if (li == 0 and fault == .unsafe_path) "../escape.safetensors" else rel0;
        const sz0_out = if (li == 0 and fault == .shard_size_mismatch) sz0 + 1 else sz0;
        const k0: u8 = if (li == 0 and fault == .k_mismatch) 3 else bits_by_expert[0];
        try manifest.print(a, "\"{d}\":{{\"layer\":{d},\"status\":\"complete\",\"K_by_expert\":[{d},{d},{d},{d}],\"batches\":[", .{
            li, li, k0, bits_by_expert[1], bits_by_expert[2], bits_by_expert[3],
        });
        try manifest.print(a, "{{\"start\":0,\"end\":2,\"file\":\"{s}\",\"bytes\":{d}}},", .{ rel0_out, sz0_out });
        try manifest.print(a, "{{\"start\":{d},\"end\":4,\"file\":\"{s}\",\"bytes\":{d}}}]}}", .{ b1_start, rel1, sz1 });
    }
    try manifest.appendSlice(a, "}}");

    const m_sub = if (layout == .single_dense) "experts/checkpoint.json" else "checkpoint.json";
    try dir.writeFile(io, .{ .sub_path = m_sub, .data = manifest.items });
    if (fault == .ambiguous_layout) {
        try dir.writeFile(io, .{ .sub_path = "checkpoint.json", .data = manifest.items });
    }
}

test "admitSmallConfig admits Inkling 2 Small EXL3 and refuses ModelOpt, Mini or altered geometry" {
    var why: config.Why = .{};
    const modelopt_small = try config.parse(t.allocator, config_test.real, &why);
    try t.expectError(error.UnsupportedQuant, pack.admitSmallConfig(&modelopt_small, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), pack.format_id) != null);

    var small = modelopt_small;
    small.quant = .exl3_kortexa_v1;
    try pack.admitSmallConfig(&small, &why);

    var small_floor8k = small;
    small_floor8k.text.log_floor = 8192;
    try pack.admitSmallConfig(&small_floor8k, &why);

    var mini = try config.parse(t.allocator, config_test.mini, &why);
    mini.quant = .exl3_kortexa_v1;
    try t.expectError(error.NotInkling2Small, pack.admitSmallConfig(&mini, &why));
    try t.expect(std.mem.indexOf(u8, why.text(), "non-looped") != null);

    var bad = small;
    bad.text.experts = 256;
    try t.expectError(error.UnsupportedInkling2Geometry, pack.admitSmallConfig(&bad, &why));
    var bad_floor = small;
    bad_floor.text.log_floor = 4096;
    try t.expectError(error.UnsupportedInkling2Geometry, pack.admitSmallConfig(&bad_floor, &why));
    var bad_const = small;
    bad_const.text.logit_cap = 20;
    try t.expectError(error.UnsupportedInkling2Constants, pack.admitSmallConfig(&bad_const, &why));
}

test "refusal rejects --vision, --loops, --precision and CFG on text-only EXL3 packs" {
    try t.expect(pack.refusal(.{}) == null);
    try t.expect(std.mem.indexOf(u8, pack.refusal(.{ .vision = true }).?, "--vision") != null);
    try t.expect(std.mem.indexOf(u8, pack.refusal(.{ .loops = 3 }).?, "--loops") != null);
    try t.expect(std.mem.indexOf(u8, pack.refusal(.{ .precision = "fp8" }).?, "--precision") != null);
    try t.expect(pack.refusal(.{ .cfg = true }) != null);
}

test "inspectDir validates both single-dense and split-dense synthetic EXL3 packs with mixed 2..4-bit widths" {
    for ([_]pack.Layout{ .single_dense, .split_dense }) |layout| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        try buildTinyPack(arena.allocator(), t.io, tmp.dir, layout, .none);
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
        defer t.allocator.free(dir);
        var why: config.Why = .{};
        const summary = try pack.inspectDir(t.allocator, t.io, dir, &why);
        try t.expectEqual(layout, summary.layout);
        try t.expectEqual(@as(u32, 6), summary.layers);
        try t.expectEqual(@as(u32, 4), summary.experts);
        try t.expectEqual(@as(usize, 12), summary.expert_shards);
        try t.expectEqual(@as(u64, 12), summary.width_counts[2]);
        try t.expectEqual(@as(u64, 24), summary.width_counts[3]);
        try t.expectEqual(@as(u64, 12), summary.width_counts[4]);
        try t.expect(summary.has_mtp);
        try t.expect(summary.dense_bytes > 0 and summary.expert_bytes > 0 and summary.mtp_bytes > 0);
    }
}

test "inspectDir refuses incomplete, unsafe, wrong-codebook, wrong-width and corrupted packs" {
    const Case = struct { fault: Fault, want: anyerror };
    const cases = [_]Case{
        .{ .fault = .modelopt_config, .want = error.UnsupportedQuant },
        .{ .fault = .incomplete_manifest, .want = error.IncompletePack },
        .{ .fault = .wrong_format, .want = error.UnsupportedFormat },
        .{ .fault = .missing_expert, .want = error.MissingExpert },
        .{ .fault = .duplicate_expert, .want = error.DuplicateExpert },
        .{ .fault = .unsafe_path, .want = error.UnsafePath },
        .{ .fault = .shard_size_mismatch, .want = error.ShardSizeMismatch },
        .{ .fault = .wrong_marker_value, .want = error.UnsupportedCodebook },
        .{ .fault = .mcg_codebook, .want = error.UnsupportedCodebook },
        .{ .fault = .bad_bits_five, .want = error.UnsupportedWidth },
        .{ .fault = .half_bits, .want = error.UnsupportedWidth },
        .{ .fault = .k_mismatch, .want = error.WidthMismatch },
        .{ .fault = .extra_shard_tensor, .want = error.UnexpectedTensor },
        .{ .fault = .extra_dense_expert, .want = error.BadDenseWeights },
        .{ .fault = .missing_dense_tensor, .want = error.BadDenseWeights },
        .{ .fault = .ambiguous_layout, .want = error.AmbiguousLayout },
    };
    for (cases) |c| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        try buildTinyPack(arena.allocator(), t.io, tmp.dir, .single_dense, c.fault);
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
        defer t.allocator.free(dir);
        var why: config.Why = .{};
        try t.expectError(c.want, pack.inspectDir(t.allocator, t.io, dir, &why));
        try t.expect(why.text().len > 0);
    }
}

test "exl3_config source admission, EXL3 layer plan and drift refusal" {
    const a = t.allocator;
    const raw = try std.json.parseFromSlice(std.json.Value, a, config_test.real, .{});
    defer raw.deinit();
    const text_json = try std.json.Stringify.valueAlloc(a, raw.value.object.get("text_config").?, .{});
    defer a.free(text_json);
    const manifest = try std.fmt.allocPrint(a, "{{\"format\":\"kortexa-inkling2-exl3-v1\",\"status\":\"complete\",\"text_config\":{s}}}", .{text_json});
    defer a.free(manifest);
    const c = try exl3_cfg.admitSource(a, manifest, config_test.real);
    const p = try exl3_cfg.layerPlan(&c);
    try t.expectEqual(@as(u32, 48), p.n);
    try t.expectEqual(@as(u32, 8), p.count(.global));
    try t.expectEqual(@as(u32, 40), p.count(.local));
    for (p.layers[0..p.n]) |l| try t.expectEqual(plan.Precision.exl3, l.precision);

    // Changing an arithmetic field only in the manifest must be refused.
    const text_obj = raw.value.object.getPtr("text_config").?;
    try text_obj.object.put(raw.arena.allocator(), "rms_norm_eps", .{ .float = 0.1 });
    const drifted_text = try std.json.Stringify.valueAlloc(a, text_obj.*, .{});
    defer a.free(drifted_text);
    const drifted = try std.fmt.allocPrint(a, "{{\"format\":\"kortexa-inkling2-exl3-v1\",\"status\":\"complete\",\"text_config\":{s}}}", .{drifted_text});
    defer a.free(drifted);
    try t.expectError(error.InklingSourceMismatch, exl3_cfg.admitSource(a, drifted, config_test.real));

    // names.all with .exl3 omits routed expert tensors while keeping dense weights.
    const ExpertCounter = struct {
        total: usize = 0,
        experts: usize = 0,
        pub fn put(self: *@This(), name: []const u8, _: st.DType, _: []const u64) !void {
            self.total += 1;
            if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) self.experts += 1;
        }
    };
    const formats: [48]plan.Precision = @splat(.exl3);
    var ec: ExpertCounter = .{};
    try names.all(&c, &formats, .model, &ec);
    try t.expect(ec.total > 996);
    try t.expectEqual(@as(usize, 0), ec.experts);
}

test "real Inkling 2 Small EXL3 pack headers validate cleanly (TF_INKLING2_EXL3_DIR=<dir>)" {
    const dir = (try environment.path("TF_INKLING2_EXL3_DIR", environment.Process.get(.{}, "TF_INKLING2_EXL3_DIR"))) orelse return error.SkipZigTest;
    var why: config.Why = .{};
    const summary = try pack.inspectDir(t.allocator, t.io, dir, &why);
    try t.expectEqual(@as(u32, 48), summary.layers);
    try t.expectEqual(@as(u32, 384), summary.experts);
    try t.expectEqual(@as(usize, 996), summary.dense_tensors);
}
