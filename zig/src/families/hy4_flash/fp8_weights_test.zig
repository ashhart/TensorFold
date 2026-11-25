//! Host tests for Hy4-flash W8A16 weight mapping, TP=1/2/4 partitioning/memory floors, and 100-byte FP8 Engram rows.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");
const engram = @import("engram.zig");
const fp8_weights = @import("fp8_weights.zig");
const part = @import("fp8_partition.zig");
const ep = @import("engram_fp8.zig");

const testing = std.testing;
const io = testing.io;
const Entry = core.checkpoint_fixture.Entry;
const GiB = 1 << 30;

fn release() !std.json.Parsed(config.Model) {
    return config.parse(testing.allocator, @embedFile("config_fixture.json"));
}

const Fixture = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(Entry) = .empty,

    fn add(f: *Fixture, dtype: []const u8, shape: []const usize, comptime fmt: []const u8, args: anytype) !void {
        var n: usize = if (std.mem.eql(u8, dtype, "BF16")) 2 else if (std.mem.eql(u8, dtype, "F8_E4M3")) 1 else 4;
        for (shape) |x| n *= x;
        const bytes = try f.a.alloc(u8, n);
        @memset(bytes, 0);
        try f.list.append(f.a, .{
            .name = try std.fmt.allocPrint(f.a, fmt, args),
            .dtype = dtype,
            .shape = try f.a.dupe(usize, shape),
            .bytes = bytes,
        });
    }

    fn proj(f: *Fixture, q: bool, n: usize, k: usize, comptime fmt: []const u8, args: anytype) !void {
        if (!q) return f.add("BF16", &.{ n, k }, fmt ++ ".weight", args);
        try f.add("F8_E4M3", &.{ n, k }, fmt ++ ".weight", args);
        try f.add("BF16", &.{ core.quant.fp8.blocks(n), core.quant.fp8.blocks(k) }, fmt ++ ".weight_scale_inv", args);
    }
};

fn tinyText(a: std.mem.Allocator, base: config.Text) !config.Text {
    var t = base;
    t.num_hidden_layers = 4;
    t.hidden_size = 256;
    t.vocab_size = 8;
    t.q_lora_rank = 128;
    t.num_attention_heads = 1;
    t.swa_num_attention_heads = 2;
    t.layer_types = &.{ .deepseek_sparse_attention, .sliding_attention, .deepseek_sparse_attention, .sliding_attention };
    t.mlp_layer_types = &.{ .dense, .sparse, .sparse, .sparse };
    t.intermediate_size = 128;
    t.n_routed_experts = 2;
    t.moe_intermediate_size = 128;
    t.routed_expert_hidden_size = 128;
    t.engram_layer_ids = &.{ 1, 2, 3, 4 };
    t.engram_lookup_table_size = 16;
    var vocab: @FieldType(config.Text, "engram_layer_head_vocab_sizes") = .{};
    const row8 = try a.dupe(u32, &.{ 2, 2, 2, 2, 2, 2, 2, 2 });
    const orders = try a.dupe([]const u32, &.{ row8, row8 });
    for ([_][]const u8{ "1", "2", "3", "4" }) |k| try vocab.map.put(a, k, orders);
    t.engram_layer_head_vocab_sizes = vocab;
    return t;
}

const BadMode = enum { none, extra_tensor, stray_scale, bad_dtype };

fn writeTiny(a: std.mem.Allocator, tmp: testing.TmpDir, t: config.Text, bad: BadMode) !void {
    var f: Fixture = .{ .a = a };
    const d: usize = t.hidden_size;
    const m: usize = t.hc_mult;
    try f.add("BF16", &.{ t.vocab_size, d }, "model.embed_tokens.weight", .{});
    try f.add("BF16", &.{d}, "model.norm.weight", .{});
    try f.add("BF16", &.{ t.vocab_size, d }, "lm_head.weight", .{});
    try f.add("F32", &.{ m, m * d }, "model.hc_head.hc_head_fn", .{});
    try f.add("F32", &.{m}, "model.hc_head.hc_head_base", .{});
    try f.add("F32", &.{1}, "model.hc_head.hc_head_scale", .{});
    try f.add("BF16", &.{ 2, 2 }, "model.dfly_layers.0.mlp.up_proj.weight", .{});
    try f.add("BF16", &.{ 2, 2 }, "vit.embeddings.patch_embedding.weight", .{});
    if (bad == .extra_tensor) try f.add("BF16", &.{2}, "model.layers.0.self_attn.q_norm.weight", .{});
    for (0..t.num_hidden_layers) |i| {
        const h = t.headShape(@intCast(i));
        const q = i > 0;
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |n| try f.add("BF16", &.{d}, "model.layers.{d}.{s}.weight", .{ i, n });
        for ([_][]const u8{ "hc_attn_layer", "hc_mlp_layer" }) |s| {
            try f.add("F32", &.{ 2 * m, m * d }, "model.layers.{d}.{s}.hc_pre.hc_fn", .{ i, s });
            try f.add("F32", &.{2 * m}, "model.layers.{d}.{s}.hc_pre.hc_base", .{ i, s });
            try f.add("F32", &.{2}, "model.layers.{d}.{s}.hc_pre.hc_scale", .{ i, s });
        }
        const a_ = "model.layers.{d}.self_attn.";
        try f.add("BF16", &.{ t.q_lora_rank, d }, a_ ++ "q_a_proj.weight", .{i});
        try f.add("BF16", &.{t.q_lora_rank}, a_ ++ "q_a_layernorm.weight", .{i});
        if (i == 0 and bad == .bad_dtype) {
            try f.add("F32", &.{ h.queryRows(), t.q_lora_rank }, a_ ++ "q_b_proj.weight", .{i});
        } else {
            try f.proj(q, h.queryRows(), t.q_lora_rank, a_ ++ "q_b_proj", .{i});
            if (i == 0 and bad == .stray_scale) try f.add("BF16", &.{ 2, 1 }, a_ ++ "q_b_proj.weight_scale_inv", .{i});
        }
        try f.add("BF16", &.{ @as(usize, t.kv_lora_rank) + h.rotary, d }, a_ ++ "kv_a_proj_with_mqa.weight", .{i});
        try f.add("BF16", &.{t.kv_lora_rank}, a_ ++ "kv_a_layernorm.weight", .{i});
        try f.add("BF16", &.{ h.kvRows(), t.kv_lora_rank }, a_ ++ "kv_b_proj.weight", .{i});
        try f.proj(q, @as(usize, h.heads) * h.value, d, a_ ++ "linear_gate", .{i});
        try f.proj(q, d, @as(usize, h.heads) * h.value, a_ ++ "o_proj", .{i});
        try f.add("BF16", &.{d}, a_ ++ "post_proj_norm.weight", .{i});
        try f.add("F32", &.{h.heads}, a_ ++ "learnable_sink_param", .{i});
        if (t.layer_types[i] == .deepseek_sparse_attention) {
            const x = a_ ++ "compressed_indexer.";
            const n: usize = t.compressed_indexer_head_dim;
            try f.add("BF16", &.{ @as(usize, t.compressed_indexer_num_attention_heads) * n, t.q_lora_rank }, x ++ "query_projection.weight", .{i});
            try f.add("BF16", &.{ t.compressed_indexer_num_attention_heads, d }, x ++ "head_weight_projection.weight", .{i});
            try f.add("BF16", &.{ n, d }, x ++ "token_key_projection.weight", .{i});
            try f.add("BF16", &.{n}, x ++ "token_key_norm.weight", .{i});
            try f.add("F32", &.{ n, n }, x ++ "key_value_projection.weight", .{i});
            try f.add("F32", &.{ n, n }, x ++ "key_gate_projection.weight", .{i});
            try f.add("F32", &.{n}, x ++ "key_norm.weight", .{i});
            try f.add("F32", &.{ t.compressed_indexer_compress_ratio, n }, x ++ "intra_group_ape", .{i});
        }
        const p = "model.layers.{d}.mlp.";
        if (t.mlp_layer_types[i] == .dense) {
            try f.add("BF16", &.{ t.intermediate_size, d }, p ++ "gate_proj.weight", .{i});
            try f.add("BF16", &.{ t.intermediate_size, d }, p ++ "up_proj.weight", .{i});
            try f.add("BF16", &.{ d, t.intermediate_size }, p ++ "down_proj.weight", .{i});
            try f.add("BF16", &.{d}, p ++ "post_fc2_norm.weight", .{i});
        } else {
            const e: usize = t.n_routed_experts;
            const w: usize = t.moe_intermediate_size;
            const l: usize = t.routed_expert_hidden_size;
            try f.add("BF16", &.{ e, d }, p ++ "gate.weight", .{i});
            try f.add("F32", &.{e}, p ++ "gate.e_score_correction_bias", .{i});
            try f.add("BF16", &.{ l, d }, p ++ "routed_expert_down_proj.weight", .{i});
            try f.add("BF16", &.{ d, l }, p ++ "routed_expert_up_proj.weight", .{i});
            try f.add("F8_E4M3", &.{ e, 2 * w, l }, p ++ "experts.gate_up_proj", .{i});
            try f.add("BF16", &.{ e, 2 * w / 128, l / 128 }, p ++ "experts.gate_up_proj_scale_inv", .{i});
            try f.add("F8_E4M3", &.{ e, l, w }, p ++ "experts.down_proj", .{i});
            try f.add("BF16", &.{ e, l / 128, w / 128 }, p ++ "experts.down_proj_scale_inv", .{i});
            try f.add("BF16", &.{ w, d }, p ++ "shared_experts.gate_proj.weight", .{i});
            try f.add("BF16", &.{ w, d }, p ++ "shared_experts.up_proj.weight", .{i});
            try f.add("BF16", &.{ d, w }, p ++ "shared_experts.down_proj.weight", .{i});
            try f.add("BF16", &.{d}, p ++ "combine_output_norm.weight", .{i});
        }
        const g = "model.layers.{d}.engram_layer.";
        const head_dim: usize = t.engram_n_embed_per_ngram / t.engram_n_head_per_ngram;
        const rows = try t.engramRows(@intCast(i + 1));
        const k = engram.columns * head_dim;
        try f.add("BF16", &.{ rows, head_dim }, g ++ "multi_head_embedding.embedding.weight", .{i});
        try f.add("I32", &.{t.engram_lookup_table_size}, g ++ "lookup_table", .{i});
        for (0..engram.columns) |head| try f.add("BF16", &.{head_dim}, g ++ "lookup_per_head_norms.{d}.weight", .{ i, head });
        try f.add("BF16", &.{ d, k }, g ++ "value_proj.weight", .{i});
        try f.add("BF16", &.{d}, g ++ "value_proj.bias", .{i});
        for (0..4) |j| {
            try f.add("BF16", &.{ d, k }, g ++ "key_projs.{d}.weight", .{ i, j });
            try f.add("BF16", &.{d}, g ++ "key_projs.{d}.bias", .{ i, j });
            try f.add("BF16", &.{d}, g ++ "norm_key.{d}.weight", .{ i, j });
            try f.add("BF16", &.{d}, g ++ "norm_query.{d}.weight", .{ i, j });
            try f.add("BF16", &.{d}, g ++ "unified_per_hc_scales.{d}", .{ i, j });
        }
        try f.add("BF16", &.{d}, g ++ "unified_pre_gamma", .{i});
    }
    try core.checkpoint_fixture.write(a, io, tmp.dir, "model.safetensors", f.list.items);
}

fn tmpPath(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    return tmp.dir.realPathFileAlloc(io, ".", a);
}

test "W8A16 checkpoint map admits bf16 and e4m3 projections and refuses leftover or invalid tensors" {
    const parsed = try release();
    defer parsed.deinit();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try tinyText(a, parsed.value.text_config);

    {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeTiny(a, tmp, t, .none);
        var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, io, try tmpPath(a, tmp));
        defer ck.close();
        var why: core.modelopt.Why = .{};
        var m = try fp8_weights.map(testing.allocator, &ck, t, &why);
        defer m.deinit();
        try testing.expect(m.layers[0].attn.q_b == .bf16 and m.layers[1].attn.q_b == .fp8 and m.layers[3].attn.o == .fp8);
        try testing.expectEqual(core.checkpoint.DType.bf16, m.layers[1].attn.o.fp8.scale_inv.dtype);
        try testing.expect(m.layers[0].attn.indexer != null and m.layers[1].attn.indexer == null);
        try testing.expect(m.layers[0].mlp == .dense and m.layers[1].mlp == .moe);
        try testing.expectEqual(@as(usize, 32), m.layers[2].engram.?.table.dim(0));
        try testing.expectEqual(@as(usize, 2 * 256), m.layers[3].attn.q_b.rows());
    }
    for ([_]struct { bad: BadMode, err: anyerror }{
        .{ .bad = .extra_tensor, .err = error.UnusedTensors },
        .{ .bad = .stray_scale, .err = error.UnexpectedTensor },
        .{ .bad = .bad_dtype, .err = error.UnsupportedQuantization },
    }) |case| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeTiny(a, tmp, t, case.bad);
        var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, io, try tmpPath(a, tmp));
        defer ck.close();
        var why: core.modelopt.Why = .{};
        try testing.expectError(case.err, fp8_weights.map(testing.allocator, &ck, t, &why));
        if (case.bad == .extra_tensor) {
            try testing.expect(std.mem.indexOf(u8, why.text(), "model.layers.0.self_attn.q_norm.weight") != null);
        }
    }
}

test "W8A16 TP=1/2/4 splits and memory floors match official Hy4-flash-FP8 byte totals" {
    const parsed = try release();
    defer parsed.deinit();
    const t = parsed.value.text_config;
    const experts_bytes: usize = 3_624_321_024;
    const engram_bytes: usize = 47_782_856;
    try testing.expectEqual(@as(usize, 402_782_160), try part.layerBytes(t, 0, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 139_598_800 + experts_bytes + engram_bytes), try part.layerBytes(t, 1, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 207_793_488 + experts_bytes), try part.layerBytes(t, 2, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 139_598_800 + experts_bytes), try part.layerBytes(t, 5, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 158_445_453_396), try part.weightBytes(t, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 126_721_231_872), try part.engramTableBytes(t, 0, 1, .stored));
    try testing.expectEqual(@as(usize, 12 * (1_152 + 256)), part.tokenBytes(t));
    try testing.expectEqual(@as(usize, 30 * 512 * 1_152), part.ringBytes(t));

    var why: core.modelopt.Why = .{};
    for ([_]usize{ 2, 4 }) |world| {
        try part.check(t, world, &why);
        const w0 = try part.weightBytes(t, 0, world, .packed_);
        var row_sum: usize = 0;
        for (0..world) |r| {
            try testing.expectEqual(w0, try part.weightBytes(t, r, world, .packed_));
            const e = try part.Engram.of(t, r, world);
            try testing.expectEqual(r == 0, e.bias);
            row_sum += try part.engramTableBytes(t, r, world, .packed_);
        }
        try testing.expectEqual(@as(usize, 660_006_416 * 100), row_sum);
        if (world == 2) try testing.expect(w0 > 74 * GiB and w0 < 76 * GiB);
        if (world == 4) try testing.expect(w0 > 38 * GiB and w0 < 39 * GiB);
    }
    try testing.expectEqual(part.Share{ .lo = 24, .len = 8 }, (try part.Attn.of(t, 0, 3, 4)).heads);
    try testing.expectEqual(part.Share{ .lo = 48, .len = 16 }, (try part.Attn.of(t, 2, 3, 4)).heads);
    try testing.expectEqual(part.Share{ .lo = 1_152, .len = 384 }, try part.expert(t, 3, 4));
    try testing.expectEqual(part.Share{ .lo = 90_624, .len = 30_208 }, try part.vocab(t, 3, 4));

    try testing.expectError(error.UnsupportedWorld, part.check(t, 0, &why));
    try testing.expectError(error.UnsupportedWorld, part.check(t, 3, &why));
    try testing.expectError(error.UnsupportedWorld, part.check(t, 5, &why));
    var bad = t;
    bad.moe_intermediate_size = 1_408;
    try testing.expectError(error.SplitNotOnBlocks, part.check(bad, 2, &why));
    try testing.expect(std.mem.indexOf(u8, why.text(), "routed expert width 1408") != null);
}

fn refE4m3Rne(x: f32) u8 {
    if (std.math.isNan(x)) return if (@as(u32, @bitCast(x)) >> 31 != 0) 0xff else 0x7f;
    const sign: u8 = if (@as(u32, @bitCast(x)) >> 31 != 0) 0x80 else 0x00;
    const a: f64 = @abs(@as(f64, x));
    if (a > 464.0) return sign | 0x7f;
    if (a == 0.0) return sign;
    var best_code: u8 = 0;
    var best_dist: f64 = a;
    for (1..0x7f) |code| {
        const c: u8 = @intCast(code);
        const d = @abs(a - @abs(core.quant.fp8.e4m3(c)));
        if (d < best_dist or (d == best_dist and (c & 1) == 0)) {
            best_dist = d;
            best_code = c;
        }
    }
    return sign | best_code;
}

test "Engram 100-byte FP8 rows match exact e4m3fn RNE, fp64 scaling, HY4ENGR1 headers, and layer tiling" {
    try testing.expectEqual(@as(u8, 0x00), ep.e4m3Rne(0.0));
    try testing.expectEqual(@as(u8, 0x80), ep.e4m3Rne(-0.0));
    try testing.expectEqual(@as(u8, 0x7e), ep.e4m3Rne(448.0));
    try testing.expectEqual(@as(u8, 0x7f), ep.e4m3Rne(464.0001));
    try testing.expectEqual(@as(u8, 0x38), ep.e4m3Rne(1.0625));
    try testing.expectEqual(@as(u8, 0x3a), ep.e4m3Rne(1.1875));
    for (0..65536) |u| {
        const x: f32 = @bitCast(@as(u32, @intCast(u)) << 16);
        try testing.expectEqual(refE4m3Rne(x), ep.e4m3Rne(x));
    }

    var raw: [ep.dim * 2]u8 = @splat(0);
    var rec: [ep.row_bytes]u8 = undefined;
    ep.packRow(&raw, &rec);
    for (rec[0..ep.dim]) |b| try testing.expectEqual(@as(u8, 0), b);
    var prng = std.Random.DefaultPrng.init(2026);
    for (0..128) |_| {
        var amax: f64 = 1e-12;
        for (0..ep.dim) |j| {
            const u: u16 = @intCast(prng.random().uintLessThan(u32, 0x7f00) | (prng.random().uintLessThan(u32, 2) << 15));
            std.mem.writeInt(u16, raw[2 * j ..][0..2], u, .little);
            amax = @max(amax, @abs(@as(f64, @as(f32, @bitCast(@as(u32, u) << 16)))));
        }
        ep.packRow(&raw, &rec);
        var vals: [ep.dim]f32 = undefined;
        ep.unpackRowF32(&rec, &vals);
        const s64 = amax / ep.fp8_max;
        for (0..ep.dim) |j| try testing.expect(@abs(@as(f64, vals[j]) - core.quant.fp8.e4m3(rec[j]) * s64) / amax <= 1e-6);
    }

    const parsed = try release();
    defer parsed.deinit();
    const t = parsed.value.text_config;
    const expected_rows = [_]u64{ 164_994_380, 164_999_008, 165_004_086, 165_008_942 };
    for (t.engram_layer_ids, expected_rows) |layer_id, want_rows| {
        _ = try engram.Layer.open(t, layer_id);
        const h = try ep.Header.of(t, layer_id);
        try testing.expectEqual(want_rows, h.rows);
        const enc = h.encode();
        const got = try ep.Header.parse(&enc, ep.header_bytes + want_rows * ep.row_bytes);
        try got.checkConfig(t, layer_id);
        try testing.expectError(error.BadHy4EngramPack, ep.Header.parse(&enc, ep.header_bytes + want_rows * ep.row_bytes - 1));
        var bad_magic = enc;
        bad_magic[0] ^= 1;
        try testing.expectError(error.NotHy4EngramPack, ep.Header.parse(&bad_magic, ep.header_bytes + want_rows * ep.row_bytes));
        var tiled: usize = 0;
        for (0..4) |r| {
            const sh = try (try part.Engram.of(t, r, 4)).rows(t, layer_id);
            try testing.expectEqual(tiled, sh.lo);
            tiled += sh.len;
        }
        try testing.expectEqual(@as(usize, @intCast(want_rows)), tiled);
    }
    var seeded = t;
    seeded.engram_seed = 1;
    const h2 = try ep.Header.of(t, 2);
    try testing.expectError(error.UnsupportedEngram, h2.checkConfig(seeded, 2));
}
