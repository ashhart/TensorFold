//! Execution-slot geometry and text tensor selection on the host.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const loop_weights = @import("loop_weights.zig");
const t = std.testing;

fn config(kind: []const u8, extra: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(t.allocator);
    try out.print(t.allocator, "{{\"model_type\":\"{s}\",\"tie_word_embeddings\":false,{s}\"text_config\":{{", .{ kind, extra });
    try out.appendSlice(t.allocator,
        \\"model_type":"qwen3_5_text","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu",
        \\"head_dim":256,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        \\"hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":32,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":32,"num_attention_heads":16,"num_key_value_heads":4,
        \\"rms_norm_eps":1e-6,"max_position_embeddings":262144,"layer_types":[
    );
    for (0..32) |i| {
        if (i > 0) try out.append(t.allocator, ',');
        try out.appendSlice(t.allocator, if (i % 4 == 3) "\"full_attention\"" else "\"linear_attention\"");
    }
    try out.appendSlice(t.allocator, "] ,\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25}}}");
    return out.toOwnedSlice(t.allocator);
}

const settings = "\"loop_settings\":{\"start\":4,\"end\":28,\"eval_count\":2,\"counts\":[2]},\"mhc\":{\"start\":4,\"end\":28,\"streams\":2,\"iterations\":20,\"fixed_identity\":false},";

test "SenseNova loop: geometry counts executions and keeps physical weights" {
    const bytes = try config("sensenova_loop", settings);
    defer t.allocator.free(bytes);
    const g = try geometry.parse(t.allocator, bytes);
    try t.expect(g.bf16 and !g.tied and g.loop != null);
    try t.expectEqual(@as(usize, 32), g.layers);
    try t.expectEqual(@as(usize, 24), g.physicalDeltas());
    try t.expectEqual(@as(usize, 8), g.physicalAttentions());
    try t.expectEqual(@as(usize, 42), g.deltas());
    try t.expectEqual(@as(usize, 14), g.attentions());
    try t.expectEqual(@as(usize, 56), g.schedule().len);
    try t.expectEqual(42 * (g.recBytes() + g.convBytes()), g.stateBytes());
    try t.expectEqual(@as(usize, 14 * 2 * 4 * 128 * 512), g.kvBytes(128, false));
    try t.expect(geometry.supported(g, 1, false, 64));
    const layout: @import("cuda_reuse.zig").Layout = .{ .g = g, .format = .bf16, .cap = 128, .at = 64 };
    try t.expectEqual(@as(usize, 1 + 14 * 2 * 4), layout.count());
    try t.expectEqual(@as(u64, g.stateBytes() + 14 * 2 * 4 * 64 * 512), layout.total());
    for (1..5) |loops| {
        const overridden = try g.withLoops(loops);
        try t.expectEqual(6 + 18 * loops, overridden.deltas());
        try t.expectEqual(2 + 6 * loops, overridden.attentions());
    }
    try t.expectError(error.BadStageRange, g.stage(0, 16, false));
    try t.expect(loop_weights.loaded(g, "lm_head.weight"));
    try t.expect(loop_weights.loaded(g, "model.language_model.layers.4.mhc_attn.weight"));
    try t.expect(!loop_weights.loaded(g, "model.language_model.layers.3.mhc_attn.weight"));
    try t.expect(!loop_weights.loaded(g, "model.language_model.loop_gate.logits"));
    try t.expect(!loop_weights.loaded(g, "model.visual.blocks.0.weight"));
    try t.expect(!loop_weights.loaded(g, "mtp.fc.weight"));
}

test "SenseNova loop: Qwen identity state and KV sizes stay physical" {
    const bytes = try config("qwen3_5", "");
    defer t.allocator.free(bytes);
    const g = try geometry.parse(t.allocator, bytes);
    try t.expect(g.loop == null);
    try t.expectEqual(@as(usize, 24), g.deltas());
    try t.expectEqual(@as(usize, 8), g.attentions());
    try t.expectEqual(@as(usize, 24 * (32 * 128 * 128 * 4 + 3 * (2 * 16 + 32) * 128 * 2)), g.stateBytes());
    try t.expectEqual(@as(usize, 8 * 2 * 4 * 128 * 512), g.kvBytes(128, false));
    const s = g.schedule();
    for (s.slice(), 0..) |step, i| try t.expectEqual(i, step.layer);
}

test "SenseNova loop: in-tree 2B and 27B identity geometry keeps existing state sizes" {
    const Model = struct { layers: usize, hidden: usize, intermediate: usize, values: usize, queries: usize, kv: usize };
    const models = [_]Model{
        .{ .layers = 24, .hidden = 2048, .intermediate = 6144, .values = 16, .queries = 8, .kv = 2 },
        .{ .layers = 64, .hidden = 5120, .intermediate = 17408, .values = 48, .queries = 24, .kv = 4 },
    };
    for (models) |m| {
        var g: geometry.Geometry = .{ .hidden = m.hidden, .intermediate = m.intermediate, .layers = m.layers, .vocab = 248320, .k_heads = 16, .v_heads = m.values, .q_heads = m.queries, .kv_heads = m.kv, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
        for (0..m.layers) |i| g.linear[i] = i % 4 != 3;
        const delta = m.layers * 3 / 4;
        const attention = m.layers / 4;
        try t.expectEqual(delta, g.deltas());
        try t.expectEqual(attention, g.attentions());
        try t.expectEqual(delta * (m.values * 128 * 128 * 4 + 3 * (32 + m.values) * 128 * 2), g.stateBytes());
        try t.expectEqual(attention * 2 * m.kv * 128 * 512, g.kvBytes(128, false));
    }
}

const MockArena = struct {
    ptr: u64 = 0,
    copies: usize = 0,
    pub fn upload(self: *MockArena, _: u64, _: []const u8) !void {
        self.copies += 1;
    }
};
const MockLoad = struct {
    g: geometry.Geometry,
    arena: MockArena = .{},
    at: u64 = 0,
    pub fn alloc(self: *MockLoad, bytes: usize) !u64 {
        const at = self.at;
        self.at += bytes;
        return at;
    }
};

test "SenseNova loop: FP32 routers check shape and dtype before upload" {
    const header =
        \\{"model.language_model.layers.4.mhc_attn.weight":{"dtype":"F32","shape":[8,16],"data_offsets":[0,512]},
        \\ "model.language_model.layers.4.mhc_attn.bias":{"dtype":"F32","shape":[8],"data_offsets":[512,544]},
        \\ "model.language_model.layers.4.mhc_ffn.weight":{"dtype":"F32","shape":[8,16],"data_offsets":[544,1056]},
        \\ "model.language_model.layers.4.mhc_ffn.bias":{"dtype":"F32","shape":[8],"data_offsets":[1056,1088]}}
    ;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    var bytes: [8 + header.len + 1088]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = &bytes });
    var ck = try @import("core").Checkpoint.openModel(t.allocator, t.io, dir);
    defer ck.close();
    const g: geometry.Geometry = .{ .hidden = 8, .intermediate = 16, .layers = 1, .vocab = 2, .k_heads = 1, .v_heads = 1, .q_heads = 1, .kv_heads = 1, .eps = 1e-6, .tied = false, .context = 32, .linear = @splat(false) };
    var e: MockLoad = .{ .g = g };
    const result = try loop_weights.load(&e, &ck, "model.language_model.layers.4.");
    try t.expectEqual(@as(usize, 4), e.arena.copies);
    try t.expectEqual(@as(u64, 0), result.attn_weight);
    try t.expectEqual(@as(u64, 512), result.attn_bias);
    try t.expectEqual(@as(u64, 544), result.ffn_weight);
    try t.expectEqual(@as(u64, 1056), result.ffn_bias);
    const weight = try ck.get("model.language_model.layers.4.mhc_attn.weight");
    try t.expect(!weight.is(.bf16, &.{ 8, 16 }));
    try t.expect(!weight.is(.f32, &.{ 8, 18 }));
}

test "SenseNova loop: MoE and MTP configurations are refused" {
    const original = try config("sensenova_loop", settings);
    defer t.allocator.free(original);
    const moe = try std.mem.replaceOwned(u8, t.allocator, original, "\"intermediate_size\":12288", "\"intermediate_size\":12288,\"num_experts\":8");
    defer t.allocator.free(moe);
    try t.expectError(error.SenseNovaMoe, geometry.parse(t.allocator, moe));
    const mtp = try std.mem.replaceOwned(u8, t.allocator, original, "\"intermediate_size\":12288", "\"intermediate_size\":12288,\"mtp_num_hidden_layers\":1");
    defer t.allocator.free(mtp);
    try t.expectError(error.SenseNovaMtp, geometry.parse(t.allocator, mtp));
    const pocket = try std.mem.replaceOwned(u8, t.allocator, original, "\"intermediate_size\":12288", "\"intermediate_size\":12288,\"tensorfold_learned_neurons\":64");
    defer t.allocator.free(pocket);
    try t.expectError(error.SenseNovaPocket, geometry.parse(t.allocator, pocket));
}

test "SenseNova loop: checkpoint YaRN settings pass CUDA geometry" {
    const original = try config("sensenova_loop", settings);
    defer t.allocator.free(original);
    const context = try std.mem.replaceOwned(u8, t.allocator, original, "\"max_position_embeddings\":262144", "\"max_position_embeddings\":524288");
    defer t.allocator.free(context);
    const yarn = try std.mem.replaceOwned(u8, t.allocator, context, "\"rope_type\":\"default\"", "\"rope_type\":\"yarn\",\"factor\":2.0,\"original_max_position_embeddings\":262144");
    defer t.allocator.free(yarn);
    const g = try geometry.parse(t.allocator, yarn);
    try t.expectEqual(@as(usize, 524288), g.context);
    try t.expectEqual(@as(f64, 2), g.rope.yarn.factor);
}

test "SenseNova loop: a direct tap request is refused" {
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.g.loop = .{ .start = 1, .end = 2, .count = 2, .mhc = null };
    try t.expectError(error.SenseNovaDrafter, e.setTaps(&.{1}));
}

test "SenseNova loop: F32 delta bias converts to BF16 with ties to even" {
    const bits = [_]u32{ 0x3f808000, 0x3f818000, 0xbf808000, 0x3f807fff, 0x3f808001 };
    const tensor: @import("core").checkpoint.Tensor = .{ .dtype = .f32, .rank = 1, .shape = .{ 5, 0, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(&bits) };
    var output: [5]u16 = undefined;
    try loop_weights.convertDtBias(tensor, &output);
    try t.expectEqualSlices(u16, &.{ 0x3f80, 0x3f82, 0xbf80, 0x3f80, 0x3f81 }, &output);
    var wrong = tensor;
    wrong.dtype = .f16;
    try t.expectError(error.UnexpectedTensor, loop_weights.convertDtBias(wrong, &output));
    wrong = tensor;
    wrong.shape[0] = 4;
    try t.expectError(error.UnexpectedTensor, loop_weights.convertDtBias(wrong, &output));
}

test "SenseNova loop: mHC scratch is persistent and counted by capacity" {
    const bytes = try config("sensenova_loop", settings);
    defer t.allocator.free(bytes);
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.g = try geometry.parse(t.allocator, bytes);
    e.mhc = .{};
    e.scratch_buf = .{ .d = undefined, .ptr = 0x200000, .len = 1 << 30 };
    const rows = 17;
    const with_mhc = try e.scratch(rows, .allocate);
    const streams = e.mhc.streams;
    const post = e.mhc.post;
    const mix = e.mhc.mix;
    try t.expect(post >= streams + rows * 2 * e.g.hidden * 2);
    try t.expect(mix >= post + rows * 2 * 4);
    try t.expect(e.scratch_buf.?.ptr + with_mhc >= mix + rows * 4 * 4);
    e.g.loop = null;
    const without = try e.scratch(rows, .measure);
    const round_up = @import("cuda_scratch.zig").aligned;
    try t.expectEqual(round_up(rows * 2 * e.g.hidden * 2) + round_up(rows * 2 * 4) + round_up(rows * 4 * 4), with_mhc - without);
}

test {
    _ = @import("cuda_mhc_test.zig");
}

test "SenseNova loop: unsupported mHC settings are refused" {
    const original = try config("sensenova_loop", settings);
    defer t.allocator.free(original);
    const replacements = [_][2][]const u8{
        .{ "\"iterations\":20", "\"iterations\":19" },
        .{ "\"fixed_identity\":false", "\"fixed_identity\":true" },
        .{ "\"iterations\":20", "\"iterations\":20,\"pre_scale\":2" },
    };
    for (replacements) |pair| {
        const changed = try std.mem.replaceOwned(u8, t.allocator, original, pair[0], pair[1]);
        defer t.allocator.free(changed);
        try t.expectError(error.UnsupportedSenseNovaMhc, geometry.parse(t.allocator, changed));
    }
}
