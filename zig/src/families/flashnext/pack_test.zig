//! The weight-pack builder against a synthetic checkpoint, each transform checked against a hand-computed value.
const std = @import("std");
const Io = std.Io;
const pack = @import("pack.zig");
const st = @import("../../core/safetensors.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;

const bf16_one: u16 = 0x3F80; // 1.0
const bf16_half: u16 = 0x3F00; // 0.5
const bf16_one_and_a_quarter: u16 = 0x3FA0; // 1.25
const f32_two_and_a_quarter: u32 = 0x40100000; // 1.0 + 1.25
const f32_eps: u32 = 0x358637BD; // 1e-6

/// A tensor's explicit bytes for the synthetic shard.
const Spec = struct { name: []const u8, dtype: []const u8, shape: []const u64, bytes: []const u8 };

/// One quantized linear's three tensors: u32 words base+i, then bf16 scales 0x3C00+row and biases 0x3B00+row.
fn qspecs(list: *std.ArrayList(Spec), a: std.mem.Allocator, name: []const u8, rows: usize, base: u32, scale_base: u16, bias_base: u16) !void {
    const wshape = try a.dupe(u64, &.{ rows, 6 });
    const sshape = try a.dupe(u64, &.{ rows, 1 });
    const words = try a.alloc(u32, rows * 6);
    for (words, 0..) |*w, i| w.* = base + @as(u32, @intCast(i));
    const s = try a.alloc(u16, rows);
    for (s, 0..) |*v, j| v.* = scale_base + @as(u16, @intCast(j % 256));
    const b = try a.alloc(u16, rows);
    for (b, 0..) |*v, j| v.* = bias_base + @as(u16, @intCast(j % 256));
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{name}), .dtype = "U32", .shape = wshape, .bytes = std.mem.sliceAsBytes(words) });
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.scales", .{name}), .dtype = "BF16", .shape = sshape, .bytes = std.mem.sliceAsBytes(s) });
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.biases", .{name}), .dtype = "BF16", .shape = sshape, .bytes = std.mem.sliceAsBytes(b) });
}

/// One hc module: down (32 rows), block_inject (32 rows, inject only), hc_norm (1.25), up (32 rows).
fn hcSpecs(list: *std.ArrayList(Spec), a: std.mem.Allocator, stem: []const u8, base: u32, inject: bool, down_scale: u16, inject_scale: u16) !void {
    try qspecs(list, a, try std.fmt.allocPrint(a, "{s}.input_mix_weight_down", .{stem}), 32, base, down_scale, down_scale - 0x100);
    if (inject) try qspecs(list, a, try std.fmt.allocPrint(a, "{s}.block_inject_weight", .{stem}), 32, base + 1_000_000, inject_scale, inject_scale - 0x100);
    const norm = try a.alloc(u16, 32);
    @memset(norm, bf16_one_and_a_quarter);
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.hc_norm.weight", .{stem}), .dtype = "BF16", .shape = &.{32}, .bytes = std.mem.sliceAsBytes(norm) });
    try qspecs(list, a, try std.fmt.allocPrint(a, "{s}.input_mix_weight_up", .{stem}), 32, base + 2_000_000, down_scale, down_scale - 0x100);
}

fn gateSpecs(list: *std.ArrayList(Spec), a: std.mem.Allocator, stem: []const u8, base: u16) !void {
    const gate = try a.alloc(u16, 4 * 32);
    for (gate, 0..) |*v, i| v.* = base + @as(u16, @intCast(i % 256));
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.mlp.gate.weight", .{stem}), .dtype = "BF16", .shape = &.{ 4, 32 }, .bytes = std.mem.sliceAsBytes(gate) });
    // The shared gate is a u32-affine linear whose dequantized row joins the router.
    const words = try a.alloc(u32, 6);
    words[0] = 0xC00000C0;
    words[1] = 0x0000000F;
    words[2] = 0x00000300; // code 12 (bit 72) = 3
    @memset(words[3..], 0);
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.mlp.shared_expert_gate.weight", .{stem}), .dtype = "U32", .shape = &.{ 1, 6 }, .bytes = std.mem.sliceAsBytes(words) });
    const one = try a.alloc(u16, 1);
    one[0] = 0x3F80; // 1.0
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.mlp.shared_expert_gate.scales", .{stem}), .dtype = "BF16", .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(one) });
    const zero = try a.alloc(u16, 1);
    zero[0] = 0x0000; // 0.0
    try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.mlp.shared_expert_gate.biases", .{stem}), .dtype = "BF16", .shape = &.{ 1, 1 }, .bytes = std.mem.sliceAsBytes(zero) });
}

const config_json =
    \\{
    \\  "model_type": "qwen4_exp",
    \\  "hidden_size": 32,
    \\  "num_attention_heads": 2,
    \\  "head_dim": 8,
    \\  "vocab_size": 128,
    \\  "num_hidden_layers": 2,
    \\  "rms_norm_eps": 1e-6,
    \\  "num_key_value_heads": 1,
    \\  "linear_num_key_heads": 1,
    \\  "linear_key_head_dim": 8,
    \\  "linear_num_value_heads": 1,
    \\  "linear_value_head_dim": 32,
    \\  "linear_conv_kernel_dim": 4,
    \\  "num_experts": 4,
    \\  "num_experts_per_tok": 1,
    \\  "moe_intermediate_size": 16,
    \\  "shared_expert_intermediate_size": 16,
    \\  "hc_count": 1,
    \\  "hc_lowrank": 32,
    \\  "indexer_n_heads": 2,
    \\  "indexer_head_dim": 16,
    \\  "ngram_size": 3,
    \\  "heads_per_ngram": 1,
    \\  "split_ngram_parts": 16,
    \\  "ple_embed_dim": 8,
    \\  "ple_layer_ids": [2],
    \\  "layer_types": ["linear_attention", "full_attention"],
    \\  "quantization": {"bits": 6, "group_size": 32, "quant_method": "mlx"},
    \\  "mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
    \\}
;

/// The synthetic checkpoint is gdn then attention, and every quantized K is 32 so stacks land on the lane width.
pub fn writeCheckpoint(tmp: std.testing.TmpDir, a: std.mem.Allocator) !void {
    var list: std.ArrayList(Spec) = .empty;
    for (0..2) |i| {
        const stem = try std.fmt.allocPrint(a, "language_model.model.layers.{d}", .{i});
        try hcSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.attn_hyper_connection", .{stem}), 3_000_000 + @as(u32, @intCast(i)) * 0x1000000, true, 0x3C00, 0x3400);
        try hcSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.mlp_hyper_connection", .{stem}), 3_500_000 + @as(u32, @intCast(i)) * 0x1000000, true, 0x3C00, 0x3400);
        try gateSpecs(&list, a, stem, 0x3F80);
    }
    { // layer 0's gdn block
        const stem = "language_model.model.layers.0.linear_attn";
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.in_proj_qkv", .{stem}), 24, 1_000_000, 0x3C00, 0x3B00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.in_proj_z", .{stem}), 4, 1_100_000, 0x3D00, 0x3C00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.in_proj_b", .{stem}), 2, 1_200_000, 0x3E00, 0x3D00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.in_proj_a", .{stem}), 2, 1_300_000, 0x3F00, 0x3E00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.out_proj", .{stem}), 32, 1_400_000, 0x3C00, 0x3B00);
        const conv = try a.alloc(u16, 32 * 4);
        @memset(conv, bf16_half);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.conv1d.weight", .{stem}), .dtype = "BF16", .shape = &.{ 32, 4, 1 }, .bytes = std.mem.sliceAsBytes(conv) });
        const alog = try a.alloc(u32, 8);
        for (alog, 0..) |*v, i| v.* = @bitCast(@as(f32, @floatFromInt(2 + i))); // 2.0..9.0, exact in bf16
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.A_log", .{stem}), .dtype = "F32", .shape = &.{8}, .bytes = std.mem.sliceAsBytes(alog) });
        const dt = try a.alloc(u32, 8);
        for (dt, 0..) |*v, i| v.* = @bitCast(1.5 + @as(f32, @floatFromInt(i))); // 1.5..8.5, exact in bf16
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.dt_bias", .{stem}), .dtype = "F32", .shape = &.{8}, .bytes = std.mem.sliceAsBytes(dt) });
        const norm = try a.alloc(u16, 32);
        @memset(norm, bf16_one);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.norm.weight", .{stem}), .dtype = "BF16", .shape = &.{32}, .bytes = std.mem.sliceAsBytes(norm) });
    }
    { // layer 1's attention block and the PLE tables
        const stem = "language_model.model.layers.1";
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.self_attn.q_proj", .{stem}), 32, 5_000_000, 0x3C00, 0x3B00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.self_attn.k_proj", .{stem}), 8, 5_100_000, 0x3D00, 0x3C00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.self_attn.v_proj", .{stem}), 8, 5_200_000, 0x3E00, 0x3D00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.self_attn.indexer.index_qk_proj", .{stem}), 48, 5_300_000, 0x3F00, 0x3E00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.self_attn.o_proj", .{stem}), 32, 5_400_000, 0x3C00, 0x3B00);
        for ([_][]const u8{ "q_norm", "k_norm" }) |n| {
            const norm = try a.alloc(u16, 8);
            @memset(norm, bf16_one_and_a_quarter);
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.self_attn.{s}.weight", .{ stem, n }), .dtype = "BF16", .shape = &.{8}, .bytes = std.mem.sliceAsBytes(norm) });
        }
        for ([_][]const u8{ "q_layernorm", "k_layernorm" }) |n| {
            const norm = try a.alloc(u16, 16);
            @memset(norm, bf16_one_and_a_quarter);
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.self_attn.indexer.{s}.weight", .{ stem, n }), .dtype = "BF16", .shape = &.{16}, .bytes = std.mem.sliceAsBytes(norm) });
        }
        const ple = "language_model.model.layers.1.ple";
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.key_proj", .{ple}), 32, 7_000_000, 0x3C00, 0x3B00);
        try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.value_proj", .{ple}), 32, 7_100_000, 0x3D00, 0x3C00);
        for ([_][]const u8{ "norm_key", "norm_query", "norm_conv" }) |n| {
            const norm = try a.alloc(u16, 32);
            @memset(norm, bf16_one_and_a_quarter);
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.{s}.weight", .{ ple, n }), .dtype = "BF16", .shape = &.{32}, .bytes = std.mem.sliceAsBytes(norm) });
        }
        const pconv = try a.alloc(u16, 32 * 4);
        @memset(pconv, bf16_half);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.conv1d.weight", .{ple}), .dtype = "BF16", .shape = &.{ 32, 4, 1 }, .bytes = std.mem.sliceAsBytes(pconv) });
        for (0..16) |s| {
            const rows = s + 1;
            const emb = try a.alloc(u16, rows * 4);
            @memset(emb, bf16_half);
            try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.ple_embedding.ngram_embedding.shard_{d}.weight", .{ ple, s }), .dtype = "BF16", .shape = try a.dupe(u64, &.{ rows, 4 }), .bytes = std.mem.sliceAsBytes(emb) });
        }
    }
    try hcSpecs(&list, a, "language_model.model.hyper_connection_mixer", 8_000_000, false, 0x3C00, 0x3400);
    // the vision tower's rank-five entry: the builder never reads it, and it must not stop the load
    try list.append(a, .{ .name = "model.visual.patch_embed.proj.weight", .dtype = "BF16", .shape = &.{ 1, 1, 1, 1, 1 }, .bytes = &.{ 0, 0 } });
    try qspecs(&list, a, "language_model.lm_head", 128, 500_000, 0x3C00, 0x3B00);
    const mtp = "language_model.mtp";
    try hcSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.attn_hyper_connection", .{mtp}), 13_000_000, true, 0x3C00, 0x3400);
    try hcSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.mlp_hyper_connection", .{mtp}), 13_500_000, true, 0x3C00, 0x3400);
    try gateSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0", .{mtp}), 0x3F80);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.q_proj", .{mtp}), 32, 15_000_000, 0x3C00, 0x3B00);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.k_proj", .{mtp}), 8, 15_100_000, 0x3D00, 0x3C00);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.v_proj", .{mtp}), 8, 15_200_000, 0x3E00, 0x3D00);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.indexer.index_qk_proj", .{mtp}), 48, 15_300_000, 0x3F00, 0x3E00);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.o_proj", .{mtp}), 32, 15_400_000, 0x3C00, 0x3B00);
    for ([_][]const u8{ "q_norm", "k_norm" }) |n| {
        const norm = try a.alloc(u16, 8);
        @memset(norm, bf16_one_and_a_quarter);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.{s}.weight", .{ mtp, n }), .dtype = "BF16", .shape = &.{8}, .bytes = std.mem.sliceAsBytes(norm) });
    }
    for ([_][]const u8{ "q_layernorm", "k_layernorm" }) |n| {
        const norm = try a.alloc(u16, 16);
        @memset(norm, bf16_one_and_a_quarter);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.layers.0.self_attn.indexer.{s}.weight", .{ mtp, n }), .dtype = "BF16", .shape = &.{16}, .bytes = std.mem.sliceAsBytes(norm) });
    }
    try hcSpecs(&list, a, try std.fmt.allocPrint(a, "{s}.hyper_connection_mixer", .{mtp}), 18_000_000, false, 0x3C00, 0x3400);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.fc_embedding", .{mtp}), 32, 16_000_000, 0x3C00, 0x3B00);
    try qspecs(&list, a, try std.fmt.allocPrint(a, "{s}.fc_hidden", .{mtp}), 32, 16_100_000, 0x3D00, 0x3C00);
    for ([_][]const u8{ "pre_fc_norm_embedding", "pre_fc_norm_hidden" }) |n| {
        const norm = try a.alloc(u16, 32);
        @memset(norm, bf16_one_and_a_quarter);
        try list.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.{s}.weight", .{ mtp, n }), .dtype = "BF16", .shape = &.{32}, .bytes = std.mem.sliceAsBytes(norm) });
    }
    // one shard holding every tensor, plus the index the PLE starts and the shard spelling come from
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: u64 = 0;
    var index: std.Io.Writer.Allocating = .init(a);
    try index.writer.writeAll("{\"weight_map\":{");
    for (list.items, 0..) |s, i| {
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", s.name, s.dtype });
        for (s.shape, 0..) |d, j| try header.writer.print("{s}{d}", .{ if (j == 0) "" else ",", d });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + s.bytes.len });
        at += s.bytes.len;
        try index.writer.print("{s}\"{s}\":\"model.safetensors\"", .{ if (i == 0) "" else ",", s.name });
    }
    try header.writer.writeAll("}");
    try index.writer.writeAll("}}");
    const head_bytes = header.written();
    var image = try a.alloc(u8, 8 + head_bytes.len + at);
    std.mem.writeInt(u64, image[0..8], head_bytes.len, .little);
    @memcpy(image[8..][0..head_bytes.len], head_bytes);
    var p: usize = 8 + head_bytes.len;
    for (list.items) |s| {
        @memcpy(image[p..][0..s.bytes.len], s.bytes);
        p += s.bytes.len;
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
}

pub fn tmpPath(a: std.mem.Allocator, tmp: std.testing.TmpDir, comptime name: []const u8) ![]u8 {
    // std.testing.tmpDir parents the dir at <cwd>/.zig-cache/tmp
    const cwd = try std.process.currentPathAlloc(io, a);
    defer a.free(cwd);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}/{s}", .{ cwd, tmp.sub_path, name });
}

fn tensorBytes(file: *st.File, name: []const u8) []const u8 {
    const e = file.names.get(name) orelse return "";
    return file.map.memory[file.data + e.begin .. file.data + e.end];
}

fn expectTensor(file: *st.File, name: []const u8, dtype: st.DType, shape: []const usize) !void {
    const e = file.names.get(name) orelse return error.MissingPackTensor;
    try std.testing.expectEqual(dtype, e.dtype);
    try std.testing.expectEqualSlices(usize, shape, e.shape[0..e.rank]);
}

test "centered scale subtracts one only when the checkpoint stores gamma around one" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var list: std.ArrayList(Spec) = .empty;
    const bits = [_]u16{ bf16_one_and_a_quarter, bf16_half, bf16_one };
    try list.append(a, .{ .name = "w", .dtype = "BF16", .shape = &.{3}, .bytes = std.mem.sliceAsBytes(bits[0..]) });
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.print("{{\"w\":{{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[0,{d}]}}}}", .{bits.len * 2});
    const head_bytes = header.written();
    var image = try a.alloc(u8, 8 + head_bytes.len + bits.len * 2);
    defer a.free(image);
    std.mem.writeInt(u64, image[0..8], head_bytes.len, .little);
    @memcpy(image[8..][0..head_bytes.len], head_bytes);
    @memcpy(image[8 + head_bytes.len ..], std.mem.sliceAsBytes(bits[0..]));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
    const dir = try tmpPath(a, tmp, ".");
    defer a.free(dir);
    var ck = try @import("../../core/checkpoint.zig").Checkpoint.openModel(a, io, dir);
    defer ck.close();
    // Stored around one, the pack holds the f32 round trip, so 1.25, 0.5 and 1.0 stay unchanged.
    const around_one = try pack.centeredScale(a, &ck, "w", true);
    defer a.free(around_one);
    const one_words = std.mem.bytesAsSlice(u32, around_one);
    try std.testing.expectEqual(@as(u32, 0x3FA00000), one_words[0]); // 1.0 + (1.25 - 1.0)
    try std.testing.expectEqual(@as(u32, 0x3F000000), one_words[1]); // 1.0 + (0.5 - 1.0)
    try std.testing.expectEqual(@as(u32, 0x3F800000), one_words[2]); // 1.0 + (1.0 - 1.0)
    // stored around zero: the -1 step does not run, so the pack holds 1.0 + fp32(w) directly
    const around_zero = try pack.centeredScale(a, &ck, "w", false);
    defer a.free(around_zero);
    const zero_words = std.mem.bytesAsSlice(u32, around_zero);
    try std.testing.expectEqual(f32_two_and_a_quarter, zero_words[0]); // 1.0 + 1.25
    try std.testing.expectEqual(@as(u32, 0x3FC00000), zero_words[1]); // 1.0 + 0.5
    try std.testing.expectEqual(@as(u32, 0x40000000), zero_words[2]); // 1.0 + 1.0
}

test "norm storage detection reads every layer's hc norm mean and refuses an ambiguous checkpoint" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_]struct { value: u16, want: ?bool }{
        .{ .value = bf16_one, .want = true }, // mean 1.0: share 1.0, median 1.0
        .{ .value = bf16_half, .want = null }, // mean 0.5: share 0.0 but median 0.5 outside both bands
        .{ .value = 0x3E80, .want = false }, // 0.25: share 0.0, median 0.25
    }) |case| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var list: std.ArrayList(Spec) = .empty;
        defer list.deinit(a);
        var header: std.Io.Writer.Allocating = .init(a);
        defer header.deinit();
        try header.writer.writeAll("{");
        var at: u64 = 0;
        const norm = try a.alloc(u16, 32);
        defer a.free(norm);
        @memset(norm, case.value);
        for (0..8) |i| {
            const name = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
            defer a.free(name);
            if (i > 0) try header.writer.writeAll(",");
            try header.writer.print("\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[32],\"data_offsets\":[{d},{d}]}}", .{ name, at, at + 64 });
            at += 64;
        }
        try header.writer.writeAll("}");
        const head_bytes = header.written();
        var image = try a.alloc(u8, 8 + head_bytes.len + at);
        defer a.free(image);
        std.mem.writeInt(u64, image[0..8], head_bytes.len, .little);
        @memcpy(image[8..][0..head_bytes.len], head_bytes);
        for (0..8) |i| @memcpy(image[8 + head_bytes.len + i * 64 ..][0..64], std.mem.sliceAsBytes(norm[0..32]));
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
        const config = try std.fmt.allocPrint(a,
            \\{{"model_type": "qwen4_exp", "hidden_size": 32, "num_attention_heads": 2, "vocab_size": 128,
            \\  "num_hidden_layers": 8, "rms_norm_eps": 1e-6, "num_key_value_heads": 1, "linear_num_key_heads": 1,
            \\  "linear_key_head_dim": 8, "linear_num_value_heads": 1, "linear_value_head_dim": 32,
            \\  "linear_conv_kernel_dim": 4, "num_experts": 4, "num_experts_per_tok": 1, "moe_intermediate_size": 16,
            \\  "shared_expert_intermediate_size": 16, "hc_count": 1, "hc_lowrank": 32,
            \\  "layer_types": ["full_attention","full_attention","full_attention","full_attention",
            \\    "full_attention","full_attention","full_attention","full_attention"],
            \\  "quantization": {{"bits": 6, "group_size": 32, "quant_method": "mlx"}}}}
        , .{});
        defer a.free(config);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        const dir = try tmpPath(a, tmp, ".");
        defer a.free(dir);
        var cfg = try @import("config.zig").Config.read(a, io, dir);
        defer cfg.deinit();
        var ck = try @import("../../core/checkpoint.zig").Checkpoint.openModel(a, io, dir);
        defer ck.close();
        if (case.want) |want| {
            try std.testing.expectEqual(want, try pack.normsAroundOne(&ck, &cfg, 32));
        } else {
            try std.testing.expectError(error.AmbiguousNormStorage, pack.normsAroundOne(&ck, &cfg, 32));
        }
    }
    // model.py:353 returns false before reading anything when there are fewer than 8 anchors
    {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const config = try std.fmt.allocPrint(a,
            \\{{"model_type": "qwen4_exp", "hidden_size": 32, "num_attention_heads": 2, "vocab_size": 128,
            \\  "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "num_key_value_heads": 1, "linear_num_key_heads": 1,
            \\  "linear_key_head_dim": 8, "linear_num_value_heads": 1, "linear_value_head_dim": 32,
            \\  "linear_conv_kernel_dim": 4, "num_experts": 4, "num_experts_per_tok": 1, "moe_intermediate_size": 16,
            \\  "shared_expert_intermediate_size": 16, "hc_count": 1, "hc_lowrank": 32,
            \\  "layer_types": ["full_attention", "full_attention"],
            \\  "quantization": {{"bits": 6, "group_size": 32, "quant_method": "mlx"}}}}
        , .{});
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        var header: std.Io.Writer.Allocating = .init(a);
        try header.writer.writeAll("{\"w\":{\"dtype\":\"BF16\",\"shape\":[4],\"data_offsets\":[0,8]}}");
        const head_bytes = header.written();
        var image = try a.alloc(u8, 8 + head_bytes.len + 8);
        std.mem.writeInt(u64, image[0..8], head_bytes.len, .little);
        @memcpy(image[8..][0..head_bytes.len], head_bytes);
        @memset(image[8 + head_bytes.len ..], 0);
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image });
        const dir = try tmpPath(a, tmp, ".");
        var cfg = try @import("config.zig").Config.read(a, io, dir);
        defer cfg.deinit();
        var ck = try @import("../../core/checkpoint.zig").Checkpoint.openModel(a, io, dir);
        defer ck.close();
        try std.testing.expectEqual(false, try pack.normsAroundOne(&ck, &cfg, 32));
    }
}

test "the full build writes the three packs byte for byte against hand-computed values" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpoint(tmp, a);
    const model_dir = try tmpPath(a, tmp, ".");
    defer a.free(model_dir);
    const out_dir = try tmpPath(a, tmp, "out");
    defer a.free(out_dir);
    try tmp.dir.createDirPath(io, "out");
    const report = try pack.build(a, io, model_dir, out_dir, null);
    try std.testing.expectEqual(@as(usize, 63), report.decode_tensors);
    try std.testing.expectEqual(@as(usize, 15), report.mlx_tensors);
    try std.testing.expectEqual(@as(usize, 9), report.mtp_mlx_tensors);
    try std.testing.expect(!report.norms_around_one); // model.py:353: fewer than 8 anchors stores around zero

    const pack_path = try tmpPath(a, tmp, "out/pack.safetensors");
    defer a.free(pack_path);
    var file = try st.File.open(a, io, pack_path);
    defer file.close(io);
    try expectTensor(&file, "eps", .f32, &.{1});
    try std.testing.expectEqual(f32_eps, std.mem.bytesAsSlice(u32, tensorBytes(&file, "eps"))[0]);
    try expectTensor(&file, "L0.ahc.scale", .f32, &.{32});
    for (std.mem.bytesAsSlice(u32, tensorBytes(&file, "L0.ahc.scale"))) |w| try std.testing.expectEqual(f32_two_and_a_quarter, w); // 1.0 + 1.25, the around-zero branch
    // down stack: input_mix_weight_down (base 3,000,000) then block_inject (base 3,100,000), pure byte concat
    try expectTensor(&file, "L0.ahc.down.w", .u32, &.{ 64, 6 });
    const down = std.mem.bytesAsSlice(u32, tensorBytes(&file, "L0.ahc.down.w"));
    try std.testing.expectEqual(@as(u32, 3_000_000), down[0]);
    try std.testing.expectEqual(@as(u32, 4_000_000), down[32 * 6]); // block_inject row 0
    try expectTensor(&file, "L0.ahc.down.s", .bf16, &.{ 64, 1 });
    const down_s = std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.ahc.down.s"));
    try std.testing.expectEqual(@as(u16, 0x3C00), down_s[0]);
    try std.testing.expectEqual(@as(u16, 0x3400), down_s[32]); // block_inject's own scale base
    try expectTensor(&file, "L0.ahc.up.w", .u32, &.{ 32, 6 });
    try std.testing.expectEqual(@as(u32, 5_000_000), std.mem.bytesAsSlice(u32, tensorBytes(&file, "L0.ahc.up.w"))[0]);
    // gdn in stack: qkv(24) z(4) b(2) a(2) rows, kw 6, one group per row so the tile is the identity
    try expectTensor(&file, "L0.gdn.in.wq", .u32, &.{ 32, 6 });
    const in_wq = std.mem.bytesAsSlice(u32, tensorBytes(&file, "L0.gdn.in.wq"));
    try std.testing.expectEqual(@as(u32, 1_000_000), in_wq[0]);
    try std.testing.expectEqual(@as(u32, 1_100_000), in_wq[24 * 6]); // z row 0 follows qkv row 23
    try std.testing.expectEqual(@as(u32, 1_300_000), in_wq[30 * 6]); // a row 0
    try expectTensor(&file, "L0.gdn.in.sbt", .bf16, &.{ 1, 32, 2 });
    const in_sbt = std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.gdn.in.sbt"));
    try std.testing.expectEqual(@as(u16, 0x3C00), in_sbt[0]); // qkv scale row 0
    try std.testing.expectEqual(@as(u16, 0x3B00), in_sbt[1]); // its bias
    try std.testing.expectEqual(@as(u16, 0x3F00), in_sbt[30 * 2]); // a's scale
    try expectTensor(&file, "L0.gdn.conv", .bf16, &.{ 32, 4 });
    try std.testing.expectEqual(@as(u16, bf16_half), std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.gdn.conv"))[0]);
    try expectTensor(&file, "L0.gdn.alog", .bf16, &.{8});
    const alog_bits = std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.gdn.alog"));
    try std.testing.expectEqual(@as(u16, 0x4000), alog_bits[0]); // 2.0, rounded from the f32 store
    try std.testing.expectEqual(@as(u16, 0x4040), alog_bits[1]); // 3.0
    try std.testing.expectEqual(@as(u16, 0x4110), alog_bits[7]); // 9.0
    try expectTensor(&file, "L0.gdn.dt", .bf16, &.{8});
    const dt_bits = std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.gdn.dt"));
    try std.testing.expectEqual(@as(u16, 0x3FC0), dt_bits[0]); // 1.5
    try std.testing.expectEqual(@as(u16, 0x4108), dt_bits[7]); // 8.5
    try expectTensor(&file, "L0.gdn.norm", .bf16, &.{32});
    // attention stack: q(32) k(8) v(8) index_qk(48) rows over three 32-row tiles, identity per group of one
    try expectTensor(&file, "L1.att.proj.wq", .u32, &.{ 96, 6 });
    const proj = std.mem.bytesAsSlice(u32, tensorBytes(&file, "L1.att.proj.wq"));
    try std.testing.expectEqual(@as(u32, 5_000_000), proj[0]);
    try std.testing.expectEqual(@as(u32, 5_100_000), proj[32 * 6]);
    try std.testing.expectEqual(@as(u32, 5_300_000), proj[48 * 6]);
    try expectTensor(&file, "L1.att.qn", .f32, &.{8});
    try expectTensor(&file, "L1.att.pool", .f32, &.{16});
    // router rows: the gate then the shared-expert gate; the shared row dequantizes code*scale+bias
    try expectTensor(&file, "L0.moe.router", .bf16, &.{ 5, 32 });
    const router = std.mem.bytesAsSlice(u16, tensorBytes(&file, "L0.moe.router"));
    try std.testing.expectEqual(@as(u16, 0x3F80), router[0]);
    try std.testing.expectEqual(@as(u16, 0x0000), router[4 * 32]); // code 0
    try std.testing.expectEqual(@as(u16, 0x4040), router[4 * 32 + 1]); // code 1 = 3.0
    try std.testing.expectEqual(@as(u16, 0x427C), router[4 * 32 + 5]); // code 5 = 63.0, across words 0 and 1
    try std.testing.expectEqual(@as(u16, 0x4040), router[4 * 32 + 12]); // code 12 = 3.0, in word 2
    // head: 128 rows over four tiles
    try expectTensor(&file, "head.wq", .u32, &.{ 128, 6 });
    const head = std.mem.bytesAsSlice(u32, tensorBytes(&file, "head.wq"));
    try std.testing.expectEqual(@as(u32, 500_000), head[0]);
    try std.testing.expectEqual(@as(u32, 500_000 + 127 * 6 + 5), head[127 * 6 + 5]);
    // ple: the lane over key then value, f32 widened conv, and the table starts
    try expectTensor(&file, "ple.kv.wq", .u32, &.{ 64, 6 });
    const kv = std.mem.bytesAsSlice(u32, tensorBytes(&file, "ple.kv.wq"));
    try std.testing.expectEqual(@as(u32, 7_000_000), kv[0]);
    try std.testing.expectEqual(@as(u32, 7_100_000), kv[32 * 6]);
    try expectTensor(&file, "ple.conv", .f32, &.{ 32, 4 });
    try std.testing.expectEqual(@as(u32, 0x3F000000), std.mem.bytesAsSlice(u32, tensorBytes(&file, "ple.conv"))[0]);
    try expectTensor(&file, "ple.starts", .u32, &.{8});
    const starts = std.mem.bytesAsSlice(u32, tensorBytes(&file, "ple.starts"));
    for ([_]u32{ 0, 3, 10, 21, 36, 55, 78, 105 }, 0..) |want, g| try std.testing.expectEqual(want, starts[g]);
    // the mixer keeps only input_mix_weight_down (no block inject)
    try expectTensor(&file, "mix.down.w", .u32, &.{ 32, 6 });
    try std.testing.expect(file.names.get("mix.down.w") != null and file.names.get("mix.block_inject.w") == null);

    const mlx_path = try tmpPath(a, tmp, "out/pack_mlx.safetensors");
    defer a.free(mlx_path);
    var mlx = try st.File.open(a, io, mlx_path);
    defer mlx.close(io);
    try expectTensor(&mlx, "L0.gdn.in.mw", .u32, &.{ 32, 6 });
    const mw = std.mem.bytesAsSlice(u32, tensorBytes(&mlx, "L0.gdn.in.mw"));
    try std.testing.expectEqual(@as(u32, 1_000_000), mw[0]);
    try std.testing.expectEqual(@as(u32, 1_100_000), mw[24 * 6]);
    try expectTensor(&mlx, "L0.gdn.in.ms", .bf16, &.{ 32, 1 });
    try expectTensor(&mlx, "ple.kv.mw", .u32, &.{ 64, 6 });

    const mtp_path = try tmpPath(a, tmp, "out/pack_mtp_mlx.safetensors");
    defer a.free(mtp_path);
    var mtp = try st.File.open(a, io, mtp_path);
    defer mtp.close(io);
    try expectTensor(&mtp, "mtp.att.proj.mw", .u32, &.{ 96, 6 });
    try std.testing.expectEqual(@as(u32, 15_000_000), std.mem.bytesAsSlice(u32, tensorBytes(&mtp, "mtp.att.proj.mw"))[0]);
    try expectTensor(&mtp, "mtp.fce.mw", .u32, &.{ 32, 6 });
    try expectTensor(&mtp, "mtp.fch.mw", .u32, &.{ 32, 6 });
}

test "the draft vocabulary builds the mtp decode tensors and gathers the head's rows" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeCheckpoint(tmp, a);
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    for (10..72) |id| try text.writer.print("{d}\n", .{id});
    try tmp.dir.writeFile(io, .{ .sub_path = "vocab.txt", .data = text.written() });
    const model_dir = try tmpPath(a, tmp, ".");
    defer a.free(model_dir);
    const out_dir = try tmpPath(a, tmp, "out");
    defer a.free(out_dir);
    try tmp.dir.createDirPath(io, "out");
    const vocab = try tmpPath(a, tmp, "vocab.txt");
    defer a.free(vocab);
    const report = try pack.build(a, io, model_dir, out_dir, vocab);
    try std.testing.expectEqual(@as(usize, 102), report.decode_tensors); // 63 + the 39 mtp decode tensors
    try std.testing.expectEqual(@as(usize, 64), report.draft_ids);
    const pack_path = try tmpPath(a, tmp, "out/pack.safetensors");
    defer a.free(pack_path);
    var file = try st.File.open(a, io, pack_path);
    defer file.close(io);
    try expectTensor(&file, "mtp.draft_ids", .u32, &.{64});
    const ids = std.mem.bytesAsSlice(u32, tensorBytes(&file, "mtp.draft_ids"));
    try std.testing.expectEqual(@as(u32, 0), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 10), ids[2]);
    try std.testing.expectEqual(@as(u32, 71), ids[63]);
    // the draft head gathers lm_head's rows by id, then tiles (identity at one group); sorted ids put the pads first
    try expectTensor(&file, "mtp.draft.wq", .u32, &.{ 64, 6 });
    const draft = std.mem.bytesAsSlice(u32, tensorBytes(&file, "mtp.draft.wq"));
    try std.testing.expectEqual(@as(u32, 500_000), draft[0]); // pad id 0's row
    try std.testing.expectEqual(@as(u32, 500_000 + 10 * 6), draft[2 * 6]); // id 10's first word
    try std.testing.expectEqual(@as(u32, 500_000 + 71 * 6 + 5), draft[63 * 6 + 5]);
    try expectTensor(&file, "mtp.draft.sbt", .bf16, &.{ 1, 64, 2 });
    try expectTensor(&file, "mtp.ahc.scale", .f32, &.{32});
    try expectTensor(&file, "mtp.enorm.scale", .f32, &.{32});
    try expectTensor(&file, "mtp.hnorm.scale", .f32, &.{32});
    try expectTensor(&file, "mtp.fce.wq", .u32, &.{ 32, 6 });
}
