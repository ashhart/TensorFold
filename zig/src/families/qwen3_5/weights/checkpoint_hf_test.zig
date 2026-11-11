//! Host tests of an HF BF16 checkpoint on small synthetic shards: names, dtypes, the conv layout, norms and files.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const table = @import("table.zig");
const projection = @import("projection.zig");
const convert = @import("core").quant.convert;

const io = std.testing.io;
const gpa = std.testing.allocator;

const M = "model.language_model.";

const config =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": false,
    \\ "text_config": {"hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 2,
    \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 1,
    \\  "linear_num_value_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 16,
    \\  "linear_conv_kernel_dim": 4, "vocab_size": 16, "full_attention_interval": 2, "mtp_num_hidden_layers": 0,
    \\  "layer_types": ["linear_attention", "full_attention"], "eos_token_id": 1}}
;

/// One synthetic safetensors file: tensors with name-seeded finite BF16 (or F32) values.
const File = struct {
    arena: std.heap.ArenaAllocator,
    names: std.ArrayList([]const u8) = .empty,
    header: std.ArrayList(u8) = .empty,
    data: std.ArrayList(u8) = .empty,

    fn init() File {
        return .{ .arena = .init(gpa) };
    }

    fn deinit(f: *File) void {
        f.arena.deinit();
    }

    fn put(f: *File, name: []const u8, dtype: []const u8, shape: []const usize) !void {
        const a = f.arena.allocator();
        var n: usize = 1;
        for (shape) |d| n *= d;
        var seed: u16 = 0;
        for (name) |c| seed +%= c;
        const start = f.data.items.len;
        for (0..n) |i| {
            // values in [0.25, 0.5): small, finite and different per tensor and element
            const v: f32 = 0.25 + @as(f32, @floatFromInt((seed +% @as(u16, @truncate(i * 7))) % 64)) / 256.0;
            if (std.mem.eql(u8, dtype, "F32")) {
                try f.data.appendSlice(a, &std.mem.toBytes(v));
            } else if (std.mem.eql(u8, dtype, "U8") or std.mem.eql(u8, dtype, "F8_E4M3")) {
                try f.data.append(a, @truncate(seed +% @as(u16, @truncate(i))));
            } else try f.data.appendSlice(a, &std.mem.toBytes(convert.bf16(v)));
        }
        try f.header.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (f.names.items.len > 0) "," else "", name, dtype });
        for (shape, 0..) |d, j| try f.header.print(a, "{s}{d}", .{ if (j > 0) "," else "", d });
        try f.header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ start, f.data.items.len });
        try f.names.append(a, try a.dupe(u8, name));
    }

    fn write(f: *File, dir: std.Io.Dir, path: []const u8) !void {
        const a = f.arena.allocator();
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u64, f.header.items.len + 2)));
        try out.append(a, '{');
        try out.appendSlice(a, f.header.items);
        try out.append(a, '}');
        try out.appendSlice(a, f.data.items);
        try dir.writeFile(io, .{ .sub_path = path, .data = out.items });
    }
};

/// What to plant: one text tensor as F32, the conv weight in another layout, or an index entry of a vision-only shard.
const Plant = struct { f32_tensor: ?[]const u8 = null, conv: []const usize = &.{ 96, 1, 4 }, vision_shard: bool = false };

/// Two shards and their index, a vision tensor among them, and a joint head file the index does not name.
fn write(dir: std.Io.Dir, plant: Plant) !void {
    var one = File.init();
    defer one.deinit();
    var two = File.init();
    defer two.deinit();
    const dt = struct {
        fn of(p: Plant, name: []const u8) []const u8 {
            return if (p.f32_tensor) |n| (if (std.mem.eql(u8, n, name)) "F32" else "BF16") else "BF16";
        }
    }.of;
    const first = [_]struct { []const u8, []const usize }{
        .{ M ++ "embed_tokens.weight", &.{ 16, 64 } },
        .{ M ++ "norm.weight", &.{64} },
        .{ M ++ "layers.0.input_layernorm.weight", &.{64} },
        .{ M ++ "layers.0.post_attention_layernorm.weight", &.{64} },
        .{ M ++ "layers.0.linear_attn.in_proj_qkv.weight", &.{ 96, 64 } },
        .{ M ++ "layers.0.linear_attn.in_proj_z.weight", &.{ 32, 64 } },
        .{ M ++ "layers.0.linear_attn.in_proj_a.weight", &.{ 2, 64 } },
        .{ M ++ "layers.0.linear_attn.in_proj_b.weight", &.{ 2, 64 } },
        .{ M ++ "layers.0.linear_attn.conv1d.weight", plant.conv },
        .{ M ++ "layers.0.linear_attn.A_log", &.{2} },
        .{ M ++ "layers.0.linear_attn.dt_bias", &.{2} },
        .{ M ++ "layers.0.linear_attn.norm.weight", &.{16} },
        .{ M ++ "layers.0.linear_attn.out_proj.weight", &.{ 64, 32 } },
        .{ "model.visual.patch_embed.proj.weight", &.{ 8, 8 } },
    };
    for (first) |t| try one.put(t[0], dt(plant, t[0]), t[1]);
    const second = [_]struct { []const u8, []const usize }{
        .{ M ++ "layers.0.mlp.gate_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.0.mlp.up_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.0.mlp.down_proj.weight", &.{ 64, 32 } },
        .{ M ++ "layers.1.input_layernorm.weight", &.{64} },
        .{ M ++ "layers.1.post_attention_layernorm.weight", &.{64} },
        .{ M ++ "layers.1.self_attn.q_proj.weight", &.{ 128, 64 } },
        .{ M ++ "layers.1.self_attn.k_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.1.self_attn.v_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.1.self_attn.o_proj.weight", &.{ 64, 64 } },
        .{ M ++ "layers.1.self_attn.q_norm.weight", &.{32} },
        .{ M ++ "layers.1.self_attn.k_norm.weight", &.{32} },
        .{ M ++ "layers.1.mlp.gate_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.1.mlp.up_proj.weight", &.{ 32, 64 } },
        .{ M ++ "layers.1.mlp.down_proj.weight", &.{ 64, 32 } },
        .{ "lm_head.weight", &.{ 16, 64 } },
    };
    for (second) |t| try two.put(t[0], dt(plant, t[0]), t[1]);
    try one.write(dir, "model-00001-of-00002.safetensors");
    try two.write(dir, "model-00002-of-00002.safetensors");
    var head = File.init();
    defer head.deinit();
    try head.put("joint.w", "BF16", &.{ 4, 4 });
    try head.write(dir, "joint_head.safetensors");
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(gpa);
    try index.appendSlice(gpa, "{\"metadata\": {}, \"weight_map\": {");
    for (one.names.items, 0..) |n, i| try index.print(gpa, "{s}\"{s}\": \"model-00001-of-00002.safetensors\"", .{ if (i > 0) "," else "", n });
    for (two.names.items) |n| try index.print(gpa, ",\"{s}\": \"model-00002-of-00002.safetensors\"", .{n});
    // a shard the index names for vision tensors only, absent from the disk
    if (plant.vision_shard) try index.appendSlice(gpa, ",\"model.visual.blocks.0.attn.qkv.weight\": \"model-vision.safetensors\"");
    try index.appendSlice(gpa, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
}

fn rootOf(tmp: std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}

fn tensorOf(ck: *const checkpoint.Checkpoint, name: []const u8) ![]const u8 {
    return (try ck.main.get(name)).bytes;
}

test "HF names: the embedding, head and layers read as BF16, the files from the index" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp.dir, .{});
    const root = try rootOf(tmp);
    defer gpa.free(root);
    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    defer ck.close();
    try std.testing.expectEqualStrings(M, ck.names.text);
    try std.testing.expectEqualStrings("lm_head", ck.names.head);
    try std.testing.expectEqual(@as(usize, 2), ck.files.model.len);
    for (ck.files.model) |p| try std.testing.expect(std.mem.indexOf(u8, p, "joint") == null);
    try std.testing.expect(!ck.main.has("joint.w"));
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const a = scratch.allocator();
    const embed = try ck.embed(a);
    try std.testing.expectEqualSlices(u8, try tensorOf(&ck, M ++ "embed_tokens.weight"), embed.bf16.weight.bytes);
    const head = (try ck.head(a)).?;
    try std.testing.expectEqualSlices(u8, try tensorOf(&ck, "lm_head.weight"), head.bf16.weight.bytes);
    var l1 = try ck.layer(1);
    defer l1.deinit();
    try std.testing.expectEqualSlices(u8, try tensorOf(&ck, M ++ "layers.1.self_attn.q_proj.weight"), l1.body.full.q.bf16.weight.bytes);
    try std.testing.expectEqual(@as(usize, 32), l1.body.full.mlp.dense.gate.rows());
    try std.testing.expect(try ck.mtp() == null);
}

test "1 + w on the input, post, final, q and k norms; linear_attn.norm and the conv as stored" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp.dir, .{});
    const root = try rootOf(tmp);
    defer gpa.free(root);
    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    defer ck.close();
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    var l0 = try ck.layer(0);
    defer l0.deinit();
    var l1 = try ck.layer(1);
    defer l1.deinit();
    const x = l0.body.linear;
    const f = l1.body.full;
    const centred = [_]struct { []const u8, []const u8 }{
        .{ M ++ "norm.weight", (try ck.finalNorm(scratch.allocator())).bytes },
        .{ M ++ "layers.0.input_layernorm.weight", x.input_norm.bytes },
        .{ M ++ "layers.0.post_attention_layernorm.weight", x.post_norm.bytes },
        .{ M ++ "layers.1.input_layernorm.weight", f.input_norm.bytes },
        .{ M ++ "layers.1.post_attention_layernorm.weight", f.post_norm.bytes },
        .{ M ++ "layers.1.self_attn.q_norm.weight", f.q_norm.bytes },
        .{ M ++ "layers.1.self_attn.k_norm.weight", f.k_norm.bytes },
    };
    for (centred) |c| {
        const stored = try tensorOf(&ck, c[0]);
        for (0..stored.len / 2) |i| try std.testing.expectEqual(1.0 + convert.load(.bf16, stored, i), convert.load(.f32, c[1], i));
    }
    const gnorm = try tensorOf(&ck, M ++ "layers.0.linear_attn.norm.weight");
    for (0..16) |i| try std.testing.expectEqual(convert.load(.bf16, gnorm, i), convert.load(.f32, x.gnorm.bytes, i));
    // conv1d (96, 1, 4) read as (96, 4): the same element order, widened to fp32
    try std.testing.expectEqual(@as(u8, 2), x.conv.rank);
    try std.testing.expectEqual(@as(usize, 96), x.conv.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), x.conv.shape[1]);
    const conv = try tensorOf(&ck, M ++ "layers.0.linear_attn.conv1d.weight");
    for (0..96 * 4) |i| try std.testing.expectEqual(convert.load(.bf16, conv, i), convert.load(.f32, x.conv.bytes, i));
}

test "a bf16 conv weight of rank 2 or 4, or with its 1 elsewhere, is refused" {
    for ([_][]const usize{ &.{ 96, 4 }, &.{ 96, 1, 4, 1 }, &.{ 96, 4, 1 } }) |shape| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try write(tmp.dir, .{ .conv = shape });
        const root = try rootOf(tmp);
        defer gpa.free(root);
        var ck = try checkpoint.Checkpoint.open(gpa, io, root);
        defer ck.close();
        try std.testing.expect(projection.squeezed(try ck.main.get(M ++ "layers.0.linear_attn.conv1d.weight")) == null);
    }
}

/// The shards of `root` mapped as a bf16 checkpoint's table, without the checkpoint's checks.
fn rawTable(root: []const u8, files: *table.Files) !table.Table {
    files.* = try table.Files.indexed(gpa, io, root);
    errdefer files.deinit();
    return table.Table.open(gpa, io, files.model, "", .{ .bf16 = {} });
}

test "a text tensor or the head in another dtype is named for refusal; the vision tower is not checked" {
    const names = [_][]const u8{ M ++ "layers.1.self_attn.k_norm.weight", "lm_head.weight", M ++ "layers.0.mlp.up_proj.weight", "model.visual.patch_embed.proj.weight" };
    for (names, 0..) |name, i| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try write(tmp.dir, .{ .f32_tensor = name });
        const root = try rootOf(tmp);
        defer gpa.free(root);
        var files: table.Files = undefined;
        var t = try rawTable(root, &files);
        defer files.deinit();
        defer t.close(io);
        const got = checkpoint.mixed(&t, checkpoint.Names.of(&t));
        if (i < 3) try std.testing.expectEqualStrings(name, got.?) else try std.testing.expect(got == null);
    }
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp.dir, .{ .f32_tensor = "model.visual.patch_embed.proj.weight" });
    const root = try rootOf(tmp);
    defer gpa.free(root);
    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    ck.close();
}

test "textBytes counts the text tower and the head, not the vision tower or the joint head" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp.dir, .{});
    const root = try rootOf(tmp);
    defer gpa.free(root);
    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    defer ck.close();
    var want: u64 = 0;
    var it = ck.main.keys();
    while (it.next()) |k| if (std.mem.startsWith(u8, k.*, M) or std.mem.eql(u8, k.*, "lm_head.weight")) {
        want += (try ck.main.get(k.*)).bytes.len;
    };
    try std.testing.expectEqual(want, try table.textBytes(gpa, io, root));
    try std.testing.expect(want < (try ck.main.get("model.visual.patch_embed.proj.weight")).bytes.len + want);
}

test "names and format go together: bf16 with HF's names, a quantized checkpoint with MLX's" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var one = File.init();
    defer one.deinit();
    try one.put(checkpoint.prefix ++ "embed_tokens.weight", "BF16", &.{ 16, 64 });
    try one.write(tmp.dir, "model.safetensors");
    const root = try rootOf(tmp);
    defer gpa.free(root);
    // no index: the one `model.safetensors`
    var files: table.Files = undefined;
    var t = try rawTable(root, &files);
    defer files.deinit();
    defer t.close(io);
    const names = checkpoint.Names.of(&t);
    try std.testing.expectEqualStrings(checkpoint.prefix, names.text);
    try std.testing.expect(names.refusal(true) != null);
    try std.testing.expect(names.refusal(false) == null);
    try std.testing.expect(checkpoint.Names.hf.refusal(true) == null);
    try std.testing.expect(checkpoint.Names.hf.refusal(false) != null);
}

test "a shard the index names only for vision tensors is not opened, even when it is absent" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(tmp.dir, .{ .vision_shard = true });
    const root = try rootOf(tmp);
    defer gpa.free(root);
    var files = try table.Files.indexed(gpa, io, root);
    defer files.deinit();
    try std.testing.expectEqual(@as(usize, 2), files.model.len);
    for (files.model) |p| try std.testing.expect(std.mem.indexOf(u8, p, "vision") == null);
    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    ck.close();
    try std.testing.expect(try table.textBytes(gpa, io, root) > 0);
}

test "device bytes count a ModelOpt checkpoint's widened tables, not its stored bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var f = File.init();
    defer f.deinit();
    try f.put(M ++ "layers.0.mlp.up_proj.weight", "U8", &.{ 8, 16 });
    try f.put(M ++ "layers.0.mlp.up_proj.weight_scale", "F8_E4M3", &.{ 8, 2 });
    try f.put(M ++ "layers.0.mlp.up_proj.weight_scale_2", "F32", &.{1});
    try f.put(M ++ "embed_tokens.weight", "BF16", &.{ 8, 4 });
    try f.put("model.visual.patch_embed.proj.weight", "BF16", &.{ 8, 8 });
    try f.write(tmp.dir, "model.safetensors");
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(gpa);
    try index.appendSlice(gpa, "{\"metadata\": {}, \"weight_map\": {");
    for (f.names.items, 0..) |n, i| try index.print(gpa, "{s}\"{s}\": \"model.safetensors\"", .{ if (i > 0) "," else "", n });
    try index.appendSlice(gpa, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    const root = try rootOf(tmp);
    defer gpa.free(root);
    // stored: codes 128, E4M3 table 16, global 4, text 64; the table doubles to 32 on the device, the tower stays out
    try std.testing.expectEqual(@as(u64, 228), try table.deviceBytes(gpa, io, root));
}

test "a ModelOpt qwen3_5_moe checkpoint is refused by name before its shards are read" {
    // the same MoE shape config.zig's own test parses, with ModelOpt's quantization block beside it
    const text =
        \\{"model_type": "qwen3_5_moe",
        \\ "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 2,
        \\ "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        \\ "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 1000, "rope_theta": 5000000,
        \\ "partial_rotary_factor": 0.5, "num_experts": 256, "num_experts_per_tok": 8,
        \\ "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        \\ "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    ;
    const cfg = try qcfg.Config.parse(gpa, text, null);
    defer cfg.deinit();
    try std.testing.expect(cfg.modelopt and cfg.spec.experts > 0);
    // before this refusal the same checkpoint opened and its stack reached experts.routed, which is not read
    try std.testing.expectError(error.UnsupportedQuantization, checkpoint.refuseModelOptExperts(&cfg));
}
