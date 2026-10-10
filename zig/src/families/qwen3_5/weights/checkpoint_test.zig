//! Host tests of the checkpoint index on small synthetic safetensors files, checked against the Python loader's rules.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const convert = @import("core").quant.convert;
const host = @import("host.zig");
const DType = @import("table.zig").DType;

const io = std.testing.io;
const gpa = std.testing.allocator;

/// One tensor of a synthetic file: `bytes` (or `size` bytes of a name-seeded pattern).
const Entry = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8 = &.{} };

const Image = struct {
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(Entry) = .empty,

    fn init() Image {
        return .{ .arena = .init(gpa) };
    }

    fn deinit(m: *Image) void {
        m.arena.deinit();
    }

    fn elems(shape: []const usize) usize {
        var n: usize = 1;
        for (shape) |d| n *= d;
        return n;
    }

    /// A tensor whose bytes are a pattern seeded by its name (`bits` per element for the size).
    fn put(m: *Image, name: []const u8, dtype: []const u8, shape: []const usize) !void {
        const a = m.arena.allocator();
        const size: usize = if (std.mem.eql(u8, dtype, "BF16")) 2 else 4;
        const bytes = try a.alloc(u8, elems(shape) * size);
        var seed: u8 = 0;
        for (name) |c| seed +%= c;
        for (bytes, 0..) |*b, i| b.* = seed +% @as(u8, @truncate(i * 7));
        // bf16 patterns stay finite: a high byte under 0x40 is a small positive number
        if (size == 2) for (bytes[1..], 0..) |*b, i| {
            if (i % 2 == 0) b.* &= 0x3F;
        };
        try m.entries.append(a, .{ .name = try a.dupe(u8, name), .dtype = dtype, .shape = try a.dupe(usize, shape), .bytes = bytes });
    }

    /// An affine projection [n, k] at `bits`: words, scales and biases.
    fn affine(m: *Image, name: []const u8, n: usize, k: usize, bits: usize, group: usize) !void {
        var buf: [160]u8 = undefined;
        try m.put(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), "U32", &.{ n, k * bits / 32 });
        try m.put(try std.fmt.bufPrint(&buf, "{s}.scales", .{name}), "BF16", &.{ n, k / group });
        try m.put(try std.fmt.bufPrint(&buf, "{s}.biases", .{name}), "BF16", &.{ n, k / group });
    }

    fn find(m: *const Image, name: []const u8) []const u8 {
        for (m.entries.items) |e| if (std.mem.eql(u8, e.name, name)) return e.bytes;
        unreachable;
    }

    fn write(m: *Image, dir: std.Io.Dir, file: []const u8) !void {
        const a = m.arena.allocator();
        var header: std.ArrayList(u8) = .empty;
        var data: std.ArrayList(u8) = .empty;
        try header.append(a, '{');
        for (m.entries.items, 0..) |e, i| {
            const start = data.items.len;
            try data.appendSlice(a, e.bytes);
            try header.print(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i > 0) "," else "", e.name, e.dtype });
            for (e.shape, 0..) |d, j| try header.print(a, "{s}{d}", .{ if (j > 0) "," else "", d });
            try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ start, data.items.len });
        }
        try header.append(a, '}');
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u64, header.items.len)));
        try out.appendSlice(a, header.items);
        try out.appendSlice(a, data.items);
        try dir.writeFile(io, .{ .sub_path = file, .data = out.items });
    }
};

const dense_config =
    \\{"model_type": "qwen3_5", "tie_word_embeddings": true,
    \\ "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
    \\ "text_config": {"hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 2,
    \\  "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 1,
    \\  "linear_num_value_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 16,
    \\  "linear_conv_kernel_dim": 4, "vocab_size": 16, "full_attention_interval": 2,
    \\  "layer_types": ["linear_attention", "full_attention"]}}
;

const moe_config =
    \\{"model_type": "qwen3_5_moe", "tie_word_embeddings": true,
    \\ "quantization": {"group_size": 32, "bits": 4, "mode": "affine",
    \\   "language_model.model.layers.0.mlp.gate": {"group_size": 32, "bits": 8},
    \\   "language_model.model.layers.0.mlp.shared_expert_gate": {"group_size": 32, "bits": 8}},
    \\ "text_config": {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
    \\  "num_key_value_heads": 1, "head_dim": 32, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    \\  "linear_key_head_dim": 32, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "vocab_size": 16,
    \\  "full_attention_interval": 1, "num_experts": 2, "num_experts_per_tok": 1, "moe_intermediate_size": 32,
    \\  "shared_expert_intermediate_size": 32}}
;

const L = "language_model.model.";

fn attention(m: *Image, base: []const u8) !void {
    var buf: [160]u8 = undefined;
    const shapes = [_]struct { []const u8, usize }{ .{ "q_proj", 64 }, .{ "k_proj", 32 }, .{ "v_proj", 32 }, .{ "o_proj", 64 } };
    for (shapes) |p| try m.affine(try std.fmt.bufPrint(&buf, "{s}self_attn.{s}", .{ base, p[0] }), p[1], 64, 4, 32);
    try m.put(try std.fmt.bufPrint(&buf, "{s}self_attn.q_norm.weight", .{base}), "BF16", &.{32});
    try m.put(try std.fmt.bufPrint(&buf, "{s}self_attn.k_norm.weight", .{base}), "BF16", &.{32});
}

fn norms(m: *Image, base: []const u8) !void {
    var buf: [160]u8 = undefined;
    try m.put(try std.fmt.bufPrint(&buf, "{s}input_layernorm.weight", .{base}), "BF16", &.{64});
    try m.put(try std.fmt.bufPrint(&buf, "{s}post_attention_layernorm.weight", .{base}), "BF16", &.{64});
}

fn trunk(m: *Image) !void {
    try m.affine(L ++ "embed_tokens", 16, 64, 4, 32);
    try m.put(L ++ "norm.weight", "BF16", &.{64});
}

test "a dense model: layers, embedding, tied head and the MTP side file" {
    var img = Image.init();
    defer img.deinit();
    var side = Image.init();
    defer side.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try trunk(&img);
    try norms(&img, L ++ "layers.0.");
    const lin = L ++ "layers.0.linear_attn.";
    try img.affine(lin ++ "in_proj_qkv", 96, 64, 4, 32);
    try img.affine(lin ++ "in_proj_z", 32, 64, 4, 32);
    try img.affine(lin ++ "in_proj_a", 2, 64, 4, 32);
    try img.affine(lin ++ "in_proj_b", 2, 64, 4, 32);
    try img.put(lin ++ "conv1d.weight", "BF16", &.{ 96, 4, 1 });
    try img.put(lin ++ "A_log", "BF16", &.{2});
    try img.put(lin ++ "dt_bias", "BF16", &.{2});
    try img.put(lin ++ "norm.weight", "BF16", &.{16});
    try img.affine(lin ++ "out_proj", 64, 32, 4, 32);
    inline for (.{ "layers.0.", "layers.1." }) |l| {
        try img.affine(L ++ l ++ "mlp.gate_proj", 32, 64, 4, 32);
        try img.affine(L ++ l ++ "mlp.up_proj", 32, 64, 4, 32);
        try img.affine(L ++ l ++ "mlp.down_proj", 64, 32, 4, 32);
    }
    try norms(&img, L ++ "layers.1.");
    try attention(&img, L ++ "layers.1.");
    try img.put("vision_tower.patch_embed.weight", "BF16", &.{4});

    try side.put("mtp.pre_fc_norm_embedding.weight", "BF16", &.{64});
    try side.put("mtp.pre_fc_norm_hidden.weight", "BF16", &.{64});
    try side.put("mtp.norm.weight", "BF16", &.{64});
    try side.affine("mtp.fc", 64, 128, 4, 32);
    try norms(&side, "mtp.layers.0.");
    try attention(&side, "mtp.layers.0.");
    try side.affine("mtp.layers.0.mlp.gate_proj", 32, 64, 4, 32);
    try side.affine("mtp.layers.0.mlp.up_proj", 32, 64, 4, 32);
    try side.affine("mtp.layers.0.mlp.down_proj", 64, 32, 4, 32);

    try img.write(tmp.dir, "model.safetensors");
    try side.write(tmp.dir, "mtp.safetensors");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = dense_config });
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(root);

    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    defer ck.close();
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const embed = try ck.embed(scratch.allocator());
    try std.testing.expectEqual(@as(u8, 4), embed.mlx.bits);
    try std.testing.expectEqual(host.Tensor{ .dtype = .i32, .rank = 2, .shape = .{ 16, 8, 1, 1, 1 }, .bytes = embed.mlx.words.bytes }, embed.mlx.words);
    try std.testing.expect(try ck.head(scratch.allocator()) == null);
    const final = try ck.finalNorm(scratch.allocator());
    try std.testing.expectEqual(convert.load(.bf16, img.find(L ++ "norm.weight"), 5), convert.load(.f32, final.bytes, 5));

    var l0 = try ck.layer(0);
    defer l0.deinit();
    const x = l0.body.linear;
    // the conv weight loses its trailing 1 and is widened to fp32
    try std.testing.expectEqual(@as(u8, 2), x.conv.rank);
    try std.testing.expectEqual(DType.f32, x.conv.dtype);
    try std.testing.expectEqual(@as(usize, 4), x.conv.shape[1]);
    try std.testing.expectEqual(convert.load(.bf16, img.find(lin ++ "conv1d.weight"), 9), convert.load(.f32, x.conv.bytes, 9));
    try std.testing.expectEqual(@as(usize, 96), x.qkv.mlx.words.shape[0]);
    try std.testing.expectEqualSlices(u8, img.find(lin ++ "in_proj_qkv.weight"), x.qkv.mlx.words.bytes);
    try std.testing.expectEqual(DType.bf16, x.qkv.mlx.scales.dtype);
    try std.testing.expectEqual(@as(usize, 16), x.gnorm.numel());
    try std.testing.expect(x.mlp == .dense);

    var l1 = try ck.layer(1);
    defer l1.deinit();
    try std.testing.expectEqual(@as(usize, 64), l1.body.full.q.rows());
    try std.testing.expectError(error.InvalidModel, ck.layer(2));

    var head = (try ck.mtp()).?;
    defer head.deinit();
    try std.testing.expect(head.gated and head.head == null and head.mlp.? == .dense);
    // fc [64, 128] splits into two [64, 64] halves, row by row, words and tables on the same boundary
    const words = side.find("mtp.fc.weight");
    const scales = side.find("mtp.fc.scales");
    for ([_]usize{ 0, 63 }) |r| {
        try std.testing.expectEqualSlices(u8, words[r * 64 ..][0..32], head.fc_e.mlx.words.bytes[r * 32 ..][0..32]);
        try std.testing.expectEqualSlices(u8, words[r * 64 + 32 ..][0..32], head.fc_h.mlx.words.bytes[r * 32 ..][0..32]);
        try std.testing.expectEqualSlices(u8, scales[r * 8 + 4 ..][0..4], head.fc_h.mlx.scales.bytes[r * 4 ..][0..4]);
    }
}

test "MoE experts: gate and up fused per expert, the shared expert last, router rows with the shared gate last" {
    var img = Image.init();
    defer img.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base = L ++ "layers.0.";
    const mlp = base ++ "mlp.";
    try trunk(&img);
    try norms(&img, base);
    try attention(&img, base);
    try img.affine(mlp ++ "gate", 2, 64, 8, 32);
    try img.affine(mlp ++ "shared_expert_gate", 1, 64, 8, 32);
    for ([_][]const u8{ "gate_proj", "up_proj" }) |p| {
        var buf: [160]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
            const small = std.mem.eql(u8, part, "weight");
            try img.put(try std.fmt.bufPrint(&buf, "{s}switch_mlp.{s}.{s}", .{ mlp, p, part }), if (small) "U32" else "BF16", if (small) &.{ 2, 32, 8 } else &.{ 2, 32, 2 });
            try img.put(try std.fmt.bufPrint(&buf, "{s}shared_expert.{s}.{s}", .{ mlp, p, part }), if (small) "U32" else "BF16", if (small) &.{ 32, 8 } else &.{ 32, 2 });
        }
    }
    for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
        var buf: [160]u8 = undefined;
        const small = std.mem.eql(u8, part, "weight");
        try img.put(try std.fmt.bufPrint(&buf, "{s}switch_mlp.down_proj.{s}", .{ mlp, part }), if (small) "U32" else "BF16", if (small) &.{ 2, 64, 4 } else &.{ 2, 64, 1 });
        // the shared expert may carry a leading 1
        try img.put(try std.fmt.bufPrint(&buf, "{s}shared_expert.down_proj.{s}", .{ mlp, part }), if (small) "U32" else "BF16", if (small) &.{ 1, 64, 4 } else &.{ 1, 64, 1 });
    }
    try img.write(tmp.dir, "model.safetensors");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = moe_config });
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(root);

    var ck = try checkpoint.Checkpoint.open(gpa, io, root);
    defer ck.close();
    try std.testing.expect(try ck.mtp() == null);
    var layer = try ck.layer(0);
    defer layer.deinit();
    const r = layer.body.full.mlp.routed;
    const e = r.experts;
    try std.testing.expect(e.gated and e.count == 3 and e.width == 32 and e.dims == 64);
    try std.testing.expectEqual(@as(usize, 3), e.fused.mlx.words.shape[0]);
    try std.testing.expectEqual(@as(usize, 64), e.fused.mlx.words.shape[1]);
    const per = 32 * 8 * 4;
    const gate = img.find(mlp ++ "switch_mlp.gate_proj.weight");
    const up = img.find(mlp ++ "switch_mlp.up_proj.weight");
    const fused = e.fused.mlx.words.bytes;
    try std.testing.expectEqualSlices(u8, gate[per .. 2 * per], fused[(2 * per) * 1 ..][0..per]);
    try std.testing.expectEqualSlices(u8, up[per .. 2 * per], fused[(2 * per) * 1 + per ..][0..per]);
    try std.testing.expectEqualSlices(u8, img.find(mlp ++ "shared_expert.gate_proj.weight"), fused[2 * 2 * per ..][0..per]);
    try std.testing.expectEqualSlices(u8, img.find(mlp ++ "shared_expert.up_proj.weight"), fused[2 * 2 * per + per ..][0..per]);
    try std.testing.expectEqualSlices(u8, img.find(mlp ++ "shared_expert.down_proj.scales"), e.down.mlx.scales.bytes[2 * 64 * 2 ..][0 .. 64 * 2]);
    // the router: 8-bit codes unpacked as s * q + b, rounded to bf16, the shared gate row last
    try std.testing.expectEqualSlices(usize, &.{ 3, 64 }, r.router.shape[0..2]);
    const words = img.find(mlp ++ "gate.weight");
    const scales = img.find(mlp ++ "gate.scales");
    const biases = img.find(mlp ++ "gate.biases");
    for ([_]usize{ 0, 7, 70, 127 }) |i| {
        const row = i / 64;
        const code: f32 = @floatFromInt(words[i]);
        const g = row * 2 + (i % 64) / 32;
        const want = convert.bf16(code * convert.load(.bf16, scales, g) + convert.load(.bf16, biases, g));
        try std.testing.expectEqual(want, std.mem.readInt(u16, r.router.bytes[2 * i ..][0..2], .little));
        try std.testing.expectEqual(@as(u32, want) << 16, std.mem.readInt(u32, r.rows32.bytes[4 * i ..][0..4], .little));
    }
    const shared_words = img.find(mlp ++ "shared_expert_gate.weight");
    const last: f32 = @floatFromInt(shared_words[5]);
    const sg = 0;
    try std.testing.expectEqual(
        convert.bf16(last * convert.load(.bf16, img.find(mlp ++ "shared_expert_gate.scales"), sg) + convert.load(.bf16, img.find(mlp ++ "shared_expert_gate.biases"), sg)),
        std.mem.readInt(u16, r.router.bytes[2 * (128 + 5) ..][0..2], .little),
    );
}
