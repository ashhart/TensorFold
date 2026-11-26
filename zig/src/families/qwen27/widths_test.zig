const std = @import("std");
const cfg = @import("config.zig");
const affine = @import("affine.zig");
const widths = @import("widths.zig");
const Width = widths.Width;
const Widths = widths.Widths;
const Chip = widths.Chip;
const Why = widths.Why;
const Tfq3Lookup = widths.Tfq3Lookup;
const uniform = widths.uniform;
const bf16 = widths.bf16;
const tfq3_group = widths.tfq3_group;
const runs = widths.runs;
const costShape = widths.costShape;
const admit = widths.admit;
const admitTfq3 = widths.admitTfq3;

const all_rows = [_]u8{ 2, 3, 4, 5, 6, 8 };

fn testConfig(a: std.mem.Allocator, quantization: []const u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5","text_config":{{"model_type":"qwen3_5_text","hidden_size":5120,"intermediate_size":17408,
        \\"num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}},"quantization":{s}}}
    , .{quantization});
}

fn testAdmit(quantization: []const u8, chip: Chip, why: *Why) !void {
    const a = std.testing.allocator;
    const text = try testConfig(a, quantization);
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    return admit(&w, c, chip, why);
}

fn testAdmitRouted(quantization: []const u8, why: *Why) !void {
    const a = std.testing.allocator;
    const text = try std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5_moe","text_config":{{"model_type":"qwen3_5_moe_text","hidden_size":3072,"intermediate_size":1024,
        \\"moe_intermediate_size":1024,"shared_expert_intermediate_size":1024,"num_experts":256,"num_experts_per_tok":8,
        \\"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":64,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}},"quantization":{s}}}
    , .{quantization});
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    return admit(&w, c, .{ .tensor_units = true, .row_bits = &all_rows }, why);
}

/// Qwen3.6-35B-A3B's shapes (no intermediate_size) at `quantization`, on a chip with or without tensor units.
fn testAdmitA3b(quantization: []const u8, units: bool, why: *Why) !void {
    const a = std.testing.allocator;
    const text = try std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5_moe","text_config":{{"model_type":"qwen3_5_moe_text","hidden_size":2048,
        \\"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_experts":256,"num_experts_per_tok":8,
        \\"num_hidden_layers":40,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}},"quantization":{s}}}
    , .{quantization});
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    return admit(&w, c, .{ .tensor_units = units, .row_bits = &all_rows }, why);
}

/// TensorFold/Qwen3.8-27B-oQ4 override shapes: 4-bit g64, 5-bit GDN, MLP and attention members, one 6-bit v.
const oq4 =
    \\{"group_size":64,"bits":4,"mode":"affine",
    \\"language_model.model.layers.37.linear_attn.in_proj_a":{"group_size":64,"bits":5},
    \\"language_model.model.layers.36.linear_attn.in_proj_b":{"group_size":64,"bits":5},
    \\"language_model.model.layers.26.linear_attn.in_proj_z":{"group_size":64,"bits":5},
    \\"language_model.model.layers.62.linear_attn.out_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.25.mlp.down_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.k_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.o_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.27.self_attn.v_proj":{"group_size":64,"bits":6}}
;

/// The override shapes of TensorFold/Qwen3.8-27B-oQ2: 2-bit g64 with an 8-bit embedding and 3/5/6-bit members.
const oq2 =
    \\{"group_size":64,"bits":2,"mode":"affine",
    \\"language_model.model.embed_tokens":{"group_size":64,"bits":8},
    \\"language_model.model.layers.50.linear_attn.in_proj_a":{"group_size":64,"bits":3},
    \\"language_model.model.layers.44.linear_attn.in_proj_a":{"group_size":64,"bits":5},
    \\"language_model.model.layers.42.linear_attn.in_proj_b":{"group_size":64,"bits":5},
    \\"language_model.model.layers.12.linear_attn.in_proj_z":{"group_size":64,"bits":5},
    \\"language_model.model.layers.10.mlp.down_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.35.self_attn.k_proj":{"group_size":64,"bits":5},
    \\"language_model.model.layers.35.self_attn.v_proj":{"group_size":64,"bits":6}}
;

test "uniform 4-bit g64 admits on both chip kinds and ignores draft-head and vision overrides" {
    var why: Why = .{};
    const q = "{\"bits\":4,\"group_size\":64,\"mtp.layers.0.mlp.gate_proj\":{\"bits\":8,\"group_size\":64},\"vision_tower.blocks.0.attn.qkv\":false}";
    try testAdmit(q, .{ .tensor_units = false }, &why);
    try testAdmit(q, .{ .tensor_units = true }, &why);
}

test "oQ4 and oQ2 formats admit once the row kernels unpack every width they declare" {
    var why: Why = .{};
    try testAdmit(oq4, .{ .tensor_units = false, .row_bits = &all_rows }, &why);
    try testAdmit(oq2, .{ .tensor_units = false, .row_bits = &all_rows }, &why);
}

test "on tensor units a run that is not whole 32-row tiles takes the row kernels: oQ4's 48-row in_proj_a" {
    var why: Why = .{};
    try testAdmit(oq4, .{ .tensor_units = true }, &why);
    try testAdmit(oq2, .{ .tensor_units = true }, &why);
}

test "a width the row kernels do not unpack refuses naming the module, never guessing four bits" {
    var why: Why = .{};
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit(oq4, .{ .tensor_units = false, .row_bits = &.{ 2, 4, 6, 8 } }, &why));
    try std.testing.expectEqual(@as(?usize, 25), why.layer);
    try std.testing.expectEqualStrings("mlp.down_proj", why.module);
    try std.testing.expectEqual(Width{ .bits = 5, .group = 64 }, why.width);
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit(oq2, .{ .tensor_units = false, .row_bits = &.{ 2, 4, 6, 8 } }, &why));
    try std.testing.expectEqual(@as(u8, 5), why.width.bits);
}

test "a dense projection and an unsupported global group refuse" {
    var why: Why = .{};
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmit("{\"bits\":4,\"group_size\":64,\"model.layers.3.self_attn.q_proj\":false}", .{ .tensor_units = false }, &why));
    try std.testing.expectError(error.BadAffine, testAdmit("{\"bits\":4,\"group_size\":16}", .{ .tensor_units = false }, &why));
}

test "a mixed stack runs every member alone, even beside a member of its own width" {
    var buf: [3][2]usize = undefined;
    const four = uniform;
    const five = Width{ .bits = 5, .group = 64 };
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 3 }}, runs(&.{ four, four, four }, &buf));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 } }, runs(&.{ five, four, four }, &buf));
    try std.testing.expectEqualSlices([2]usize, &.{ .{ 0, 1 }, .{ 1, 1 }, .{ 2, 1 } }, runs(&.{ four, five, four }, &buf));
}

test "timed costs keep a uniform checkpoint's shape and give each mixed one its own" {
    const a = std.testing.allocator;
    var tags: [3]u64 = undefined;
    for ([_][]const u8{ "{\"bits\":4,\"group_size\":64}", oq4, oq2 }, &tags) |quantization, *t| {
        const text = try testConfig(a, quantization);
        defer a.free(text);
        const c = try cfg.parse(a, text, null);
        var f = try affine.Formats.init(a, text);
        defer f.deinit();
        var w = try Widths.resolve(a, &f, c);
        defer w.deinit();
        t.* = w.tag();
        var buf: [96]u8 = undefined;
        const got = try costShape(c, t.*, &buf, 16, true);
        try std.testing.expectEqual(t.* == 0, std.mem.eql(u8, got, "64/5120/17408/248320/16/1"));
    }
    try std.testing.expectEqual(@as(u64, 0), tags[0]);
    try std.testing.expect(tags[1] != 0 and tags[2] != 0 and tags[1] != tags[2]);
}

test "resolved widths follow overrides under any checkpoint prefix" {
    const a = std.testing.allocator;
    const text = try testConfig(a, oq4);
    defer a.free(text);
    const c = try cfg.parse(a, text, null);
    var f = try affine.Formats.init(a, text);
    defer f.deinit();
    var w = try Widths.resolve(a, &f, c);
    defer w.deinit();
    try std.testing.expectEqual(Width{ .bits = 6, .group = 64 }, w.of(27, .v));
    try std.testing.expectEqual(Width{ .bits = 5, .group = 64 }, w.of(27, .k));
    try std.testing.expectEqual(uniform, w.of(27, .q));
    try std.testing.expectEqual(uniform, w.head);
}

test "the routed block admits the publisher's widths and refuses those its kernels are not checked at" {
    var why: Why = .{};
    // ornith-ai/Ornith-1.5-122B-MLX-4bit: 4-bit g64, the router and shared gate at 8 bits
    const published =
        \\{"group_size":64,"bits":4,"mode":"affine",
        \\"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":8},
        \\"language_model.model.layers.0.mlp.shared_expert_gate":{"group_size":64,"bits":8}}
    ;
    try testAdmitRouted(published, &why);
    try testAdmitRouted("{\"group_size\":64,\"bits\":8,\"mode\":\"affine\"}", &why);
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmitRouted("{\"group_size\":32,\"bits\":4,\"mode\":\"affine\"}", &why));
    try std.testing.expectEqual(error.UncheckedRoutedWidth, why.reason);
    const five =
        \\{"group_size":64,"bits":4,"mode":"affine",
        \\"language_model.model.layers.3.mlp.switch_mlp.up_proj":{"group_size":64,"bits":5}}
    ;
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmitRouted(five, &why));
    try std.testing.expectEqual(error.UncheckedRoutedWidth, why.reason);
}

test "a shared expert whose width differs from the experts' is refused at admission, not at the first decode" {
    var why: Why = .{};
    const mixed =
        \\{"group_size":64,"bits":4,"mode":"affine",
        \\"language_model.model.layers.5.mlp.shared_expert.down_proj":{"group_size":64,"bits":8}}
    ;
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmitRouted(mixed, &why));
    try std.testing.expectEqual(error.SharedExpertWidth, why.reason);
    try std.testing.expectEqualStrings("mlp.shared_expert.down_proj", why.module);
}

test "Qwen3.6-35B-A3B's published widths are admitted with and without tensor units" {
    var why: Why = .{};
    // mlx-community/Qwen3.6-35B-A3B-4bit: 4-bit g64, every layer's router and shared gate at 8 bits
    const published =
        \\{"group_size":64,"bits":4,"mode":"affine",
        \\"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":8},
        \\"language_model.model.layers.0.mlp.shared_expert_gate":{"group_size":64,"bits":8},
        \\"language_model.model.layers.39.mlp.gate":{"group_size":64,"bits":8},
        \\"language_model.model.layers.39.mlp.shared_expert_gate":{"group_size":64,"bits":8}}
    ;
    for ([_]bool{ true, false }) |units| try testAdmitA3b(published, units, &why);
    // a router at 8 bits with its shared gate left at 4 cannot share one matrix
    const split_gate =
        \\{"group_size":64,"bits":4,"mode":"affine",
        \\"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":8}}
    ;
    try std.testing.expectError(error.UnqualifiedQwenFormat, testAdmitA3b(split_gate, false, &why));
    try std.testing.expectEqual(error.SharedGateWidth, why.reason);
}

test "looped costShape appends cache-busting tag without altering Qwen shapes" {
    const a = std.testing.allocator;
    const text = try testConfig(a, "{\"bits\":4,\"group_size\":64}");
    defer a.free(text);
    var c = try cfg.parse(a, text, null);
    var buf: [96]u8 = undefined;
    try std.testing.expectEqualStrings("64/5120/17408/248320/16/1", try costShape(c, 0, &buf, 16, true));
    c.looped = true;
    try std.testing.expectEqualStrings("64/5120/17408/248320/16/1/sn2", try costShape(c, 0, &buf, 16, true));
    try std.testing.expectEqualStrings("64/5120/17408/248320/16/0/w000000000000002a/sn2", try costShape(c, 0x2a, &buf, 16, false));
}

test "affine widths tag keeps main's three-byte key while non-affine kinds hash distinctly" {
    const a = std.testing.allocator;
    var w = try Widths.uniformAll(a, 1);
    defer w.deinit();
    w.layers[0][0] = .{ .bits = 8, .group = 64 };
    var want_hash = std.hash.Wyhash.init(0x27);
    for (w.layers) |l| for (l) |x| want_hash.update(&.{ x.bits, @as(u8, @truncate(x.group)), @as(u8, @truncate(x.group >> 8)) });
    for ([_]Width{ w.embed, w.head }) |x| want_hash.update(&.{ x.bits, @as(u8, @truncate(x.group)), @as(u8, @truncate(x.group >> 8)) });
    try std.testing.expectEqual(want_hash.final() | 1, w.tag());
    w.layers[0][0] = .{ .bits = 4, .group = 128, .kind = .awq };
    try std.testing.expect(w.tag() != (want_hash.final() | 1));
}

test "looped affine resolves explicit unquantized embeddings as bf16 while keeping strict head/layer checks" {
    const a = std.testing.allocator;
    const base_text = try testConfig(a, "{\"bits\":4,\"group_size\":64}");
    defer a.free(base_text);
    var c_loop = try cfg.parse(a, base_text, null);
    c_loop.looped = true;
    const c_plain = try cfg.parse(a, base_text, null);

    for ([_][]const u8{
        "{\"bits\":4,\"group_size\":64,\"mode\":\"affine\",\"model.embed_tokens\":false}",
        "{\"bits\":4,\"group_size\":64,\"mode\":\"affine\",\"model.embed_tokens\":{}}",
    }) |override| {
        const text = try testConfig(a, override);
        defer a.free(text);
        var w = try Widths.resolveModel(a, text, c_loop);
        defer w.deinit();
        try std.testing.expectEqual(Width{ .bits = 16, .group = 0, .kind = .bf16 }, w.embed);
        try std.testing.expectEqual(uniform, w.head);
        try std.testing.expectEqual(uniform, w.layers[0][0]);
        var why: Why = .{};
        try admit(&w, c_loop, .{ .tensor_units = true }, &why);

        // Non-looped checkpoints still refuse unquantized affine embeddings.
        try std.testing.expectError(error.UnqualifiedQwenFormat, Widths.resolveModel(a, text, c_plain));
    }

    const bad_head = try testConfig(a, "{\"bits\":4,\"group_size\":64,\"mode\":\"affine\",\"model.embed_tokens\":false,\"lm_head\":false}");
    defer a.free(bad_head);
    try std.testing.expectError(error.UnqualifiedQwenFormat, Widths.resolveModel(a, bad_head, c_loop));
}
test "qwen27 tfq3: BF16 base and MLX-8bit base both admit with TFQ3 sidecar while plain BF16 still refuses" {
    const a = std.testing.allocator;
    const bf16_cfg =
        \\{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","hidden_size":5120,"intermediate_size":17408,
        \\"num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}}
    ;
    const c = try cfg.parse(a, bf16_cfg, null);
    var f_bf16 = try affine.Formats.init(a, bf16_cfg);
    defer f_bf16.deinit();
    try std.testing.expectError(error.MissingAffineFormat, Widths.resolve(a, &f_bf16, c));

    const FakeTfq3 = struct {
        fn bits(_: *const anyopaque, name: []const u8, n: usize, k: usize) ?u8 {
            if (n % 128 != 0 or k % 128 != 0) return null;
            if (std.mem.endsWith(u8, name, "mlp.gate_proj") or std.mem.endsWith(u8, name, "mlp.up_proj") or std.mem.endsWith(u8, name, "mlp.down_proj")) return 3;
            return 4;
        }
    };
    const lookup: Tfq3Lookup = .{ .ctx = &FakeTfq3.bits, .bitsFn = FakeTfq3.bits };
    var w_bf16 = try Widths.resolveWithTfq3(a, &f_bf16, c, lookup);
    defer w_bf16.deinit();
    try std.testing.expectEqual(bf16, w_bf16.embed);
    try std.testing.expectEqual(bf16, w_bf16.of(0, .a));
    try std.testing.expectEqual(Width{ .bits = 3, .group = tfq3_group }, w_bf16.of(0, .gate));
    try std.testing.expectEqual(Width{ .bits = 4, .group = tfq3_group }, w_bf16.head);
    var why: Why = .{};
    for ([_]bool{ false, true }) |units| try admitTfq3(&w_bf16, c, .{ .tensor_units = units }, &why);

    const q8_cfg = try testConfig(a, "{\"bits\":8,\"group_size\":64,\"mode\":\"affine\"}");
    defer a.free(q8_cfg);
    var f_q8 = try affine.Formats.init(a, q8_cfg);
    defer f_q8.deinit();
    var w_q8 = try Widths.resolveWithTfq3(a, &f_q8, c, lookup);
    defer w_q8.deinit();
    try std.testing.expectEqual(Width{ .bits = 8, .group = 64 }, w_q8.embed);
    try std.testing.expectEqual(Width{ .bits = 8, .group = 64 }, w_q8.of(0, .b));
    try std.testing.expectEqual(Width{ .bits = 4, .group = tfq3_group }, w_q8.of(0, .z));
    for ([_]bool{ false, true }) |units| try admitTfq3(&w_q8, c, .{ .tensor_units = units }, &why);
}
