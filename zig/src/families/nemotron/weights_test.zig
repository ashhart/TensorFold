//! weights.zig's loader checks: dtypes, widths stored against config.json's, and a mixed-width block's layout.
const std = @import("std");
const ckpt = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");
const wts = @import("weights.zig");

const Builder = wts.Builder;
const Weights = wts.Weights;

/// One tensor of a fake safetensors header.
const Fake = struct { name: []const u8, dtype: []const u8 = "BF16", shape: []const usize };

/// A checkpoint indexed from a fake header naming `tensors` back to back (the checks read headers, never data).
fn fakeCheckpoint(a: std.mem.Allocator, tensors: []const Fake) !ckpt.Checkpoint {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(a);
    try json.appendSlice(a, "{\"__metadata__\": {\"format\": \"mlx\"}");
    var at: usize = 0;
    for (tensors) |t| {
        var bytes = (ckpt.DType.parse(t.dtype) orelse return error.UnsupportedDType).size();
        try json.print(a, ", \"{s}\": {{\"dtype\": \"{s}\", \"shape\": [", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, i| {
            bytes *= d;
            try json.print(a, "{s}{d}", .{ if (i > 0) ", " else "", d });
        }
        try json.print(a, "], \"data_offsets\": [{d}, {d}]}}", .{ at, at + bytes });
        at += bytes;
    }
    try json.append(a, '}');
    var ck = ckpt.Checkpoint.init(a);
    errdefer ck.deinit();
    try ck.index(undefined, json.items, at, "");
    return ck;
}

/// A one-layer MoE model: hidden 64, vocab 128, 4 experts of width 64.
fn tinyMoe() cfg.Config {
    var c = cfg.Config{ .hidden = 64, .vocab = 128, .layers = 1, .mamba_heads = 2, .mamba_head_dim = 32, .groups = 1, .state = 16, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 32, .experts = 4, .top_k = 2, .expert_width = 64, .shared_width = 128, .routed_scaling = 1.0, .eps = 1e-5 };
    c.kinds[0] = .moe;
    return c;
}

/// The tensors `fill` reads for tinyMoe before its first buffer, each of `swaps` in place of its namesake or added.
fn tinyTensors(swaps: []const Fake, out: *[24]Fake) []const Fake {
    const base = [_]Fake{
        .{ .name = "backbone.norm_f.weight", .shape = &.{64} },
        .{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 8 } },
        .{ .name = "backbone.embeddings.scales", .shape = &.{ 128, 1 } },
        .{ .name = "backbone.embeddings.biases", .shape = &.{ 128, 1 } },
        .{ .name = "backbone.layers.0.norm.weight", .shape = &.{64} },
        .{ .name = "backbone.layers.0.mixer.gate.weight", .shape = &.{ 4, 64 } },
        .{ .name = "backbone.layers.0.mixer.gate.e_score_correction_bias", .dtype = "F32", .shape = &.{4} },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 4, 64, 8 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.scales", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.biases", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 4, 64, 8 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.scales", .shape = &.{ 4, 64, 1 } },
        .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.biases", .shape = &.{ 4, 64, 1 } },
    };
    @memcpy(out[0..base.len], &base);
    var n: usize = base.len;
    for (swaps) |swap| {
        const at = for (out[0..n], 0..) |t, i| {
            if (std.mem.eql(u8, t.name, swap.name)) break i;
        } else blk: {
            n += 1;
            break :blk n - 1;
        };
        out[at] = swap;
    }
    return out[0..n];
}

/// fill() on tinyMoe with `swaps` must fail with `err`, its reason naming every one of `words`.
fn expectRefused(swaps: []const Fake, err: anyerror, words: []const []const u8) !void {
    const a = std.testing.allocator;
    var buf: [24]Fake = undefined;
    var ck = try fakeCheckpoint(a, tinyTensors(swaps, &buf));
    defer ck.deinit();
    const c = tinyMoe();
    var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
    defer w.deinit();
    var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
    defer b.jobs.deinit(a);
    try std.testing.expectError(err, b.fill(null));
    for (words) |word| if (std.mem.indexOf(u8, b.why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ b.why.text(), word });
        return error.TestUnexpectedResult;
    };
}

test "the Metal loader refuses scale, bias and norm dtypes its kernels do not read" {
    const fc2 = "backbone.layers.0.mixer.switch_mlp.fc2";
    try expectRefused(&.{.{ .name = "backbone.embeddings.scales", .dtype = "F16", .shape = &.{ 128, 1 } }}, error.UnsupportedQuantization, &.{ "backbone.embeddings", "f16 scales" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.biases", .dtype = "F32", .shape = &.{ 4, 64, 1 } }}, error.UnsupportedQuantization, &.{ "switch_mlp.fc1", "f32 biases" });
    try expectRefused(&.{.{ .name = fc2 ++ ".scales", .dtype = "F16", .shape = &.{ 4, 64, 1 } }}, error.UnsupportedQuantization, &.{ fc2, "f16 scales" });
    try expectRefused(&.{.{ .name = fc2 ++ ".weight", .dtype = "I32", .shape = &.{ 4, 64, 8 } }}, error.UnsupportedQuantization, &.{ fc2, "i32 weight" });
    try expectRefused(&.{.{ .name = "backbone.norm_f.weight", .dtype = "F16", .shape = &.{64} }}, error.UnexpectedTensor, &.{ "backbone.norm_f.weight", "f16" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.norm.weight", .dtype = "F32", .shape = &.{64} }}, error.UnexpectedTensor, &.{ "layers.0.norm.weight", "f32" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.gate.weight", .dtype = "F16", .shape = &.{ 4, 64 } }}, error.UnexpectedTensor, &.{ "gate.weight", "f16" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.gate.e_score_correction_bias", .shape = &.{4} }}, error.UnexpectedTensor, &.{ "e_score_correction_bias", "bf16" });
}

test "the Metal loader refuses a matrix stored at another width than config.json declares" {
    const fc1 = "backbone.layers.0.mixer.switch_mlp.fc1";
    const up = "backbone.layers.0.mixer.shared_experts.up_proj";
    try expectRefused(&.{.{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 12 } }}, error.UnexpectedTensor, &.{ "backbone.embeddings", "stored at 6-bit in groups of 64", "declares 4-bit in groups of 64" });
    try expectRefused(&.{
        .{ .name = fc1 ++ ".weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
        .{ .name = fc1 ++ ".scales", .shape = &.{ 4, 64, 2 } },
        .{ .name = fc1 ++ ".biases", .shape = &.{ 4, 64, 2 } },
    }, error.UnexpectedTensor, &.{ fc1, "stored at 8-bit in groups of 32" });
    try expectRefused(&.{.{ .name = fc1 ++ ".biases", .shape = &.{ 4, 64, 2 } }}, error.UnexpectedTensor, &.{ fc1, "words" });
    try expectRefused(&.{.{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 3, 64, 8 } }}, error.UnexpectedTensor, &.{ "switch_mlp.fc2", "{ 4, 64, 64 }" });
    try expectRefused(&.{
        .{ .name = up ++ ".weight", .dtype = "U32", .shape = &.{ 128, 16 } },
        .{ .name = up ++ ".scales", .shape = &.{ 128, 2 } },
        .{ .name = up ++ ".biases", .shape = &.{ 128, 2 } },
    }, error.UnexpectedTensor, &.{ up, "stored at 8-bit in groups of 32" });
}

/// tinyMoe's matrices at 8-bit groups of 64 (the shared up at `up_bits`), with the shared expert and the head.
fn eightBit(comptime up_bits: u8) [12]Fake {
    const p = "backbone.layers.0.mixer.";
    return .{
        .{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 16 } },
        .{ .name = p ++ "switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
        .{ .name = p ++ "switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
        .{ .name = p ++ "shared_experts.up_proj.weight", .dtype = "U32", .shape = &.{ 128, 64 * @as(usize, up_bits) / 32 } },
        .{ .name = p ++ "shared_experts.up_proj.scales", .shape = &.{ 128, 1 } },
        .{ .name = p ++ "shared_experts.up_proj.biases", .shape = &.{ 128, 1 } },
        .{ .name = p ++ "shared_experts.down_proj.weight", .dtype = "U32", .shape = &.{ 64, 32 } },
        .{ .name = p ++ "shared_experts.down_proj.scales", .shape = &.{ 64, 2 } },
        .{ .name = p ++ "shared_experts.down_proj.biases", .shape = &.{ 64, 2 } },
        .{ .name = "lm_head.weight", .dtype = "U32", .shape = &.{ 128, 16 } },
        .{ .name = "lm_head.scales", .shape = &.{ 128, 1 } },
        .{ .name = "lm_head.biases", .shape = &.{ 128, 1 } },
    };
}

test "8-bit matrices read in place for the row kernels: no tiles, routed experts by slot, a mixed shared pair in rows" {
    const a = std.testing.allocator;
    inline for ([_]u8{ 8, 4 }) |up_bits| {
        var buf: [24]Fake = undefined;
        const swaps = eightBit(up_bits);
        var ck = try fakeCheckpoint(a, tinyTensors(&swaps, &buf));
        defer ck.deinit();
        var c = tinyMoe();
        c.widths[0] = .{ .{ .bits = 8 }, .{ .bits = 8 }, .{ .bits = up_bits }, .{ .bits = 8 } };
        c.embed_width = .{ .bits = 8 };
        c.head_width = .{ .bits = 8 };
        var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
        defer w.deinit();
        var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
        defer b.jobs.deinit(a);
        try b.fill(null);
        try std.testing.expectEqual(@as(usize, 0), b.jobs.items.len); // nothing to tile, no buffer made
        const m = w.layers[0].moe;
        try std.testing.expect(m.shared_up.tiles == null and m.shared_down.tiles == null and w.head.tiles == null);
        try std.testing.expectEqual(@as(u8, 8), m.routed.?[1].one.bits);
        try std.testing.expectEqual(@as(usize, 64), m.routed.?[1].one.k);
        try std.testing.expectEqual(@as(usize, 4), m.routed.?[0].experts);
        try std.testing.expectEqual(up_bits, m.shared_up.rows.?.bits);
        try std.testing.expectEqual(@as(usize, 128), m.shared_down.rows.?.k);
        try std.testing.expectEqual(@as(u8, 8), w.head.rows.?.bits);
        try std.testing.expectEqual(@as(u8, 8), w.embed_width.bits);
    }
}

/// A two-layer 8-bit nemotron_h_puzzle (Bolt) model: hidden 128, latent 64, experts 4 and 8, widths 64 and 128.
fn tinyBolt() cfg.Config {
    var c = cfg.Config{ .hidden = 128, .vocab = 128, .layers = 2, .mamba_heads = 2, .mamba_head_dim = 32, .groups = 1, .state = 16, .conv_kernel = 4, .heads = 2, .kv_heads = 1, .head_dim = 32, .experts = 8, .top_k = 2, .expert_width = 128, .shared_width = 64, .routed_scaling = 5.0, .eps = 1e-5 };
    c.bolt = true;
    c.latent = 64;
    c.kinds[0] = .moe;
    c.kinds[1] = .moe;
    c.moe[0] = .{ .experts = 4, .width = 64 };
    c.moe[1] = .{ .experts = 8, .width = 128 };
    for (0..2) |i| c.widths[i] = .{ .{ .bits = 8 }, .{ .bits = 8 }, .{ .bits = 8 }, .{ .bits = 8 } };
    c.embed_width = .{ .bits = 8 };
    c.head_width = .{ .bits = 8 };
    return c;
}

test "a nemotron_h_puzzle block loads per layer: its own expert count and width, experts in the latent, two bf16 latent projections" {
    const a = std.testing.allocator;
    var ck = try fakeCheckpoint(a, &bolt_tensors);
    defer ck.deinit();
    const c = tinyBolt();
    var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
    defer w.deinit();
    var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
    defer b.jobs.deinit(a);
    try b.fill(null);
    try std.testing.expectEqual(@as(usize, 0), b.jobs.items.len); // 8-bit everywhere: nothing tiled, no device needed
    const m0 = w.layers[0].moe;
    const m1 = w.layers[1].moe;
    // both blocks carry the two bf16 latent projections, hidden <-> latent
    try std.testing.expectEqual(@as(usize, 64), m0.into.?.n); // hidden 128 -> latent 64
    try std.testing.expectEqual(@as(usize, 128), m0.into.?.k);
    try std.testing.expectEqual(@as(usize, 128), m0.out.?.n); // latent 64 -> hidden 128
    try std.testing.expectEqual(@as(usize, 64), m0.out.?.k);
    try std.testing.expect(m1.into != null and m1.out != null);
    // each layer keeps its OWN expert count and width; the routed experts read the 64-wide latent
    try std.testing.expectEqual(@as(usize, 4), m0.routed.?[0].experts);
    try std.testing.expectEqual(@as(usize, 64), m0.routed.?[0].one.n); // layer 0 expert width
    try std.testing.expectEqual(@as(usize, 64), m0.routed.?[0].one.k); // experts read the latent, not hidden
    try std.testing.expectEqual(@as(usize, 8), m1.routed.?[0].experts);
    try std.testing.expectEqual(@as(usize, 128), m1.routed.?[0].one.n); // layer 1 expert width
    try std.testing.expectEqual(@as(usize, 64), m1.routed.?[0].one.k);
    try std.testing.expectEqual(@as(usize, 128), m1.routed.?[1].one.k); // fc2 reads layer 1's own width back down
}

/// tinyBolt's checkpoint: 8-bit matrices (row kernels, no tiles) and the two bf16 latent projections per block.
const bolt_tensors = [_]Fake{
    .{ .name = "backbone.norm_f.weight", .shape = &.{128} },
    .{ .name = "backbone.embeddings.weight", .dtype = "U32", .shape = &.{ 128, 32 } },
    .{ .name = "backbone.embeddings.scales", .shape = &.{ 128, 2 } },
    .{ .name = "backbone.embeddings.biases", .shape = &.{ 128, 2 } },
    .{ .name = "lm_head.weight", .dtype = "U32", .shape = &.{ 128, 32 } },
    .{ .name = "lm_head.scales", .shape = &.{ 128, 2 } },
    .{ .name = "lm_head.biases", .shape = &.{ 128, 2 } },
    // layer 0: 4 experts of width 64
    .{ .name = "backbone.layers.0.norm.weight", .shape = &.{128} },
    .{ .name = "backbone.layers.0.mixer.gate.weight", .shape = &.{ 4, 128 } },
    .{ .name = "backbone.layers.0.mixer.gate.e_score_correction_bias", .dtype = "F32", .shape = &.{4} },
    .{ .name = "backbone.layers.0.mixer.fc1_latent_proj.weight", .shape = &.{ 64, 128 } },
    .{ .name = "backbone.layers.0.mixer.fc2_latent_proj.weight", .shape = &.{ 128, 64 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.scales", .shape = &.{ 4, 64, 1 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc1.biases", .shape = &.{ 4, 64, 1 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 4, 64, 16 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.scales", .shape = &.{ 4, 64, 1 } },
    .{ .name = "backbone.layers.0.mixer.switch_mlp.fc2.biases", .shape = &.{ 4, 64, 1 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.up_proj.weight", .dtype = "U32", .shape = &.{ 64, 32 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.up_proj.scales", .shape = &.{ 64, 2 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.up_proj.biases", .shape = &.{ 64, 2 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.down_proj.weight", .dtype = "U32", .shape = &.{ 128, 16 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.down_proj.scales", .shape = &.{ 128, 1 } },
    .{ .name = "backbone.layers.0.mixer.shared_experts.down_proj.biases", .shape = &.{ 128, 1 } },
    // layer 1: 8 experts of width 128
    .{ .name = "backbone.layers.1.norm.weight", .shape = &.{128} },
    .{ .name = "backbone.layers.1.mixer.gate.weight", .shape = &.{ 8, 128 } },
    .{ .name = "backbone.layers.1.mixer.gate.e_score_correction_bias", .dtype = "F32", .shape = &.{8} },
    .{ .name = "backbone.layers.1.mixer.fc1_latent_proj.weight", .shape = &.{ 64, 128 } },
    .{ .name = "backbone.layers.1.mixer.fc2_latent_proj.weight", .shape = &.{ 128, 64 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 8, 128, 16 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc1.scales", .shape = &.{ 8, 128, 1 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc1.biases", .shape = &.{ 8, 128, 1 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc2.weight", .dtype = "U32", .shape = &.{ 8, 64, 32 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc2.scales", .shape = &.{ 8, 64, 2 } },
    .{ .name = "backbone.layers.1.mixer.switch_mlp.fc2.biases", .shape = &.{ 8, 64, 2 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.up_proj.weight", .dtype = "U32", .shape = &.{ 64, 32 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.up_proj.scales", .shape = &.{ 64, 2 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.up_proj.biases", .shape = &.{ 64, 2 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.down_proj.weight", .dtype = "U32", .shape = &.{ 128, 16 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.down_proj.scales", .shape = &.{ 128, 1 } },
    .{ .name = "backbone.layers.1.mixer.shared_experts.down_proj.biases", .shape = &.{ 128, 1 } },
};

/// fill() on tinyBolt with `swaps` (and without `drop`) must fail with `err`, its reason naming every one of `words`.
fn expectBoltRefused(swaps: []const Fake, drop: []const u8, err: anyerror, words: []const []const u8) !void {
    const a = std.testing.allocator;
    var buf: [bolt_tensors.len + 4]Fake = undefined;
    var n: usize = 0;
    for (bolt_tensors) |t| if (!std.mem.eql(u8, t.name, drop)) {
        buf[n] = t;
        n += 1;
    };
    for (swaps) |swap| {
        const at = for (buf[0..n], 0..) |t, i| {
            if (std.mem.eql(u8, t.name, swap.name)) break i;
        } else blk: {
            n += 1;
            break :blk n - 1;
        };
        buf[at] = swap;
    }
    var ck = try fakeCheckpoint(a, buf[0..n]);
    defer ck.deinit();
    const c = tinyBolt();
    var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
    defer w.deinit();
    var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
    defer b.jobs.deinit(a);
    try std.testing.expectError(err, b.fill(null));
    for (words) |word| if (std.mem.indexOf(u8, b.why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ b.why.text(), word });
        return error.TestUnexpectedResult;
    };
}

test "an unquantized bf16 lm_head loads as Bolt's dense head, and is refused for Lightning as before" {
    const a = std.testing.allocator;
    const bf16_head = Fake{ .name = "lm_head.weight", .shape = &.{ 128, 128 } };
    {
        var buf: [bolt_tensors.len]Fake = undefined;
        var n: usize = 0;
        for (bolt_tensors) |t| if (!std.mem.startsWith(u8, t.name, "lm_head.")) {
            buf[n] = t;
            n += 1;
        };
        buf[n] = bf16_head;
        n += 1;
        var ck = try fakeCheckpoint(a, buf[0..n]);
        defer ck.deinit();
        const c = tinyBolt();
        var w = Weights{ .allocator = a, .embed = undefined, .norm_f = undefined, .head = undefined };
        defer w.deinit();
        var b = Builder{ .allocator = a, .device = undefined, .ck = &ck, .c = &c, .w = &w };
        defer b.jobs.deinit(a);
        try b.fill(null);
        try std.testing.expectEqual(@as(usize, 128), w.head_dense.?.n);
        try std.testing.expectEqual(@as(usize, 128), w.head_dense.?.k);
    }
    {
        // Lightning (tinyMoe) with the same bf16 head and no lm_head.scales: still the quantized head's load (which
        // refuses it for the missing scales, as before), never the dense one
        var ck = try fakeCheckpoint(a, &.{.{ .name = "lm_head.weight", .shape = &.{ 128, 64 } }});
        defer ck.deinit();
        try std.testing.expect(!wts.denseHead(tinyMoe(), &ck));
        var bolt_ck = try fakeCheckpoint(a, &.{.{ .name = "lm_head.weight", .shape = &.{ 128, 128 } }});
        defer bolt_ck.deinit();
        try std.testing.expect(wts.denseHead(tinyBolt(), &bolt_ck));
    }
}

test "the Metal loader refuses a puzzle block's latent tensors when missing, transposed, quantized or not bf16" {
    const p = "backbone.layers.0.mixer.";
    try expectBoltRefused(&.{}, p ++ "fc1_latent_proj.weight", error.MissingTensor, &.{ "no " ++ p ++ "fc1_latent_proj.weight", "[64, 128]" });
    try expectBoltRefused(&.{}, p ++ "fc2_latent_proj.weight", error.MissingTensor, &.{"fc2_latent_proj.weight"});
    try expectBoltRefused(&.{.{ .name = p ++ "fc2_latent_proj.weight", .shape = &.{ 64, 128 } }}, "", error.UnexpectedTensor, &.{ "fc2_latent_proj", "{ 128, 64 }" });
    try expectBoltRefused(&.{.{ .name = p ++ "fc1_latent_proj.weight", .dtype = "F16", .shape = &.{ 64, 128 } }}, "", error.UnexpectedTensor, &.{ "fc1_latent_proj", "f16" });
    try expectBoltRefused(&.{
        .{ .name = p ++ "fc1_latent_proj.weight", .dtype = "U32", .shape = &.{ 64, 32 } },
        .{ .name = p ++ "fc1_latent_proj.scales", .shape = &.{ 64, 2 } },
        .{ .name = p ++ "fc1_latent_proj.biases", .shape = &.{ 64, 2 } },
    }, "", error.UnsupportedQuantization, &.{ "fc1_latent_proj", "quantized" });
    // routed experts stored over hidden (128) rather than the 64-wide latent
    try expectBoltRefused(&.{.{ .name = p ++ "switch_mlp.fc1.weight", .dtype = "U32", .shape = &.{ 4, 64, 32 } }}, "", error.UnexpectedTensor, &.{"switch_mlp.fc1"});
}
