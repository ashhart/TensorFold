//! Host checks use served admission and contract references, without calling a device or GPU runner.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const t = std.testing;
const a = t.allocator;
const contract = @import("contract");
const refs = @import("tests/contract/metal_qwen27_moe.zig");
const rows_ref = @import("tests/contract/metal_qwen27_moe_rows.zig");
const spec = contract.spec;

const json35 =
    \\{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text",
    \\"hidden_size":2048,"num_hidden_layers":40,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
    \\"vocab_size":248320,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,
    \\"num_experts":256,"num_experts_per_tok":8,"norm_topk_prob":true,"linear_num_key_heads":16,
    \\"linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\"partial_rotary_factor":0.25,"eos_token_id":248046,"rms_norm_eps":0.000001}}
;

test "qwen35 Metal: served admission adds the 35B and preserves dense and earlier routed shapes" {
    const c = try q.config.parse(a, json35, null);
    try q.model.supported(c);
    try t.expectEqual(@as(usize, 512), c.intermediate);
    try t.expect(c.moe() and c.renorm);
    try t.expectEqual(@as(u32, 8), (try q.attention_plan.Geometry.fromConfig(c, true)).group());
    for (0..c.layers) |i| try t.expectEqual(if (i % 4 == 3) q.config.Kind.attention else .linear, c.kind(i));
    var old = c;
    old.hidden = 3072;
    old.intermediate = 1024;
    old.moe_intermediate = 1024;
    old.layers = 48;
    old.heads = 32;
    old.v_heads = 64;
    for (old.kinds[0..old.layers], 0..) |*k, i| k.* = if (i % 4 == 3) .attention else .linear;
    try q.model.supported(old);
    try t.expectEqual(@as(u32, 16), (try q.attention_plan.Geometry.fromConfig(old, true)).group());
    old.experts = 0;
    old.top_k = 0;
    old.moe_intermediate = 0;
    old.hidden = 5120;
    old.intermediate = 17408;
    old.layers = 64;
    old.heads = 24;
    old.kv_heads = 4;
    old.v_heads = 48;
    for (old.kinds[0..old.layers], 0..) |*k, i| k.* = if (i % 4 == 3) .attention else .linear;
    try q.model.supported(old);
    try t.expectEqual(@as(u32, 6), (try q.attention_plan.Geometry.fromConfig(old, true)).group());
    old.hidden = 4096;
    old.intermediate = 12288;
    old.layers = 32;
    old.heads = 16;
    old.v_heads = 32;
    try q.model.supported(old);
    try t.expectEqual(@as(u32, 4), (try q.attention_plan.Geometry.fromConfig(old, true)).group());
    inline for (.{ "hidden", "layers", "heads", "v_heads", "moe_intermediate", "experts", "top_k" }) |field| {
        var bad = c;
        @field(bad, field) += 1;
        try t.expectError(error.UnqualifiedQwenShape, q.model.supported(bad));
    }
    var bad = c;
    bad.kinds[3] = .linear;
    try t.expectError(error.UnqualifiedLayerSchedule, q.model.supported(bad));
}

test "qwen35 Metal: MLX routed widths and scratch admit on both chip paths" {
    const c = try q.config.parse(a, json35, null);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    try text.appendSlice(a, "{\"quantization\":{\"bits\":4,\"group_size\":64");
    for (0..c.layers) |i| for ([_][]const u8{ "gate", "shared_expert_gate" }) |name|
        try text.print(a, ",\"model.layers.{d}.mlp.{s}\":{{\"bits\":8,\"group_size\":64}}", .{ i, name });
    try text.appendSlice(a, "}}");
    var formats = try q.affine.Formats.init(a, text.items);
    defer formats.deinit();
    var widths = try q.widths.Widths.resolve(a, &formats, c);
    defer widths.deinit();
    for ([_]bool{ false, true }) |units| {
        var why: q.widths.Why = .{};
        try q.widths.admit(&widths, c, .{ .tensor_units = units }, &why);
    }
    for (0..c.layers) |i| {
        try t.expectEqual(@as(u8, 8), widths.of(i, .router).bits);
        try t.expectEqual(@as(u8, 8), widths.of(i, .shared_gate).bits);
        for ([_]q.widths.Member{ .gate, .up, .down, .egate, .eup, .edown }) |m| {
            try t.expectEqual(@as(u8, 4), widths.of(i, m).bits);
            try t.expectEqual(@as(u16, 64), widths.of(i, m).group);
        }
    }
    const shape = try q.gdn_contract.Shape.init(c);
    try t.expectEqual(@as(usize, 8192), shape.qkv);
    try t.expectEqual(@as(usize, 4096), shape.value);
    try t.expectEqual(@as(usize, 32 * 128 * 128), shape.state);
    for ([_]u32{ 1, 16, 128, 1024 }) |rows| {
        const sizes = q.gpu_frame.allocationSizes(c, rows);
        try t.expectEqual(@as(usize, rows) * 9 * 4, sizes[@backingInt(q.gpu_frame.Field.picks)]);
        try t.expectEqual(@as(usize, rows) * 9 * 512 * 2, sizes[@backingInt(q.gpu_frame.Field.expert_act)]);
        try t.expectEqual(@as(usize, rows) * 320 * 2, sizes[@backingInt(q.gpu_frame.Field.router)]);
    }
}

test "qwen35 Metal: group eight reaches both prompt kernels and the pre-M5 rewrite" {
    const config = try q.config.parse(a, json35, null);
    const g = try q.attention_plan.Geometry.fromConfig(config, true);
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |prompt| {
        const text = try q.simd_attention.grouped(a, units, prompt, g.group(), g.kv_heads);
        defer a.free(text);
        const body = try q.simd_attention.library(a, units, prompt);
        defer a.free(body);
        const defines = "#define Q27_G 8\n#define Q27_KVH 2\n";
        try t.expect(std.mem.startsWith(u8, text, defines));
        try t.expectEqualStrings(body, text[defines.len..]);
        if (prompt) {
            try t.expect(std.mem.indexOf(u8, text, "PG = Q27_G, PKVH = Q27_KVH") != null);
            try t.expect(std.mem.indexOf(u8, text, "kernel void q27_attn_prompt(") != null);
            if (!units) try t.expect(std.mem.indexOf(u8, text, "kernel void q27_attn_prompt_sg(") != null);
        }
    };
    var invalid = g;
    invalid.heads = 8;
    invalid.kv_heads = 1;
    try t.expectError(error.TargetAttentionShape, invalid.check());
}

test "qwen35 Metal: the extra expert FP64 reference covers 512 by 2048 and its down projection" {
    for ([_]u8{ 4, 8 }) |bits| for ([_][2]usize{ .{ 512, 2048 }, .{ 2048, 512 } }) |nk| {
        const f = try contract.affine.Affine.init(a, .{ .rows = 2, .n = 4 * nk[0], .k = nk[1] }, bits, 64);
        defer f.deinit();
        @memset(f.x, spec.toBf16(1));
        @memset(f.scales, spec.toBf16(1));
        @memset(f.biases, 0);
        for (0..4) |e| @memset(f.codes[e * nk[0] * nk[1] ..][0 .. nk[0] * nk[1]], @intCast(e + 1));
        var ids = [_]u32{ 0, 3, 4, 2, 1, 3 };
        var extra: refs.Extra = .{ .a = a, .rig = undefined, .pipes = undefined, .bits = bits, .name = "host", .f = f, .ids = &ids };
        const want = try a.alloc(f64, extra.count());
        defer a.free(want);
        const mag = try a.alloc(f64, want.len);
        defer a.free(mag);
        try extra.reference(want, mag);
        for (ids, 0..) |id, slot| for (want[slot * nk[0] ..][0..nk[0]]) |v|
            try t.expectEqual(if (id <= 3) @as(f64, @floatFromInt(nk[1] * (id + 1))) else 0, v);
    };
}

test "qwen35 Metal: the existing route FP64 oracle keeps ties, expert 255 and the separate shared gate" {
    var logits: [3 * 320]u16 = @splat(0);
    for (248..256) |e| logits[320 + e] = spec.toBf16(3);
    logits[320 + 256] = spec.toBf16(100);
    logits[2 * 320] = spec.toBf16(std.math.nan(f32));
    var route: refs.Route = .{ .a = a, .rig = undefined, .pipe = undefined, .rows = 3, .logits = &logits };
    var weights: [9]f64 = undefined;
    var ids: [8]u32 = undefined;
    for (0..2) |r| {
        try t.expect(route.referenceRow(r, &weights, &ids));
        for (ids, 0..) |id, i| try t.expectEqual(@as(u32, @intCast((if (r == 0) @as(usize, 0) else 248) + i)), id);
        for (weights[0..8]) |w| try t.expectEqual(@as(f64, 0.125), w);
        try t.expectEqual(if (r == 0) @as(f64, 0.5) else 1, weights[8]);
    }
    try t.expect(!route.referenceRow(2, &weights, &ids));
    for (weights[0..8]) |w| try t.expect(std.math.isNan(w));
}

test "qwen35 Metal: SwiGLU and both shared combine layouts use the existing FP64 references" {
    const gates = try a.alloc(u16, 2 * 9 * 512);
    defer a.free(gates);
    @memset(gates[0 .. 9 * 512], spec.toBf16(1));
    @memset(gates[9 * 512 ..], spec.toBf16(2));
    var glu: rows_ref.Glu = .{ .a = a, .rig = undefined, .pipe = undefined, .shape = .{ .rows = 9, .n = 512, .k = 1 }, .gu = gates };
    const act = try a.alloc(f64, glu.count());
    defer a.free(act);
    const act_mag = try a.alloc(f64, act.len);
    defer a.free(act_mag);
    try glu.reference(act, act_mag);
    for (act) |v| try t.expectEqual(2 / (1 + @exp(@as(f64, -1))), v);

    const y = try a.alloc(u16, 9 * 2048);
    defer a.free(y);
    var weights: [9]f32 = @splat(1);
    weights[8] = 0.5;
    const terms = [_]f32{ 65536, 1, 1, 1, 1, 1, 1, -65536 };
    for (terms, 0..) |v, k| @memset(y[k * 2048 ..][0..2048], spec.toBf16(v));
    @memset(y[8 * 2048 ..], spec.toBf16(4));
    var shared: [2048]u16 = @splat(spec.toBf16(4));
    var want: [2048]f64 = undefined;
    var mag: [2048]f64 = undefined;
    for ([_]bool{ false, true }) |in_out| {
        var combine: rows_ref.Combine = .{ .a = a, .rig = undefined, .pipe = undefined, .in_out = in_out, .shape = .{ .rows = 1, .n = 2048, .k = 8 }, .y = y, .w = &weights, .old = &shared };
        try combine.reference(&want, &mag);
        for (want) |v| try t.expectEqual(@as(f64, 8), v);
        var reduced: [2048]f64 = undefined;
        try combine.simulate(.bf16, .sequential, &reduced);
        try t.expect(reduced[0] != want[0]);
    }
}
