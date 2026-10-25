//! QSA geometry changes retain the maintained norm, score, selection and softmax operation order.
const std = @import("std");
const edit = @import("../flashnext/geometry_source.zig");
const qsa = @import("cuda_attention.zig");
const root = "../../../kernels/metal/flashnext/";
pub const parts: usize = 16;
pub const names = [_][]const u8{ "qwen4_attn_prep", "qwen4_idx_pool", "qwen4_idx_scores", "qwen4_idx_select", "qwen4_attn_parts", "qwen4_attn_merge" };
const files = [_][]const u8{
    @embedFile(root ++ "q4_attn_prep_63a42a6cd8e0434a.metal"),
    @embedFile(root ++ "q4_idx_pool_6d826f5ebecec197.metal"),
    @embedFile(root ++ "q4_idx_scores_44c714a153f460b5.metal"),
    @embedFile(root ++ "q4_idx_select_4092b8bd389cc7ed.metal"),
    @embedFile(root ++ "q4_attn_parts_f6b4ffe8e2609998.metal"),
    @embedFile(root ++ "q4_attn_merge_gate_7070840cd3c22e98.metal"),
};
const old_names = [_][]const u8{
    "custom_kernel_q4_attn_prep_63a42a6cd8e0434a_bfloat16_t_int32_t_float_float_float_floatc_floatc_bfloat16_t_bfloat16_t_bfloat16_t",
    "custom_kernel_q4_idx_pool_6d826f5ebecec197_bfloat16_t_int32_tc_float_floatc_floatc_bfloat16_t",
    "custom_kernel_q4_idx_scores_44c714a153f460b5_bfloat16_t_bfloat16_t_int32_t_float",
    "custom_kernel_q4_idx_select_4092b8bd389cc7ed_float_int32_t_int32_t_int32_t",
    "custom_kernel_q4_attn_parts_f6b4ffe8e2609998_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_int32_t_int32_t_floatc_float_float",
    "custom_kernel_q4_attn_merge_gate_7070840cd3c22e98_float_float_bfloat16_t_bfloat16_t",
};
pub fn validate(g: qsa.Geometry) !void {
    try g.validate();
    if (g.top > 512) return error.Qwen4MetalIndexBudget;
}
fn integer(a: std.mem.Allocator, text: []const u8, name: []const u8, old: usize, new: usize) ![]u8 {
    return edit.once(a, text, try std.fmt.allocPrint(a, "constexpr int {s} = {d};", .{ name, old }), try std.fmt.allocPrint(a, "constexpr int {s} = {d};", .{ name, new }));
}
pub fn source(a: std.mem.Allocator, g: qsa.Geometry) ![]u8 {
    return from(a, g, files);
}
pub fn from(a: std.mem.Allocator, g: qsa.Geometry, texts: [6][]const u8) ![]u8 {
    try validate(g);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var body: [6][]const u8 = undefined;
    for (texts, old_names, names, 0..) |text, old, name, i| {
        body[i] = try edit.once(mem, try edit.function(text, old), old, name);
        const threads: usize = if (i == 3) 1024 else if (i == 1) 128 else 256;
        body[i] = try edit.once(mem, body[i], "[[kernel]]", try std.fmt.allocPrint(mem, "[[max_total_threads_per_threadgroup({d})]] [[kernel]]", .{threads}));
    }
    body[0] = try integer(mem, body[0], "NQ", 24, g.heads);
    body[0] = try integer(mem, body[0], "NKV", 2, g.kv);
    body[0] = try integer(mem, body[0], "RD", 64, g.rotary);
    body[0] = try integer(mem, body[0], "PW", 13952, g.projectionWidth());
    body[0] = try integer(mem, body[0], "NI", 4, g.index_heads);
    body[1] = try integer(mem, body[1], "RD", 64, g.rotary);
    const pool_edits = [_][2][]const u8{
        .{ "size_t(4 * b)", "size_t(RATIO * b)" }, .{ "k < 4;", "k < RATIO;" },
        .{ "m * 0.25f", "m / float(RATIO)" },      .{ "tf_rotary_scalar(i, 4 * b,", "tf_rotary_scalar(i, RATIO * b," },
    };
    for (pool_edits) |e| body[1] = try edit.once(mem, body[1], e[0], e[1]);
    body[1] = try std.mem.concat(mem, u8, &.{ try std.fmt.allocPrint(mem, "constexpr int RATIO = {d};\n", .{g.ratio}), body[1] });
    body[2] = try integer(mem, body[2], "HI", 4, g.index_heads);
    body[2] = try integer(mem, body[2], "TOP", 512, g.top);
    body[3] = try integer(mem, body[3], "TOP", 512, g.top);
    body[3] = try integer(mem, body[3], "KW", 2051, try g.ids());
    const select_edits = [_][2][]const u8{
        .{ "j < 4;", "j < RATIO;" },                                                         .{ "keys[out * 4 + j] = 4 * (cid[i]) + j;", "keys[out * RATIO + j] = RATIO * (cid[i]) + j;" },
        .{ "keys[out * 4 + j] = 4 * (i) + j;", "keys[out * RATIO + j] = RATIO * (i) + j;" }, .{ "int k = 4 * nb;", "int k = RATIO * nb;" },
        .{ "keys[4 * TOP + (k - 4 * nb)]", "keys[RATIO * TOP + (k - RATIO * nb)]" },
    };
    // The two branch-local loops have the same bound; all other edits must occur exactly once.
    if (std.mem.count(u8, body[3], select_edits[0][0]) != 2) return error.HybridGeometrySourceChanged;
    body[3] = try std.mem.replaceOwned(u8, mem, body[3], select_edits[0][0], select_edits[0][1]);
    for (select_edits[1..]) |e| body[3] = try edit.once(mem, body[3], e[0], e[1]);
    body[4] = try integer(mem, body[4], "H", 24, g.heads);
    body[4] = try integer(mem, body[4], "KVH", 2, g.kv);
    // Token-major caches share the state and snapshot layout with CUDA; only address strides change.
    body[4] = try edit.once(mem, body[4], "Kc + size_t(kvh) * cap * D + lane * PER", "Kc + size_t(kvh) * D + lane * PER");
    body[4] = try edit.once(mem, body[4], "Vc + size_t(kvh) * cap * D + lane * PER", "Vc + size_t(kvh) * D + lane * PER");
    body[4] = try edit.once(mem, body[4], "size_t(sparse ? ids[j] : j) * D", "size_t(sparse ? ids[j] : j) * KVH * D");
    body[5] = try integer(mem, body[5], "H", 24, g.heads);
    body[5] = try integer(mem, body[5], "PW", 13952, g.projectionWidth());
    body[5] = try edit.once(mem, body[5], "  const float g = float(GP[r * PW + h * 2 * D + D + d]);\n  OUT[(size_t(r) * H + h) * D + d] = bfloat(float(bfloat(acc / total)) * bsig(g));", "  OUT[(size_t(r) * H + h) * D + d] = bfloat(acc / total);");
    // Output gating uses the same explicit rounding-policy operator on both backends.
    const rotary = try edit.once(mem, try edit.function(texts[0], "inline TF_RotaryScalar tf_rotary_scalar("), "(float(i) / 32.0f)", try std.fmt.allocPrint(mem, "(float(i) / {d}.0f)", .{@max(@as(usize, 1), g.rotary / 2)}));
    return std.mem.concat(a, u8, &.{
        "#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bfloat16_t;\nstruct TF_RotaryScalar { float frequency, angle, cosine, sine, co, si; };\n",
        rotary,
        "\n",
        try edit.function(texts[3], "inline uint tf_key("),
        "\n",
        try edit.function(texts[3], "inline void cut_bin("),
        "\n",
        body[0],
        "\n",
        body[1],
        "\n",
        body[2],
        "\n",
        body[3],
        "\n",
        body[4],
        "\n",
        body[5],
        "\n",
    });
}
