//! Nemotron's KV-page twins: main's kernels with only the K/V address changed, the big three by count-checked edits.
const std = @import("std");
const generated = @import("kernels.zig");

const kv_pages = @embedFile("../core/kv_pages.h");
const paged_h = @embedFile("paged.h");

/// One count-checked edit of a source's text: every match of `from` becomes `to`, and there must be `count` of them.
fn edit(comptime text: []const u8, comptime from: []const u8, comptime to: []const u8, comptime count: usize) []const u8 {
    @setEvalBranchQuota(20_000_000);
    if (std.mem.count(u8, text, from) != count) @compileError("nemotron paged: expected " ++ std.fmt.comptimePrint("{d}", .{count}) ++ " of: " ++ from);
    var out: []const u8 = "";
    var rest = text;
    while (std.mem.indexOf(u8, rest, from)) |i| {
        out = out ++ rest[0..i] ++ to;
        rest = rest[i + from.len ..];
    }
    return out ++ rest;
}

/// The text from `start` up to (not including) the next `end`.
fn between(comptime text: []const u8, comptime start: []const u8, comptime end: []const u8) []const u8 {
    @setEvalBranchQuota(20_000_000);
    const a = std.mem.indexOf(u8, text, start) orelse @compileError("nemotron paged: no " ++ start);
    const b = std.mem.indexOfPos(u8, text, a, end) orelse @compileError("nemotron paged: no " ++ end);
    return text[a..b];
}

fn present(comptime text: []const u8, comptime what: []const u8) void {
    @setEvalBranchQuota(20_000_000);
    if (std.mem.indexOf(u8, text, what) == null) @compileError("nemotron paged: the source no longer holds " ++ what);
}

const table_args = "  device const uint* table [[buffer({d})]],\n  constant ulong& page_bytes [[buffer({d})]],\n";

/// tf_kv_write, tf_kv_compact and prefill's kv_put through a stream's table (paged.metal), in one library.
pub const glue = "#include <metal_stdlib>\nusing namespace metal;\n" ++ kv_pages ++ paged_h ++ @embedFile("paged.metal");
pub const glue_names = [_][:0]const u8{ "nemotron_kv_write_paged", "nemotron_kv_compact_paged", "nemotron_kv_put_paged" };

/// The decode attention's partial pass (attn_partial_1..16): each 64-key tile's keys and values from its page.
pub const partial = blk: {
    var source: []const u8 = undefined;
    for (generated.all) |k| {
        if (std.mem.eql(u8, k.key, "attn_partial_1")) source = k.source;
    }
    const at = std.mem.indexOf(u8, source, "template <int G").?;
    const name = between(source[at..], "[[kernel]] void ", "(")["[[kernel]] void ".len..];
    present(source, "decltype(" ++ name ++ "<16, 128, 1, 512, 64>)");
    var body = between(source, "template <int G", "template [[host_name");
    body = edit(body, name ++ "(", "nemotron_attn_partial_paged(", 1);
    body = edit(body, "  device float* PL [[buffer(9)]],\n", "  device float* PL [[buffer(9)]],\n" ++ std.fmt.comptimePrint(table_args, .{ 10, 11 }), 1);
    // a tile's slice of the whole cache, as a slice of its page bounded the same way: the rows to L, at most the page's
    body = edit(body, "    auto bK = tK.slice(0, kt);\n", "    auto bK = nemo_page_tensor<D>(K, table, page_bytes, kt, L, hk).slice(0, 0);\n", 1);
    body = edit(body, "    auto bVlo = tV.slice(0, kt);\n", "    auto bVlo = nemo_page_tensor<D>(V, table, page_bytes, kt, L, hk).slice(0, 0);\n", 1);
    var out: []const u8 = source[0..at] ++ kv_pages ++ paged_h ++ @embedFile("paged_tensor.h") ++ body;
    for (1..17) |n| {
        const args = std.fmt.comptimePrint("<16, 128, {d}, 512, 64>", .{n});
        out = out ++ std.fmt.comptimePrint("template [[host_name(\"nemotron_attn_partial_paged_{d}\")]] [[kernel]] ", .{n}) ++
            "decltype(nemotron_attn_partial_paged" ++ args ++ ") nemotron_attn_partial_paged" ++ args ++ ";\n";
    }
    break :blk out;
};

/// The twin of attn_partial_N at N tiles a threadgroup, 1 to 16.
pub const partial_names = blk: {
    var names: [16][:0]const u8 = undefined;
    for (&names, 1..) |*n, i| n.* = std.fmt.comptimePrint("nemotron_attn_partial_paged_{d}", .{i});
    break :blk names;
};

/// A tree's tail chunks (tf_tree_tail): each gathered key and value row from its page; nemotron_tree.metal before it.
pub const tree = blk: {
    const source = @embedFile("../nemotron_tree.metal");
    present(source, "decltype(tf_tree_tail<16, 128, 512, 64, 64>)");
    var body = between(source, "// A node's keys past the shared pass", "// A node's attention output");
    body = edit(body, "[[kernel]] void tf_tree_tail(", "[[kernel]] void nemotron_tree_tail_paged(", 1);
    body = edit(body, "  device float* PL [[buffer(14)]],\n", "  device float* PL [[buffer(14)]],\n" ++ std.fmt.comptimePrint(table_args, .{ 15, 16 }), 1);
    body = edit(body, "(kbase + phys * kstep + col)", "((const device uchar*)K + nemo_paged_row(table, uint(phys), page_bytes, hk, D) + col * 2)", 1);
    body = edit(body, "(vbase + phys * vstep + col)", "((const device uchar*)V + nemo_paged_row(table, uint(phys), page_bytes, hk, D) + col * 2)", 1);
    const args = "<16, 128, 512, 64, 64>";
    break :blk source ++ kv_pages ++ paged_h ++ body ++ "template [[host_name(\"nemotron_tree_tail_paged\")]] [[kernel]] " ++
        "decltype(nemotron_tree_tail_paged" ++ args ++ ") nemotron_tree_tail_paged" ++ args ++ ";\n";
};
pub const tree_name: [:0]const u8 = "nemotron_tree_tail_paged";

/// Prompt attention (tf_attention_nax): each 32-key block's keys and values from its page; attention_nax.metal first.
pub const prefill = blk: {
    const source = @embedFile("../prefill/attention_nax.metal");
    var body = between(source, "// One 64-query block", "}  // namespace tfp");
    body = edit(body, "inline void attention_nax(", "inline void attention_nax_paged(", 1);
    body = edit(body, "const device float* F, uint3 tg, uint sg, uint lane) {", "const device float* F, device const uint* table, ulong page_bytes, uint3 tg, uint sg, uint lane) {", 1);
    // a stream's keys start at its table's first page: no batch or head offset, a block's rows found by position
    body = edit(body, "  K += long(tg.z) * P[15] + long(kv) * P[16];\n  V += long(tg.z) * P[18] + long(kv) * P[19];\n", "", 1);
    body = edit(body, "    const bool last_k = kb == P[8];\n", "    const bool last_k = kb == P[8];\n" ++
        "    const device T* Kb = (const device T*)((const device uchar*)K + nemo_paged_row(table, uint(kb * BK), page_bytes, uint(kv), uint(ks)));\n" ++
        "    const device T* Vb = (const device T*)((const device uchar*)V + nemo_paged_row(table, uint(kb * BK), page_bytes, uint(kv), uint(vs)));\n", 1);
    body = edit(body, ", K, ks, ", ", Kb, ks, ", 4);
    body = edit(body, ", V, vs, ", ", Vb, vs, ", 4);
    body = edit(body, "    K += BK * ks;\n    V += BK * vs;\n", "", 1);
    var out: []const u8 = source ++ kv_pages ++ paged_h ++ "namespace tfp {\n" ++ body ++ "}  // namespace tfp\n";
    for (prefill_tags) |t| {
        present(source, "custom_kernel_tf_attention_nax_bf16_" ++ t ++ "_t_bfloat16_t");
        out = out ++ "[[max_total_threads_per_threadgroup(128)]]\n[[kernel]] void nemotron_attention_paged_" ++ t ++ "(\n" ++
            "  const device bfloat16_t* Q [[buffer(0)]],\n  const device bfloat16_t* K [[buffer(1)]],\n  const device bfloat16_t* V [[buffer(2)]],\n" ++
            "  const device int32_t* P [[buffer(3)]],\n  const device float* F [[buffer(4)]],\n  device bfloat16_t* O [[buffer(5)]],\n" ++
            std.fmt.comptimePrint(table_args, .{ 6, 7 }) ++
            "  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n" ++
            "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n" ++
            "  tfp::attention_nax_paged<bfloat16_t, " ++ flag(t[0]) ++ ", " ++ flag(t[2]) ++ ", true>(Q, K, V, O, P, F, table, page_bytes, " ++
            "threadgroup_position_in_grid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);\n}\n";
    }
    break :blk out;
};

/// tf_attention_nax's variants by (whole query blocks, whole key blocks), as prefill_launch names them.
pub const prefill_tags = [_][]const u8{ "n_n", "n_t", "t_n", "t_t" };
pub const prefill_names = [_][:0]const u8{ "nemotron_attention_paged_n_n", "nemotron_attention_paged_n_t", "nemotron_attention_paged_t_n", "nemotron_attention_paged_t_t" };

fn flag(c: u8) []const u8 {
    return if (c == 't') "true" else "false";
}
