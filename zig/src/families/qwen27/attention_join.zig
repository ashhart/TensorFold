//! Q27_JOIN: rows on keys they hold alike share q27_attn_tail's chunks in pairs (attention_plan.Plan.pair).
const std = @import("std");
const mtl = @import("metal");

/// Attention.join's pipelines (pack_b, tail, merge, merge_single) from a library built by simd_attention.joined.
pub fn pipelines(device: mtl.Device, library: mtl.Library) ![4]mtl.Pipeline {
    var out: [4]mtl.Pipeline = undefined;
    var n: usize = 0;
    errdefer for (out[0..n]) |pipe| pipe.deinit();
    for ([_][]const u8{ "q27_attn_pack_b", "q27_attn_tail", "q27_attn_merge", "q27_attn_merge_single" }, &out) |name, *pipe| {
        pipe.* = try mtl.Pipeline.init(device, library, name, false);
        n += 1;
    }
    return out;
}

/// The runner's join before M5 (TF_QWEN27_JOIN, on unless 0); null for the M5's prefix-kernel plan or a full tile.
pub fn build(a: std.mem.Allocator, device: mtl.Device, units: bool, group: u32, kv_heads: u32, paged: bool) !?[4]mtl.Pipeline {
    if (units or group > 8 or !try @import("sys").environment.read(bool, "TF_QWEN27_JOIN", true)) return null;
    const text = try @import("simd_attention.zig").joined(a, units, false, group, kv_heads, paged);
    defer a.free(text);
    const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer library.deinit();
    return try pipelines(device, library);
}
