//! Geometry specializations read maintained Flash Next bodies and fail closed at their source boundaries.
const std = @import("std");
fn once(a: std.mem.Allocator, source: []const u8, before: []const u8, after: []const u8) ![]u8 {
    if (std.mem.count(u8, source, before) != 1) return error.HybridHcSourceChanged;
    return std.mem.replaceOwned(u8, a, source, before, after);
}
fn function(source: []const u8, name: []const u8) ![]const u8 {
    const at = std.mem.indexOf(u8, source, name) orelse return error.HybridHcSourceChanged;
    if (std.mem.indexOfPos(u8, source, at + name.len, name) != null) return error.HybridHcSourceChanged;
    const start = if (std.mem.lastIndexOfScalar(u8, source[0..at], '\n')) |i| i + 1 else 0;
    const open = std.mem.indexOfScalarPos(u8, source, at, '{') orelse return error.HybridHcSourceChanged;
    var depth: usize = 1;
    var end = open + 1;
    while (end < source.len and depth > 0) : (end += 1) {
        if (source[end] == '{') depth += 1;
        if (source[end] == '}') depth -= 1;
    }
    if (depth != 0) return error.HybridHcSourceChanged;
    return source[start..end];
}
pub fn metal(a: std.mem.Allocator, recorded: []const u8, prompt: []const u8, hidden: usize, streams: usize) ![]u8 {
    if (hidden == 0 or hidden % 256 != 0 or hidden > std.math.maxInt(i32) / 8 or streams < 2 or streams > 8) return error.HybridHcGeometry;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    const original = "custom_kernel_q4_hc_norm_none_4e0640a899baccfe_bfloat16_t_bfloat16_t_float";
    var sums = try once(mem, try function(recorded, original), original, "qwen4_hc_sums");
    sums = try once(mem, sums, "constexpr int S = 4;", try std.fmt.allocPrint(mem, "constexpr int S = {d};", .{streams}));
    sums = try once(mem, sums, "constexpr int D = 2560;", try std.fmt.allocPrint(mem, "constexpr int D = {d};", .{hidden}));
    sums = try once(mem, sums, "HN[r * W + e] = bfloat(hv);", "");
    const geometry = try std.fmt.allocPrint(mem, "constexpr int S = {d}, D = {d}, W = S * D;", .{ streams, hidden });
    var norm = try once(mem, try function(prompt, "void pf_hc_normed("), "pf_hc_normed(", "qwen4_hc_normed(");
    norm = try once(mem, norm, "constexpr int S = 4, D = 2560, W = S * D;", geometry);
    var mix = try once(mem, try function(prompt, "void pf_hc_mix("), "pf_hc_mix(", "qwen4_hc_mix(");
    mix = try once(mem, mix, "constexpr int S = 4, D = 2560, W = S * D;", geometry);
    const act = try once(mem, try function(prompt, "void pf_act("), "pf_act(", "qwen4_dense_act(");
    const body = try std.mem.concat(mem, u8, &.{ sums, "\n", norm, "\n", mix, "\n", act });
    const reserved = try std.mem.replaceOwned(u8, mem, body, "[[kernel]]", "[[max_total_threads_per_threadgroup(256)]] [[kernel]]");
    const silu = try once(mem, try function(prompt, "float pf_bsilu("), "return float(bfloat(x / (1.0f + metal::exp(-x))));", "const float e = metal::exp(-metal::abs(x)); return float(bfloat((x < 0 ? x * e : x) / (1.0f + e)));");
    return std.mem.concat(a, u8, &.{ "#include <metal_stdlib>\nusing namespace metal;\ntypedef bfloat bfloat16_t;\n", try function(prompt, "float pf_bsig("), "\n", silu, "\n", try function(prompt, "float pf_rinv("), "\n", reserved });
}
