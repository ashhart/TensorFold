//! The two-column geometry's fixed K loop, derived from the captured source without changing other instances.
const std = @import("std");

fn replace(comptime text: []const u8, comptime from: []const u8, comptime to: []const u8) []const u8 {
    return comptime blk: {
        var out: [std.mem.replacementSize(u8, text, from, to)]u8 = undefined;
        _ = std.mem.replace(u8, text, from, to, &out);
        const fixed = out;
        break :blk &fixed;
    };
}

/// A separate kernel template for RPS=2; the captured kernel and every other geometry remain intact.
pub fn body(comptime source: []const u8, comptime function: []const u8) []const u8 {
    @setEvalBranchQuota(200_000);
    const start = std.mem.indexOf(u8, source, "template <int K, int GS, int RPS>").?;
    const end = std.mem.indexOf(u8, source, "\ntemplate [[host_name").?;
    const rows = replace(source[start..end], "tf_rowdot", "tf_geo_rowdot");
    const named = replace(rows, function, "tf_geo_experts");
    const bounded = replace(named, "[[kernel]] void ", "[[max_total_threads_per_threadgroup(SG * 32)]]\n[[kernel]] void ");
    const loop = "  for (int k0 = 0; k0 < FULL; k0 += 512) {";
    if (std.mem.count(u8, bounded, loop) != 1) @compileError("captured geometry K loop changed");
    return "\n" ++ replace(bounded, loop, "  #pragma clang loop unroll(full)\n" ++ loop) ++ "\n";
}
