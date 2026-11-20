//! Resolve each specialized Qwen27 stack record to the source its production loader compiles.
const std = @import("std");
const sources = @import("kernel_sources");
pub fn source(a: std.mem.Allocator, name: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, name, "q27 stack qk ")) {
        var words = std.mem.tokenizeScalar(u8, name[13..], ' ');
        const sk = try std.fmt.parseInt(u32, words.next().?[2..], 10);
        const rows = try std.fmt.parseInt(u32, words.next().?[4..], 10);
        return std.fmt.allocPrint(a, "#define TF_SK {d}\n#define TF_ROWS {d}\n{s}", .{ sk, rows, sources.core_qk });
    }
    const xs = std.mem.indexOf(u8, name, "xs1") != null;
    const body = try std.mem.replaceOwned(u8, a, sources.q27_mlp_xs, "Q27_SPECIALIZE", "constexpr int N=17408;");
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define Q27_XS {d}\n{s}", .{ @intFromBool(xs), body });
}
