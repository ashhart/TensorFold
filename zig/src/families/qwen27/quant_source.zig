//! Qwen27 helper kernel keys and runtime source specialization.
const std = @import("std");
const sources = @import("qwen_runtime_sources");

pub const Key = struct { kind: Kind, n: u32 = 0, k: u32, slices: u32 = 0, tile: u32 = 0, tmr: u32 = 0, edge: bool = false, skip: u32 = 0, cut: [2]u32 = .{ 0, 0 }, bits: u8 = 4, group: u32 = 64, long: bool = false, xs: bool = true, xs_group: u16 = 64, shift: bool = false, cooperative: bool = false, inline_sums: bool = false };
pub const Kind = enum { projection, xsum, norm, input_norm, mlp, mlp_rows, ane_input, mlp_ane, add_ane };

/// The quantized helper specialization compiled by get.
pub fn specializedSource(a: std.mem.Allocator, key: Key) ![]u8 {
    const template = switch (key.kind) {
        .projection => unreachable,
        .xsum => sources.xsum,
        .norm => sources.norm_xs,
        .input_norm => sources.norm_input,
        .mlp, .mlp_rows => sources.mlp_xs,
        .ane_input, .mlp_ane, .add_ane => sources.mlp_ane,
    };
    const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
    const definitions = switch (key.kind) {
        .projection => unreachable,
        .xsum => try std.fmt.allocPrint(a, "constexpr int K={d}, GS=64;", .{key.k}),
        .norm, .input_norm => try std.fmt.allocPrint(a, "constexpr int K={d};{s}", .{ key.k, if (key.shift) "\n#define Q27_SHIFT 1" else "" }),
        .mlp, .mlp_rows => try std.fmt.allocPrint(a, "constexpr int N={d};", .{key.n}),
        .ane_input, .mlp_ane, .add_ane => try std.fmt.allocPrint(a, "constant constexpr int H={d}, N={d}, A={d};", .{ key.k, key.n, key.tile }),
    };
    defer a.free(definitions);
    const eff_xs: u16 = if (!key.xs) 0 else key.xs_group;
    const xs_define = switch (eff_xs) {
        0 => "#define Q27_XS 0\n",
        128 => "#define Q27_XS 128\n",
        else => "#define Q27_XS 64\n",
    };
    return std.mem.concat(a, u8, &.{ xs_define, template[0..mark], definitions, template[mark + "Q27_SPECIALIZE".len ..] });
}

test "specialized source selects zero, 64, and 128 group sums" {
    const a = std.testing.allocator;
    for ([_]struct { xs: bool, xg: u16, want: []const u8 }{
        .{ .xs = false, .xg = 64, .want = "#define Q27_XS 0\n" },
        .{ .xs = true, .xg = 64, .want = "#define Q27_XS 64\n" },
        .{ .xs = true, .xg = 128, .want = "#define Q27_XS 128\n" },
    }) |case| {
        for ([_]Kind{ .norm, .input_norm, .mlp }) |kind| {
            const src = try specializedSource(a, .{ .kind = kind, .k = 4096, .n = 12288, .xs = case.xs, .xs_group = case.xg });
            defer a.free(src);
            try std.testing.expect(std.mem.startsWith(u8, src, case.want));
        }
    }
}
