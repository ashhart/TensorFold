//! Count-checked mutations of the simdgroup-matrix expert kernels: each plants a different kind of fault.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    nv_gu_phi,
    nv_down_passes,
    mx_gu_scale,
    mx_down_chains,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .nv_gu_phi => "control: metal inkling2 mma tf_ink2_nvfp4_gu_mma phi",
            .nv_down_passes => "control: metal inkling2 mma tf_ink2_nvfp4_down_mma pass order",
            .mx_gu_scale => "control: metal inkling2 mma tf_ink2_mxfp8_gu_mma scale",
            .mx_down_chains => "control: metal inkling2 mma tf_ink2_mxfp8_down_mma chain order",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            // Both NVFP4 entries share a macro and so an edit; a control compiles only its selected entry.
            .nv_gu_phi => .{ .from = "ink2_mma<true, gu>(X, W, S, ink2_phi(amax, denominator, tg.y),", .to = "ink2_mma<true, gu>(X, W, S, 2.0f * ink2_phi(amax, denominator, tg.y)," },
            .nv_down_passes => .{
                .from = "for (uint s = 1u; s < INK2_MMA_NP; s <<= 1u)\n        _Pragma(\"clang loop unroll(full)\") for (uint p = 0u; p < INK2_MMA_NP; p += 2u * s) sums[j][p] += sums[j][p + s];",
                .to = "for (uint s = INK2_MMA_NP / 2u; s != 0u; s >>= 1u)\n        _Pragma(\"clang loop unroll(full)\") for (uint p = 0u; p < s; ++p) sums[j][p] += sums[j][p + s];",
            },
            .mx_gu_scale => .{ .from = "max(raw.code, 1u) << 23u", .to = "(max(raw.code, 1u) + 1u) << 23u" },
            .mx_down_chains => .{ .from = "sums[j][pass] = (c0 + c1) + (c2 + c3);", .to = "sums[j][pass] = (c0 + c2) + (c1 + c3);" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.experts_mma);
    const change = m.edit();
    if (std.mem.count(u8, src.experts_mma, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.experts_mma, change.from, change.to);
}

test "every simdgroup-matrix expert control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 mma "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.experts_mma, plain);
}

test "the source exports all four entries after the grouped kernels" {
    inline for ([_][]const u8{ "tf_ink2_nvfp4_gu_mma", "tf_ink2_nvfp4_down_mma", "tf_ink2_mxfp8_gu_mma", "tf_ink2_mxfp8_down_mma" }) |name|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.experts_mma, name ++ ","));
    try std.testing.expect(std.mem.indexOf(u8, src.experts_mma, "tf_ink2_nvfp4_gu_grouped") != null);
}
