//! Count-checked mutations of the tensor-unit kernels: every entry has a genuine arithmetic control.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    nv_gu_phi,
    nv_down_phi,
    mx_gu_scale,
    mx_down_scale,
    logits_cap,
    dense_store,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .nv_gu_phi => "control: metal inkling2 units tf_ink2_nvfp4_gu_units phi",
            .nv_down_phi => "control: metal inkling2 units tf_ink2_nvfp4_down_units phi",
            .mx_gu_scale => "control: metal inkling2 units tf_ink2_mxfp8_gu_units scale",
            .mx_down_scale => "control: metal inkling2 units tf_ink2_mxfp8_down_units scale",
            .logits_cap => "control: metal inkling2 units tf_ink2_logits_units cap",
            .dense_store => "control: metal inkling2 units tf_ink2_dense_units store",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .nv_gu_phi => .{ .from = "ink2_units<false, true>(X, W, S, ink2_phi(amax, denominator, tg.y),", .to = "ink2_units<false, true>(X, W, S, 2.0f * ink2_phi(amax, denominator, tg.y)," },
            .nv_down_phi => .{ .from = "ink2_units<false, false>(X, W, S, ink2_phi(amax, denominator, tg.y),", .to = "ink2_units<false, false>(X, W, S, 2.0f * ink2_phi(amax, denominator, tg.y)," },
            // Both MXFP8 entries decode through one helper; a control compiles only its selected entry.
            .mx_gu_scale, .mx_down_scale => .{ .from = "const int shift = int(scales[b / 2u]) - 127;", .to = "const int shift = int(scales[b / 2u]) - 126;" },
            .logits_cap => .{ .from = "precise::divide(float(bfloat(dot)), 30.0f)", .to = "precise::divide(float(bfloat(dot)), 15.0f)" },
            .dense_store => .{ .from = "Y[row * a.n + v] = bfloat(dot);", .to = "Y[row * a.n + v] = bfloat(2.0f * dot);" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.experts_units);
    const change = m.edit();
    if (std.mem.count(u8, src.experts_units, change.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.experts_units, change.from, change.to);
}

test "every units control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 units "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.experts_units, plain);
}

test "units source exports all four entries after the decode helpers" {
    inline for ([_][]const u8{ "tf_ink2_nvfp4_gu_units", "tf_ink2_nvfp4_down_units", "tf_ink2_mxfp8_gu_units", "tf_ink2_mxfp8_down_units" }) |name|
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.experts_units, name ++ "("));
    try std.testing.expect(std.mem.indexOf(u8, src.experts_units, "never quantize an activation") != null);
}
