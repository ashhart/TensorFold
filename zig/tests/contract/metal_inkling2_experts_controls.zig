//! Count-checked M3 source mutations: every shader entry has a genuine arithmetic control.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    nv_gu_up,
    nv_down_phi,
    mx_gu_up,
    mx_down_scale,
    nv_decode_nibble,
    mx_decode_scale,
    combine_slot_order,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .nv_gu_up => "control: metal inkling2 experts tf_ink2_nvfp4_gu up row",
            .nv_down_phi => "control: metal inkling2 experts tf_ink2_nvfp4_down phi",
            .mx_gu_up => "control: metal inkling2 experts tf_ink2_mxfp8_gu up row",
            .mx_down_scale => "control: metal inkling2 experts tf_ink2_mxfp8_down scale",
            .nv_decode_nibble => "control: metal inkling2 experts tf_ink2_nvfp4_quantize nibble",
            .mx_decode_scale => "control: metal inkling2 experts tf_ink2_mxfp8_quantize scale",
            .combine_slot_order => "control: metal inkling2 experts tf_ink2_combine route-slot order",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .nv_gu_up => .{ .from = "const uint gr = 2u * n, ur = gr + 1u;", .to = "const uint gr = 2u * n, ur = gr;" },
            .mx_gu_up => .{ .from = "const uint gr = 2u * n, ur = gr + 1u;", .to = "const uint gr = 2u * n, ur = gr;" },
            .nv_down_phi => .{ .from = "const float phi = ink2_phi(amax, denominator, expert);", .to = "const float phi = precise::divide(2.0f * amax[expert], denominator[0]);" },
            .nv_decode_nibble => .{ .from = "W[row * (a.k / 2u) + col / 2u], col,", .to = "W[row * (a.k / 2u) + col / 2u], col + 1u," },
            .mx_down_scale => .{ .from = "S + expert_s + ulong(n) * (a.k / 32u), X + ulong(group) * a.k, a.k, lane);", .to = "S + expert_s + ulong(n) * (a.k / 16u), X + ulong(group) * a.k, a.k, lane);" },
            .mx_decode_scale => .{ .from = "S[row * (a.k / 32u) + col / 32u]", .to = "S[row * (a.k / 32u) + col / 16u]" },
            .combine_slot_order => .{ .from = "if (rank == r && uint(id) < a.experts)", .to = "if (slot == r && uint(id) < a.experts)" },
        };
    }
};

pub fn apply(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, src.experts);
    const change = m.edit();
    const expected: usize = switch (m) {
        .nv_gu_up, .mx_gu_up, .nv_down_phi => 2,
        else => 1,
    };
    if (std.mem.count(u8, src.experts, change.from) != expected) return error.ControlEditCount;
    // The same textual mutation serves both formats; a control compiles only its selected entry.
    return std.mem.replaceOwned(u8, a, src.experts, change.from, change.to);
}

test "every expert control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
        try std.testing.expect(std.mem.startsWith(u8, m.name(), "control: metal inkling2 experts "));
    }
    const plain = try apply(a, .none);
    defer a.free(plain);
    try std.testing.expectEqualStrings(src.experts, plain);
}

test "expert source exports all seven entries and has no activation quantizer" {
    for ([_][]const u8{ "tf_ink2_nvfp4_gu", "tf_ink2_nvfp4_down", "tf_ink2_mxfp8_gu", "tf_ink2_mxfp8_down", "tf_ink2_nvfp4_quantize", "tf_ink2_mxfp8_quantize", "tf_ink2_combine" }) |name|
        try std.testing.expect(std.mem.indexOf(u8, src.experts, name) != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "never quantize an activation") != null);
    try std.testing.expect(std.mem.indexOf(u8, src.experts, "e2m1(float") == null);
}
