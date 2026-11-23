//! Count-checked mutations for Inkling 2 DSpark Q2/Q4 screening, draft projection, and GPU sampling kernels.
const std = @import("std");
const src = @import("inkling2_metal");
const g = @import("contract").gates;

pub const Mutation = enum {
    none,
    q2_pack,
    q4_pack,
    q2_proj,
    q2_proj_units,
    q4_units,
    q2_screen,
    q2_refine,
    q2_pick,
    q2_sample,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .q2_pack => "control: metal inkling2 dspark chain_q2_pack bad_scale",
            .q4_pack => "control: metal inkling2 dspark chain_q4_pack bad_scale",
            .q2_proj => "control: metal inkling2 dspark chain_q2_proj drop_qdot",
            .q2_proj_units => "control: metal inkling2 units tf_ink2_dspark_q2_proj_units drop_codes",
            .q4_units => "control: metal inkling2 units tf_ink2_dspark_q4_units drop_codes",
            .q2_screen => "control: metal inkling2 dspark chain_q2_screen drop_beta",
            .q2_refine => "control: metal inkling2 dspark chain_q2_refine drop_beta",
            .q2_pick => "control: metal inkling2 dspark chain_q2_pick bad_tie",
            .q2_sample => "control: metal inkling2 dspark chain_q2_sample drop_gumbel",
        };
    }

    pub fn gate(m: Mutation) ?g.Gate {
        return if (m == .none) null else .shapes;
    }

    pub fn isUnits(m: Mutation) bool {
        return m == .q2_proj_units or m == .q4_units;
    }

    const Edit = struct { from: []const u8, to: []const u8 };
    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .q2_pack => .{ .from = "const float scale = span > 0.0f ? span / 3.0f : 0.0f;", .to = "const float scale = span > 0.0f ? span / 2.0f : 0.0f;" },
            .q4_pack => .{ .from = "const float scale = span > 0.0f ? span / 15.0f : 0.0f;", .to = "const float scale = span > 0.0f ? span / 7.0f : 0.0f;" },
            .q2_proj => .{ .from = "acc[row] += float(sc * qdot + lo * xsum);", .to = "acc[row] += float(lo * xsum);" },
            .q2_proj_units => .{ .from = "const half4 q0 = pr0.y + pr0.x * ink2_q2_unpack4(c0[w] >> shift);", .to = "const half4 q0 = half4(pr0.y);" },
            .q4_units => .{ .from = "const half4 q0 = pr0.y + pr0.x * ink2_q4_unpack4(c0[w_idx] >> shift);", .to = "const half4 q0 = half4(pr0.y);" },
            .q2_screen => .{ .from = "const float raw = float(base_approx[i]) + float(beta[i]) + part;", .to = "const float raw = float(base_approx[i]) + part;" },
            .q2_refine => .{ .from = "const bfloat rbias = bfloat(float(beta[id]) + pdot);", .to = "const bfloat rbias = bfloat(pdot);" },
            .q2_pick => .{ .from = "if (pv > best || (pv == best && pi < at))", .to = "if (pv > best || (pv == best && pi > at))" },
            .q2_sample => .{ .from = "gscore = sv[t] - precise::log(-precise::log(u));", .to = "gscore = sv[t];" },
        };
    }
};

pub fn applyDspark(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none or m.isUnits()) return a.dupe(u8, src.inkling2_dspark);
    const ch = m.edit();
    if (std.mem.count(u8, src.inkling2_dspark, ch.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.inkling2_dspark, ch.from, ch.to);
}

pub fn applyUnits(a: std.mem.Allocator, m: Mutation) ![]u8 {
    if (m == .none or !m.isUnits()) return a.dupe(u8, src.experts_units);
    const ch = m.edit();
    if (std.mem.count(u8, src.experts_units, ch.from) != 1) return error.ControlEditCount;
    return std.mem.replaceOwned(u8, a, src.experts_units, ch.from, ch.to);
}

test "every dspark screen control edits exactly its planted arithmetic" {
    const a = std.testing.allocator;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = if (m.isUnits()) try applyUnits(a, m) else try applyDspark(a, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectEqual(g.Gate.shapes, m.gate().?);
    }
}
