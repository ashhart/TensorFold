//! Planted controls for Ling MLA extras: one source edit a kernel, which gate 3 must report.
const std = @import("std");
const contract = @import("contract");

pub const Which = enum { rope_cache, rope_q, gate_expand };

pub fn function(w: Which) []const u8 {
    return switch (w) {
        .rope_cache => "tf_ling_mla_rope_cache",
        .rope_q => "tf_ling_mla_rope_q",
        .gate_expand => "tf_ling_mla_gate_expand",
    };
}

pub const Mutation = enum {
    none,
    rope_cache_cs0,
    rope_q_cs0,
    gate_head0,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .rope_cache_cs0 => "control: metal ling mla rope_cache, every row rotates at cs[0]",
            .rope_q_cs0 => "control: metal ling mla rope_q, every row rotates at cs[0]",
            .gate_head0 => "control: metal ling mla gate_expand, every head takes gate 0",
        };
    }

    pub fn gate(m: Mutation) ?contract.gates.Gate {
        return switch (m) {
            .none => null,
            else => .shapes,
        };
    }

    pub fn which(m: Mutation) Which {
        return switch (m) {
            .none => unreachable,
            .rope_cache_cs0 => .rope_cache,
            .rope_q_cs0 => .rope_q,
            .gate_head0 => .gate_expand,
        };
    }

    const Edit = struct { from: []const u8, to: []const u8 };

    fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .rope_cache_cs0 => .{
                .from = "const float2 o = ling_apply(float(dst[2 * t]), float(dst[2 * t + 1]), cs, row, t);",
                .to = "const float2 o = ling_apply(float(dst[2 * t]), float(dst[2 * t + 1]), cs, 0, t);",
            },
            .rope_q_cs0 => .{
                .from = "const float2 o = ling_apply(float(z[2 * t]), float(z[2 * t + 1]), cs, row, t);",
                .to = "const float2 o = ling_apply(float(z[2 * t]), float(z[2 * t + 1]), cs, 0, t);",
            },
            .gate_head0 => .{
                .from = "g2[row * a.gate_stride + col] = g[row * a.heads + col / VD];",
                .to = "g2[row * a.gate_stride + col] = g[row * a.heads];",
            },
        };
    }
};

pub fn apply(a: std.mem.Allocator, text: []const u8, m: Mutation) ![]u8 {
    if (m == .none) return a.dupe(u8, text);
    const e = m.edit();
    if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, text, e.from, e.to);
}

test "ling mla source control: each edit hits once" {
    const a = std.testing.allocator;
    const plain = @import("ling_metal").mla;
    for (std.enums.values(Mutation)) |m| {
        if (m == .none) continue;
        const edited = try apply(a, plain, m);
        defer a.free(edited);
        try std.testing.expect(std.mem.indexOf(u8, edited, m.edit().to) != null);
        try std.testing.expectError(error.ControlEditMissing, apply(a, edited, m));
        try std.testing.expectEqual(contract.gates.Gate.shapes, m.gate().?);
    }
    const unchanged = try apply(a, plain, .none);
    defer a.free(unchanged);
    try std.testing.expectEqualStrings(plain, unchanged);
}
