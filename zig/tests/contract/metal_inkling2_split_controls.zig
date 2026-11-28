//! The split router's and gates' planted controls: a dot summed in two strided chains, and a result one index on.
const std = @import("std");

pub const Control = enum { route_strided, route_index, down_strided, down_index, apply_blend };

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(c: Control) Edit {
    return switch (c) {
        .route_strided => .{
            .from = "  for (uint j = 0; j < 2304 / 4; ++j) {\n    const float4 wv = float4(w4[j]);\n    const float4 xv = float4(v4[j]);\n    dot += wv.x * xv.x; dot += wv.y * xv.y; dot += wv.z * xv.z; dot += wv.w * xv.w;\n  }",
            .to = "  float odd = 0.0f; for (uint j = 0; j < 2304 / 4; ++j) { const float4 wv = float4(w4[j]), xv = float4(v4[j]); dot += wv.x * xv.x; odd += wv.y * xv.y; dot += wv.z * xv.z; odd += wv.w * xv.w; } dot += odd;",
        },
        .route_index => .{ .from = "scores[size_t(row) * 128 + e] = ink_sigmoid(dot);", .to = "scores[size_t(row) * 128 + (e + 1) % 128] = ink_sigmoid(dot);" },
        .down_strided => .{
            .from = "  for (uint j = 0; j < (2 * INKLING_LOOP_H) / 4; ++j) {\n    const float4 wv = float4(w4[j]);\n    const float4 xv = float4(x4[j]);\n    dot += wv.x * xv.x; dot += wv.y * xv.y; dot += wv.z * xv.z; dot += wv.w * xv.w;\n  }",
            .to = "  float odd = 0.0f; for (uint j = 0; j < (2 * INKLING_LOOP_H) / 4; ++j) { const float4 wv = float4(w4[j]), xv = float4(x4[j]); dot += wv.x * xv.x; odd += wv.y * xv.y; dot += wv.z * xv.z; odd += wv.w * xv.w; } dot += odd;",
        },
        .down_index => .{
            .from = "zeta[size_t(row) * INKLING_LOOP_RANK + r] = float(bfloat(dot));",
            .to = "zeta[size_t(row) * INKLING_LOOP_RANK + (r + 1) % INKLING_LOOP_RANK] = float(bfloat(dot));",
        },
        .apply_blend => .{
            .from = "fdot += fv.x * zv.x; fdot += fv.y * zv.y; fdot += fv.z * zv.z; fdot += fv.w * zv.w;",
            .to = "fdot += fv.x * zv.x; fdot += fv.z * zv.z; fdot += fv.y * zv.y; fdot += fv.w * zv.w;",
        },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, c: Control) ![]u8 {
    const e = edit(c);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(c: Control) []const u8 {
    return switch (c) {
        .route_strided => "control: metal inkling2 glue router split, a two-chain strided score dot",
        .route_index => "control: metal inkling2 glue router split, each score stored one expert on",
        .down_strided => "control: metal inkling2 loop gates split, a two-chain strided down dot",
        .down_index => "control: metal inkling2 loop gates split, each zeta stored one rank on",
        .apply_blend => "control: metal inkling2 loop gates split, reordered forget dot in loop_apply",
    };
}

pub fn glue(c: Control) bool {
    return c == .route_strided or c == .route_index;
}

test "every split control applies once to its own source and changes only its anchor" {
    const a = std.testing.allocator;
    const sources = @import("inkling2_metal");
    for (std.enums.values(Control)) |c| {
        const production = if (glue(c)) sources.glue else sources.loop;
        const e = edit(c);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, c);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, c));
        try std.testing.expectError(error.ControlEditMissing, source(a, if (glue(c)) sources.loop else sources.glue, c));
    }
}
