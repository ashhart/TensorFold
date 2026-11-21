//! One host-checkable arithmetic mutation for each Inkling 2 Mini loop kernel.
const std = @import("std");
const p = @import("inkling2_loop_plan");

const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: p.Kind) Edit {
    return switch (kind) {
        // Drops the required post-dot binary32(1/48) scale.
        .messages => .{ .from = "weights[s] = dot * INKLING_LOOP_SCALE;", .to = "weights[s] = dot;" },
        // Drops the model-dtype rounding between D accumulation and F/Gamma.
        .gates => .{ .from = "zeta[t] = float(bfloat(dot));", .to = "zeta[t] = dot;" },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, kind: p.Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn name(kind: p.Kind) []const u8 {
    return switch (kind) {
        .messages => "control: metal inkling2 loop messages tf_inkling2_loop_messages",
        .gates => "control: metal inkling2 loop gates tf_inkling2_loop_gates",
    };
}

test "every loop control applies once and changes only its arithmetic anchor" {
    const production = @import("inkling2_metal").loop;
    const a = std.testing.allocator;
    for (std.enums.values(p.Kind)) |kind| {
        const e = edit(kind);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, production, e.from));
        const changed = try source(a, production, kind);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, production, e.from).?;
        try std.testing.expectEqualStrings(production[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(production[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, kind));
        try std.testing.expectError(error.ControlEditMissing, source(a, "", kind));
    }
}

test "a repeated loop control anchor is ambiguous and refuses to prove a fault" {
    const a = std.testing.allocator;
    const production = @import("inkling2_metal").loop;
    for (std.enums.values(p.Kind)) |kind| {
        const twice = try std.mem.concat(a, u8, &.{ production, production });
        defer a.free(twice);
        try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
    }
}
