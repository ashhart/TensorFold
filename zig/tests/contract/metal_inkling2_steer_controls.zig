//! One planted fault for each way the Inkling 2 Mini steered visit output could leave the reference's order.
const std = @import("std");
const steer = @import("inkling2_metal").steer;

/// The add contracted into one FMA, a NaN dropped, the residual left unrounded, or the direction read one element off.
pub const Variant = enum { production, contracted, nan, unrounded, shifted };

const partial_line = "  const float hv = float(bfloat(t[i] + float(x[i])));";
const sum_line = "  p[i] = bfloat(hv + step);";
const step_line = "  const float step = a.alpha * dir[i % 2304];\n  p[i] = bfloat(hv + step);";

/// The production source with the fault a control plants; each anchor occurs once.
pub fn source(a: std.mem.Allocator, variant: Variant) ![]u8 {
    const e: [2][]const u8 = switch (variant) {
        .production => return a.dupe(u8, steer),
        .contracted => .{ sum_line, "  p[i] = bfloat(fma(a.alpha, dir[i % 2304], hv));" },
        .nan => .{ partial_line, "  const float hv = isnan(t[i]) ? 0.0f : float(bfloat(t[i] + float(x[i])));" },
        .unrounded => .{ partial_line, "  const float hv = t[i] + float(x[i]);" },
        .shifted => .{ step_line, "  const float step = a.alpha * dir[(i + 1) % 2304];\n  p[i] = bfloat(hv + step);" },
    };
    if (std.mem.count(u8, steer, e[0]) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, steer, e[0], e[1]);
}

test "every steer control applies once and the production source keeps its ordering pragma" {
    const a = std.testing.allocator;
    for (std.enums.values(Variant)) |v| {
        const text = try source(a, v);
        defer a.free(text);
        try std.testing.expectEqual(v == .production, std.mem.eql(u8, text, steer));
    }
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, steer, "#pragma clang fp contract(off)"));
}
