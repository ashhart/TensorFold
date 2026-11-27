//! One planted fault for each way the fused steered visit output could leave the separately ordered reference.
const std = @import("std");
const steer = @import("inkling2_metal").steer;

/// The add contracted into one FMA, a NaN dropped, the residual left unrounded, or the direction read one element off.
pub const Variant = enum { production, contracted, nan, unrounded, shifted };

const hv_line = "    const float hv_ar = float(bfloat(pin_row[d] + float(conv_row[d])));";
const step_line = "    const float step_ar = a.alpha * dir[d];";
const sum_line = "    const bfloat next_p = step_ar == 0.0f ? bfloat(hv_ar) : bfloat(hv_ar + step_ar);";

/// The production source with the fault a control plants; each anchor occurs once.
pub fn source(a: std.mem.Allocator, variant: Variant) ![]u8 {
    const e: [2][]const u8 = switch (variant) {
        .production => return a.dupe(u8, steer),
        .contracted => .{ sum_line, "    const bfloat next_p = step_ar == 0.0f ? bfloat(hv_ar) : bfloat(fma(a.alpha, dir[d], hv_ar));" },
        .nan => .{ hv_line, "    const float hv_ar = isnan(pin_row[d]) ? 0.0f : float(bfloat(pin_row[d] + float(conv_row[d])));" },
        .unrounded => .{ hv_line, "    const float hv_ar = pin_row[d] + float(conv_row[d]);" },
        .shifted => .{ step_line, "    const float step_ar = a.alpha * dir[(d + 1) % 2304];" },
    };
    if (std.mem.count(u8, steer, e[0]) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, steer, e[0], e[1]);
}

test "every fused steer control applies once" {
    const a = std.testing.allocator;
    for (std.enums.values(Variant)) |v| {
        const text = try source(a, v);
        defer a.free(text);
        try std.testing.expectEqual(v == .production, std.mem.eql(u8, text, steer));
    }
}
