//! One host-checkable arithmetic mutation for every Small DSpark Metal kernel.
const std = @import("std");

pub const Kind = enum { init_rows, fuse_taps, block_qkv, attention, residual_norm, kappa, base };
const Edit = struct { from: []const u8, to: []const u8 };
fn edit(kind: Kind) Edit {
    return switch (kind) {
        .init_rows => .{ .from = "rows[uint64_t(row) * SDSP_H + d] = src[d];", .to = "rows[uint64_t(row) * SDSP_H + d] = mask[d];" },
        .fuse_taps => .{ .from = "alpha[j] = float(bfloat(scores[j] / sum));", .to = "alpha[j] = scores[j] / sum;" },
        .block_qkv => .{ .from = "const float rotated = d < 64u ? x * cs - other * sn : other * sn + x * cs;", .to = "const float rotated = d < 64u ? x * cs - other * sn : x * sn + other * cs;" },
        .attention => .{ .from = "const uint first = max(from, a.local && committed > 1024u ? committed - 1024u : 0u);", .to = "const uint first = from;" },
        .residual_norm => .{ .from = "normalized[at] = bfloat(sigma * scale * float(weight[d]));", .to = "normalized[at] = bfloat(float(new_residual[at]) * scale * float(weight[d]));" },
        .kappa => .{ .from = "const float wy = float(w_kappa[uint64_t(d) * SDSP_RNN + ch]);", .to = "const float wy = float(w_kappa[uint64_t(SDSP_H + d) * SDSP_RNN + ch]);" },
        .base => .{ .from = "static inline void small_dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(val); }", .to = "static inline void small_dspark_store_base(device bfloat* out, uint64_t i, float val) { out[i] = bfloat(30.0f * tanh(val / 30.0f)); }" },
    };
}

pub fn name(kind: Kind) []const u8 {
    return switch (kind) {
        inline else => |k| "control: metal inkling2 small_dspark " ++ @tagName(k),
    };
}
pub fn source(a: std.mem.Allocator, production: []const u8, kind: Kind) ![]u8 {
    const e = edit(kind);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

test "every Small DSpark control applies once and only mutates its arithmetic anchor" {
    const production = @import("inkling2_metal").inkling2_small_dspark;
    const a = std.testing.allocator;
    for (std.enums.values(Kind)) |kind| {
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

test "repeated Small DSpark control anchors are refused" {
    const production = @import("inkling2_metal").inkling2_small_dspark;
    const a = std.testing.allocator;
    const twice = try std.mem.concat(a, u8, &.{ production, production });
    defer a.free(twice);
    for (std.enums.values(Kind)) |kind| try std.testing.expectError(error.ControlEditMissing, source(a, twice, kind));
}
