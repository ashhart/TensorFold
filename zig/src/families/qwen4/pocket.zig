//! The shared dense publication owner loads Qwen4's native checkpoint namespace and immutable base columns.
const std = @import("std");
const cfg = @import("config.zig");
const matrix = @import("pocket_matrix.zig");
pub const neurons = 64;
const Loader = struct {
    pub const shape = error.Qwen4PocketShape;
    pub const nonfinite = error.Qwen4PocketNonfinite;
    pub const budget = error.Qwen4PocketBudget;
    pub fn load(p: anytype, reader: *@import("reader.zig").Reader, c: *const cfg.Config) !void {
        if (c.learned == 0) return;
        if (c.learned != neurons or c.layers != p.layers.len or c.hidden != p.hidden) return error.Qwen4PocketShape;
        const n = neurons * c.hidden;
        const values = try p.a.alloc(f32, 3 * n);
        defer p.a.free(values);
        for (0..c.layers) |layer| {
            inline for (std.enums.values(matrix.Role), 0..) |r, i| {
                var buf: [256]u8 = undefined;
                const projection = if (r == .gate) "gate_proj" else if (r == .up) "up_proj" else "down_proj";
                const stem = try std.fmt.bufPrint(&buf, "{s}layers.{d}.mlp.{s}", .{ reader.namespace.prefix(), layer, projection });
                var parts: [3]@import("core").safetensors.Tensor = undefined;
                for (&parts, [_][]const u8{ "weight", "scales", "biases" }) |*part, leaf| {
                    var name: [320]u8 = undefined;
                    part.* = try reader.get(try std.fmt.bufPrint(&name, "{s}.{s}", .{ stem, leaf }));
                    try matrix.admit(c, stem, part.*);
                }
                try matrix.decode(c, r, parts, values[i * n ..][0..n]);
            }
            var used: usize = 0;
            for (0..neurons) |j| if (std.mem.indexOfNone(f32, values[n + j * c.hidden ..][0..c.hidden], &.{0}) != null) {
                used = j + 1;
            };
            try p.set(layer, values[0..n], values[n..][0..n], values[2 * n ..], used);
        }
    }
};
pub fn Pockets(comptime B: type) type {
    return @import("living").dense_pockets.Pockets(B, Loader);
}
