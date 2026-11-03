//! Gemma's GeGLU publications use the common owner and its exact shared affine save grid.
const std = @import("std");
const living = @import("living");
pub const matrix = living.dense_matrix;
const Loader = struct {
    pub const shape = error.DensePocketShape;
    pub const nonfinite = error.DensePocketNonfinite;
    pub const budget = error.DensePocketBudget;
    pub fn load(p: anytype, reader: anytype, c: anytype) !void {
        if (c.learned == 0) return;
        const d = c.geo.hidden;
        const n = 64 * d;
        if (c.learned != 64 or c.geo.layers != p.layers.len or d != p.hidden) return shape;
        const values = try p.a.alloc(f32, 3 * n);
        defer p.a.free(values);
        for (0..c.geo.layers) |layer| {
            inline for (std.enums.values(matrix.Role), 0..) |r, i| {
                var stem: [192]u8 = undefined;
                const prefix = try std.fmt.bufPrint(&stem, "language_model.model.layers.{d}.mlp.{s}", .{ layer, if (r == .gate) "gate_proj" else if (r == .up) "up_proj" else "down_proj" });
                var views: [3]matrix.View = undefined;
                inline for (&views, .{ "weight", "scales", "biases" }) |*v, leaf| {
                    var name: [224]u8 = undefined;
                    v.* = matrix.View.of(try reader.get(try std.fmt.bufPrint(&name, "{s}.{s}", .{ prefix, leaf })));
                }
                try matrix.decode(d, c.geo.intermediate, c.learned, r, views, values[i * n ..][0..n]);
            }
            var used: usize = 0;
            for (0..64) |j| if (std.mem.indexOfNone(f32, values[n + j * d ..][0..d], &.{0}) != null) {
                used = j + 1;
            };
            try p.set(layer, values[0..n], values[n .. 2 * n], values[2 * n ..], used);
        }
    }
};
pub fn Pockets(comptime B: type) type {
    return living.dense_pockets.Pockets(B, Loader);
}
