//! Neutral table identities for resident separate arrays and packed per-rank rows.
const std = @import("std");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;
pub const Plan = struct {
    layer: u32,
    rows: u64,
    weight: Info,
    scale: Info,
    pub fn fromPack(pack: *const Pack, layer: u32, rows: u64) !Plan {
        if (rows == 0 or rows > std.math.maxInt(u32)) return error.InvalidEngramTable;
        var buf: [128]u8 = undefined;
        const weight = try pack.need(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.weight", .{layer}));
        const scale = try pack.need(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.scale", .{layer}));
        if (!weight.is(.f8_e4m3, &.{ @intCast(rows), 256 }) or !scale.is(.f8_e8m0, &.{ @intCast(rows), 8 }))
            return error.InvalidEngramTable;
        return .{ .layer = layer, .rows = rows, .weight = weight, .scale = scale };
    }
};
test "separate rowwise scales use eight entries per row, not a matrix scale grid" {
    const a = std.testing.allocator;
    var p = Pack.init(a);
    defer p.deinit();
    try p.addHeader("table", "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4,256],\"data_offsets\":[0,1024]},\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[4,8],\"data_offsets\":[1024,1056]}}", 1056);
    _ = try Plan.fromPack(&p, 1, 4);
    try std.testing.expectError(error.InvalidEngramTable, Plan.fromPack(&p, 1, 5));
}
