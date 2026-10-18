const std = @import("std");
const block = @import("block.zig");
pub fn parseDense(a: std.mem.Allocator, w: *block.Widths, text: []const u8) !void {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line[0] == '#') continue;
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next() orelse return error.BadListing;
        try w.dense.put(a, try a.dupe(u8, name), try std.fmt.parseInt(u32, f.next() orelse return error.BadListing, 10));
    }
}

pub fn parseExperts(a: std.mem.Allocator, w: *block.Widths, text: []const u8) !void {
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line[0] == '#') continue;
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        var v: [5]u32 = undefined;
        for (&v) |*x| x.* = try std.fmt.parseInt(u32, f.next() orelse return error.BadListing, 10);
        try w.experts.put(a, v[0], .{ .{ v[1], v[2] }, .{ v[3], v[4] } });
    }
}
