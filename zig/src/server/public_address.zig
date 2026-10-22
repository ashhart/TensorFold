//! Address and host admission shared by HTTP fetchers, including cloud metadata endpoints.
const std = @import("std");

pub fn publicHost(host: []const u8) bool {
    const name = std.mem.trimEnd(u8, host, ".");
    if (name.len == 0) return false;
    if (std.ascii.endsWithIgnoreCase(name, ".localhost")) return false;
    for ([_][]const u8{ "localhost", "metadata.google.internal", "instance-data" }) |blocked|
        if (std.ascii.eqlIgnoreCase(name, blocked)) return false;
    return true;
}

pub fn publicIp4(bytes: [4]u8) bool {
    const x = std.mem.readInt(u32, &bytes, .big);
    const blocked = [_]struct { [4]u8, u5 }{
        .{ .{ 0, 0, 0, 0 }, 8 },      .{ .{ 10, 0, 0, 0 }, 8 },    .{ .{ 100, 64, 0, 0 }, 10 },   .{ .{ 127, 0, 0, 0 }, 8 },
        .{ .{ 169, 254, 0, 0 }, 16 }, .{ .{ 172, 16, 0, 0 }, 12 }, .{ .{ 192, 0, 0, 0 }, 24 },    .{ .{ 192, 0, 2, 0 }, 24 },
        .{ .{ 192, 168, 0, 0 }, 16 }, .{ .{ 198, 18, 0, 0 }, 15 }, .{ .{ 198, 51, 100, 0 }, 24 }, .{ .{ 203, 0, 113, 0 }, 24 },
        .{ .{ 224, 0, 0, 0 }, 4 },    .{ .{ 240, 0, 0, 0 }, 4 },
    };
    for (blocked) |block| {
        const mask = ~@as(u32, 0) << @intCast(32 - @as(u6, block[1]));
        if (x & mask == std.mem.readInt(u32, &block[0], .big) & mask) return false;
    }
    return !std.mem.eql(u8, &bytes, &.{ 168, 63, 129, 16 });
}

pub fn publicIp6(bytes: [16]u8) bool {
    if (bytes[0] & 0xe0 != 0x20) return false;
    if (bytes[0] == 0x20 and bytes[1] == 0x01 and bytes[2] == 0x0d and bytes[3] == 0xb8) return false;
    if (bytes[0] == 0x20 and bytes[1] == 0x01 and bytes[2] < 0x02) return false;
    return !(bytes[0] == 0x20 and bytes[1] == 0x02);
}

pub fn publicIp(ip: std.Io.net.IpAddress) bool {
    return switch (ip) {
        .ip4 => |v| publicIp4(v.bytes),
        .ip6 => |v| publicIp6(v.bytes),
    };
}

test "Server recovery: shared hostname policy blocks metadata names before resolution" {
    for ([_][]const u8{ "localhost", "API.LOCALHOST.", "metadata.google.internal", "METADATA.GOOGLE.INTERNAL.", "instance-data.", "." }) |host|
        try std.testing.expect(!publicHost(host));
    for ([_][]const u8{ "example.com", "EXAMPLE.COM.", "metadata.google.internal.example.com" }) |host|
        try std.testing.expect(publicHost(host));
}
