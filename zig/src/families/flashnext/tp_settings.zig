//! Speed-up mode's settings file: this rank, the MCDMA library and the link to the peer.
const std = @import("std");
const fabric = @import("fabric");

pub const Settings = struct { rank: u32, library: []const u8, links: []const fabric.mcdma.Link };

/// The settings file read through JSON values: a typed parse links compiler-rt's quad floats, and with them its memcpy over libSystem's.
pub fn read(gpa: std.mem.Allocator, bytes: []const u8) !Settings {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always });
    if (root != .object) return error.BadTpSettings;
    const list = root.object.get("links") orelse return error.BadTpSettings;
    if (list != .array) return error.BadTpSettings;
    const links = try gpa.alloc(fabric.mcdma.Link, list.array.items.len);
    for (links, list.array.items) |*l, v| {
        if (v != .object) return error.BadTpSettings;
        const o = v.object;
        l.* = .{ .peer = try int(u32, o, "peer", null), .device = try text(gpa, o, "device"), .via = try text(gpa, o, "via"), .port = try int(u16, o, "port", null), .peer_port = try int(u16, o, "peer_port", 0), .name = try text(gpa, o, "name"), .gid = try int(c_int, o, "gid", 1) };
    }
    return .{ .rank = try int(u32, root.object, "rank", null), .library = try text(gpa, root.object, "library"), .links = links };
}

fn int(comptime T: type, o: std.json.ObjectMap, key: []const u8, default: ?T) !T {
    const v = o.get(key) orelse return default orelse error.BadTpSettings;
    if (v != .integer) return error.BadTpSettings;
    return std.math.cast(T, v.integer) orelse error.BadTpSettings;
}

fn text(gpa: std.mem.Allocator, o: std.json.ObjectMap, key: []const u8) ![:0]const u8 {
    const v = o.get(key) orelse return error.BadTpSettings;
    if (v != .string) return error.BadTpSettings;
    return gpa.dupeSentinel(u8, v.string, 0);
}

test "speed-up settings read through JSON values, with the link defaults" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try read(arena.allocator(),
        \\{"rank":1,"library":"/opt/mcdma/libmcdma-fabric.dylib","links":[{"peer":0,"device":"rdma_en4+rdma_en3","via":"en4/10.0.0.2+en3/10.0.1.2","port":7490,"name":"tffntp"}]}
    );
    try std.testing.expectEqual(@as(u32, 1), s.rank);
    try std.testing.expectEqualStrings("/opt/mcdma/libmcdma-fabric.dylib", s.library);
    try std.testing.expectEqual(@as(usize, 1), s.links.len);
    try std.testing.expectEqualStrings("rdma_en4+rdma_en3", s.links[0].device);
    try std.testing.expectEqual(@as(u16, 7490), s.links[0].port);
    try std.testing.expectEqual(@as(u16, 0), s.links[0].peer_port);
    try std.testing.expectEqual(@as(c_int, 1), s.links[0].gid);
    try std.testing.expectError(error.BadTpSettings, read(arena.allocator(), "{\"rank\":1.5,\"library\":\"x\",\"links\":[]}"));
}
