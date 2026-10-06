//! Speed-up mode's settings file: this rank, the MCDMA library and the link to the peer.
const std = @import("std");
const fabric = @import("fabric");
const stagger = @import("../../core/stagger.zig");

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
        const device = try joined(gpa, o, "devices", "device");
        const parts = fabric.verbs.deviceParts(gpa, device) catch return error.BadTpSettings;
        defer gpa.free(parts);
        const via = try joined(gpa, o, "vias", "via");
        const vias = std.mem.count(u8, via, "+") + 1;
        if (vias != 1 and vias != parts.len) return error.BadTpSettings;
        var vi = std.mem.splitScalar(u8, via, '+');
        while (vi.next()) |one| if (one.len == 0) return error.BadTpSettings;
        const ports = try portList(gpa, o, "ports", parts.len);
        const peer_ports = try portList(gpa, o, "peer_ports", parts.len);
        const port = if (ports.len > 0) ports[0] else try int(u16, o, "port", null);
        const peer_port = try int(u16, o, "peer_port", 0);
        if (port == 0 or (ports.len == 0 and @as(usize, port) + parts.len > 65536) or
            (peer_ports.len == 0 and peer_port > 0 and @as(usize, peer_port) + parts.len > 65536)) return error.BadTpSettings;
        const gid = try int(c_int, o, "gid", if (parts.len > 2) -1 else 1);
        if (gid < -1) return error.BadTpSettings;
        l.* = .{ .peer = try int(u32, o, "peer", null), .device = device, .via = via, .port = port, .peer_port = peer_port, .name = try text(gpa, o, "name"), .gid = gid, .ports = ports, .peer_ports = peer_ports };
    }
    return .{ .rank = try int(u32, root.object, "rank", null), .library = try text(gpa, root.object, "library"), .links = links };
}

/// Rows each Mac takes before a prompt call splits across the pair (FZ_PAIR_MIN): positive and a u32 (the request head's word), else `default`.
pub fn pairMin(env: ?[]const u8, default: u32) u32 {
    const v = std.fmt.parseInt(u32, env orelse return default, 10) catch return default;
    return if (v > 0) v else default;
}

/// Arrays describe physical members of one peer; the old plus-joined string retains its meaning.
fn joined(gpa: std.mem.Allocator, o: std.json.ObjectMap, plural: []const u8, singular: []const u8) ![:0]const u8 {
    const list = o.get(plural) orelse return text(gpa, o, singular);
    if (o.contains(singular) or list != .array or list.array.items.len == 0 or list.array.items.len > fabric.verbs.max_links) return error.BadTpSettings;
    const items = try gpa.alloc([]const u8, list.array.items.len);
    defer gpa.free(items);
    for (items, list.array.items) |*out, v| {
        if (v != .string or v.string.len == 0 or std.mem.indexOfAny(u8, v.string, "+\x00") != null) return error.BadTpSettings;
        out.* = v.string;
    }
    const value = try std.mem.join(gpa, "+", items);
    defer gpa.free(value);
    return gpa.dupeSentinel(u8, value, 0);
}

fn portList(gpa: std.mem.Allocator, o: std.json.ObjectMap, key: []const u8, n: usize) ![]const u16 {
    const v = o.get(key) orelse return &.{};
    if (v != .array or v.array.items.len != n) return error.BadTpSettings;
    const ports = try gpa.alloc(u16, n);
    for (ports, v.array.items, 0..) |*p, item, i| {
        if (item != .integer) return error.BadTpSettings;
        p.* = std.math.cast(u16, item.integer) orelse return error.BadTpSettings;
        if (p.* == 0) return error.BadTpSettings;
        for (ports[0..i]) |old| if (old == p.*) return error.BadTpSettings;
    }
    return ports;
}

fn int(comptime T: type, o: std.json.ObjectMap, key: []const u8, default: ?T) !T {
    const v = o.get(key) orelse return default orelse error.BadTpSettings;
    if (v != .integer) return error.BadTpSettings;
    return std.math.cast(T, v.integer) orelse error.BadTpSettings;
}

fn text(gpa: std.mem.Allocator, o: std.json.ObjectMap, key: []const u8) ![:0]const u8 {
    const v = o.get(key) orelse return error.BadTpSettings;
    if (v != .string) return error.BadTpSettings;
    if (std.mem.indexOfScalar(u8, v.string, 0) != null) return error.BadTpSettings;
    return gpa.dupeSentinel(u8, v.string, 0);
}

test "speed-up settings accept four devices, one meeting via, explicit ports and automatic per-device GIDs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try read(arena.allocator(),
        \\{"rank":0,"library":"/opt/mcdma/libmcdma-fabric.dylib","links":[{"peer":1,"devices":["rdma_en4","rdma_en3","rdma_en2","rdma_en13"],"via":"en4/192.0.2.2","ports":[7400,7410,7420,7430],"peer_ports":[7500,7510,7520,7530],"name":"pair"}]}
    );
    try std.testing.expectEqualStrings("rdma_en4+rdma_en3+rdma_en2+rdma_en13", s.links[0].device);
    try std.testing.expectEqual(@as(c_int, -1), s.links[0].gid);
    try std.testing.expectEqualSlices(u16, &.{ 7400, 7410, 7420, 7430 }, s.links[0].ports);
    try std.testing.expectEqualSlices(u16, &.{ 7500, 7510, 7520, 7530 }, s.links[0].peer_ports);
}

test "speed-up settings validate N devices, vias, ports and port-range overflow" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{
        "\"devices\":[\"a\",\"a\"],\"via\":\"en4/192.0.2.2\",\"port\":7400",
        "\"devices\":[\"a\",\"b\",\"c\"],\"vias\":[\"en3\",\"en4\"],\"port\":7400",
        "\"devices\":[\"a\",\"b\",\"c\"],\"via\":\"en4\",\"ports\":[7400,7410]",
        "\"devices\":[\"a\",\"b\"],\"via\":\"en4\",\"ports\":[7400,7400]",
        "\"device\":\"a+b\",\"via\":\"en4\",\"port\":65535",
    }) |bad| {
        const json = try std.fmt.allocPrint(a, "{{\"rank\":0,\"library\":\"/opt/mcdma/libmcdma-fabric.dylib\",\"links\":[{{\"peer\":1,\"name\":\"pair\",{s}}}]}}", .{bad});
        try std.testing.expectError(error.BadTpSettings, read(a, json));
    }
    const s = try read(a,
        \\{"rank":1,"library":"/opt/mcdma/libmcdma-fabric.dylib","links":[{"peer":0,"devices":["a","b","c"],"vias":["en3","en4","en2"],"ports":[7400,7401,7402],"name":"pair"}]}
    );
    try std.testing.expectEqualStrings("en3+en4+en2", s.links[0].via);
}

test "speed-up settings read through JSON values, with the link defaults" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const s = try read(arena.allocator(),
        \\{"rank":1,"library":"/opt/mcdma/libmcdma-fabric.dylib","links":[{"peer":0,"device":"rdma_en4+rdma_en3","via":"en4/192.0.2.2+en3/198.51.100.2","port":7490,"name":"tffntp"}]}
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

test "a pair minimum is positive and fits the request head" {
    try std.testing.expectEqual(@as(u32, 256), pairMin(null, 256));
    try std.testing.expectEqual(@as(u32, 256), pairMin("0", 256));
    try std.testing.expectEqual(@as(u32, 256), pairMin("-1", 256));
    try std.testing.expectEqual(@as(u32, 256), pairMin("4294967296", 256));
    try std.testing.expectEqual(@as(u32, 4294967295), pairMin("4294967295", 256));
    try std.testing.expectEqual(@as(u32, 128), pairMin("128", 256));
    try std.testing.expectEqual(@as(usize, 2), stagger.next(96, 2048, 0).parts); // zero would split the 96-token warm-up
    try std.testing.expectEqual(@as(usize, 1), stagger.next(96, 2048, pairMin("0", 256)).parts);
    try std.testing.expectEqual(@as(usize, 1), stagger.next(96, 2048, pairMin("4294967295", 256)).parts);
}
