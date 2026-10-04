//! The cluster's control messages: a 40-byte little-endian header (magic, kind, sender, addressee, CRC-32) and kind bodies.
const std = @import("std");
const node = @import("node.zig");

pub const magic = "TFCL";
pub const version: u16 = 1;
pub const header_bytes = 40;

pub const Kind = enum(u16) {
    hello = 1,
    beat = 2,
    leave = 3,
    want = 4,
    inventory = 5,
    pull = 16,
    chunk = 17,
    _,
};

pub const Header = struct {
    kind: Kind,
    from: node.NodeId,
    /// The addressee; 0 means whoever is at the other end of the link (never relayed).
    to: node.NodeId = 0,
    len: u32 = 0,
    crc: u32 = 0,
    hops: u8 = 0,
};

pub const Error = error{ Short, Foreign, Corrupt, NoRoom };

/// Little-endian appends into a fixed buffer.
pub const Out = struct {
    buf: []u8,
    at: usize = header_bytes,

    pub fn int(o: *Out, comptime T: type, v: T) Error!void {
        const n = @sizeOf(T);
        if (o.at + n > o.buf.len) return error.NoRoom;
        std.mem.writeInt(T, o.buf[o.at..][0..n], v, .little);
        o.at += n;
    }

    pub fn bytes(o: *Out, b: []const u8) Error!void {
        if (o.at + b.len > o.buf.len) return error.NoRoom;
        @memcpy(o.buf[o.at..][0..b.len], b);
        o.at += b.len;
    }

    /// Write the header over the first 40 bytes and return the whole message.
    pub fn finish(o: *Out, h: Header) []const u8 {
        const body = o.buf[header_bytes..o.at];
        var f = h;
        f.len = @intCast(body.len);
        f.crc = std.hash.Crc32.hash(body);
        packHeader(o.buf[0..header_bytes], f);
        return o.buf[0..o.at];
    }
};

/// Bounds-checked little-endian reads.
pub const In = struct {
    buf: []const u8,
    at: usize = 0,

    pub fn int(i: *In, comptime T: type) Error!T {
        const n = @sizeOf(T);
        if (i.at + n > i.buf.len) return error.Short;
        defer i.at += n;
        return std.mem.readInt(T, i.buf[i.at..][0..n], .little);
    }

    pub fn bytes(i: *In, n: usize) Error![]const u8 {
        if (i.at + n > i.buf.len) return error.Short;
        defer i.at += n;
        return i.buf[i.at..][0..n];
    }
};

fn packHeader(out: *[header_bytes]u8, h: Header) void {
    @memset(out, 0);
    @memcpy(out[0..4], magic);
    std.mem.writeInt(u16, out[4..6], version, .little);
    std.mem.writeInt(u16, out[6..8], @intFromEnum(h.kind), .little);
    std.mem.writeInt(u64, out[8..16], h.from, .little);
    std.mem.writeInt(u64, out[16..24], h.to, .little);
    std.mem.writeInt(u32, out[24..28], h.len, .little);
    std.mem.writeInt(u32, out[28..32], h.crc, .little);
    out[32] = h.hops;
}

/// Check and parse a whole message; the body follows the header.
pub fn open(msg: []const u8) Error!struct { header: Header, body: []const u8 } {
    if (msg.len < header_bytes) return error.Short;
    if (!std.mem.eql(u8, msg[0..4], magic) or std.mem.readInt(u16, msg[4..6], .little) != version) return error.Foreign;
    const h: Header = .{
        .kind = @enumFromInt(std.mem.readInt(u16, msg[6..8], .little)),
        .from = std.mem.readInt(u64, msg[8..16], .little),
        .to = std.mem.readInt(u64, msg[16..24], .little),
        .len = std.mem.readInt(u32, msg[24..28], .little),
        .crc = std.mem.readInt(u32, msg[28..32], .little),
        .hops = msg[32],
    };
    if (msg.len - header_bytes < h.len) return error.Short;
    const body = msg[header_bytes..][0..h.len];
    if (std.hash.Crc32.hash(body) != h.crc) return error.Corrupt;
    return .{ .header = h, .body = body };
}

/// Re-stamp a relayed message's hop count in place (the CRC covers only the body).
pub fn setHops(msg: []u8, hops: u8) void {
    msg[32] = hops;
}

fn name(o: *Out, n: node.Name) Error!void {
    try o.int(u8, n.len);
    try o.bytes(&n.bytes);
}

fn readName(i: *In) Error!node.Name {
    var n: node.Name = .{ .len = try i.int(u8) };
    @memcpy(&n.bytes, try i.bytes(n.bytes.len));
    if (n.len > n.bytes.len) return error.Corrupt;
    return n;
}

pub fn putInventory(o: *Out, inv: *const node.Inventory) Error!void {
    try o.int(u64, inv.id);
    try name(o, inv.name);
    try o.int(u8, @intFromEnum(inv.backend));
    try name(o, inv.chip);
    try o.int(u32, inv.gpu_cores);
    try o.int(u8, @intFromBool(inv.unified));
    for ([_]u64{ inv.memory, inv.gpu_limit, inv.wired_limit_mb, inv.free, inv.bandwidth, inv.disk_free }) |v| try o.int(u64, v);
    try o.int(u8, inv.port_count);
    for (inv.ports()) |p| {
        try name(o, p.device);
        try o.int(u8, @intFromEnum(p.kind));
        try o.int(u8, @intFromBool(p.up));
        try o.int(u32, p.gbps);
        try o.bytes(&p.gid);
        try o.bytes(&p.peer_gid);
    }
}

pub fn getInventory(i: *In) Error!node.Inventory {
    var inv: node.Inventory = .{ .id = try i.int(u64), .name = try readName(i) };
    inv.backend = std.enums.fromInt(node.Backend, try i.int(u8)) orelse return error.Corrupt;
    inv.chip = try readName(i);
    inv.gpu_cores = try i.int(u32);
    inv.unified = try i.int(u8) != 0;
    inv.memory = try i.int(u64);
    inv.gpu_limit = try i.int(u64);
    inv.wired_limit_mb = try i.int(u64);
    inv.free = try i.int(u64);
    inv.bandwidth = try i.int(u64);
    inv.disk_free = try i.int(u64);
    const ports = try i.int(u8);
    if (ports > node.max_ports) return error.Corrupt;
    for (0..ports) |_| {
        var p: node.Port = .{ .device = try readName(i) };
        p.kind = std.enums.fromInt(node.LinkKind, try i.int(u8)) orelse return error.Corrupt;
        p.up = try i.int(u8) != 0;
        p.gbps = try i.int(u32);
        p.gid = (try i.bytes(16))[0..16].*;
        p.peer_gid = (try i.bytes(16))[0..16].*;
        inv.addPort(p);
    }
    return inv;
}

test "headers check their magic, length and CRC" {
    var buf: [256]u8 = undefined;
    var o: Out = .{ .buf = &buf };
    try o.int(u64, 42);
    const msg = o.finish(.{ .kind = .leave, .from = 7, .to = 9, .hops = 2 });
    const got = try open(msg);
    try std.testing.expectEqual(Kind.leave, got.header.kind);
    try std.testing.expect(got.header.from == 7 and got.header.to == 9 and got.header.hops == 2 and got.body.len == 8);
    var bad = buf;
    bad[header_bytes] ^= 1;
    try std.testing.expectError(error.Corrupt, open(bad[0..msg.len]));
    try std.testing.expectError(error.Short, open(msg[0 .. msg.len - 1]));
    bad = buf;
    bad[0] = 'X';
    try std.testing.expectError(error.Foreign, open(bad[0..msg.len]));
}

test "an inventory round-trips through its encoding" {
    var inv: node.Inventory = .{ .id = 99, .name = .of("n1"), .chip = .of("Apple M3 Ultra"), .gpu_cores = 80, .memory = 512 * node.gib, .gpu_limit = 453 * node.gib, .bandwidth = 819_000_000_000, .disk_free = 3 << 40 };
    inv.addPort(.{ .device = .of("rdma_en3"), .up = true, .gbps = 80, .gid = @splat(1), .peer_gid = @splat(2) });
    var buf: [1024]u8 = undefined;
    var o: Out = .{ .buf = &buf };
    try putInventory(&o, &inv);
    const msg = o.finish(.{ .kind = .inventory, .from = 99 });
    var i: In = .{ .buf = (try open(msg)).body };
    const back = try getInventory(&i);
    try std.testing.expectEqual(inv, back);
    var tiny: [50]u8 = undefined;
    var t: Out = .{ .buf = &tiny };
    try std.testing.expectError(error.NoRoom, putInventory(&t, &inv));
}
