//! A node's inventory (backend, the memory its GPU may hold, its RDMA ports, its disk) and the cluster's totals.
const std = @import("std");

pub const NodeId = u64;
pub const gib: u64 = 1 << 30;

pub const Backend = enum(u8) { metal = 1, cuda = 2 };

/// Thunderbolt 5 between Macs, MCDMA's ConnectX-5 between a Mac and a CUDA host, ConnectX-7 between CUDA hosts.
pub const LinkKind = enum(u8) { tb5 = 1, cx5 = 2, cx7 = 3 };

/// A GID as the verbs library reports it; on Thunderbolt GID[0] is the bus's domain UUID.
pub const Gid = [16]u8;
pub const no_gid: Gid = @splat(0);

/// A short fixed-size string, so an inventory holds no pointers and travels as bytes.
pub const Name = struct {
    bytes: [32]u8 = @splat(0),
    len: u8 = 0,

    pub fn of(s: []const u8) Name {
        var n: Name = .{};
        const k = @min(s.len, n.bytes.len);
        @memcpy(n.bytes[0..k], s[0..k]);
        n.len = @intCast(k);
        return n;
    }

    pub fn str(n: *const Name) []const u8 {
        return n.bytes[0..n.len];
    }
};

pub const Port = struct {
    device: Name = .{},
    kind: LinkKind = .tb5,
    up: bool = false,
    gbps: u32 = 0,
    gid: Gid = no_gid,
    /// The far end's GID[0] (zero while the port is down or the cable's peer is unknown).
    peer_gid: Gid = no_gid,
};

pub const max_ports = 8;

pub const Inventory = struct {
    id: NodeId = 0,
    name: Name = .{},
    backend: Backend = .metal,
    chip: Name = .{},
    gpu_cores: u32 = 0,
    /// Unified memory: the GPU's limit comes out of the same physical bytes as the host's.
    unified: bool = true,
    memory: u64 = 0,
    /// Bytes the GPU may keep resident: Metal's recommended working set (the wired limit) or the card's VRAM.
    gpu_limit: u64 = 0,
    /// iogpu.wired_limit_mb as set on a Mac; 0 means macOS chooses (gpu_limit then holds its default).
    wired_limit_mb: u64 = 0,
    free: u64 = 0,
    /// Nominal GPU memory bandwidth in bytes a second, from the chip.
    bandwidth: u64 = 0,
    disk_free: u64 = 0,
    port_count: u8 = 0,
    port_list: [max_ports]Port = @splat(.{}),

    pub fn ports(inv: *const Inventory) []const Port {
        return inv.port_list[0..inv.port_count];
    }

    pub fn addPort(inv: *Inventory, p: Port) void {
        if (inv.port_count == max_ports) return;
        inv.port_list[inv.port_count] = p;
        inv.port_count += 1;
    }

    /// The port whose own GID is `gid`, if this node has it.
    pub fn portByGid(inv: *const Inventory, gid: Gid) ?u8 {
        for (inv.ports(), 0..) |p, i| if (std.mem.eql(u8, &p.gid, &gid)) return @intCast(i);
        return null;
    }
};

/// macOS's GPU limit when iogpu.wired_limit_mb is 0: fit to 107.5 GiB on 128 GiB and 222.7 GiB on 256 GiB, two thirds below that.
pub fn defaultGpuLimit(memory: u64) u64 {
    const fit = memory / 10 * 9 -| 768 * gib / 100;
    return @max(memory / 3 * 2, fit);
}

/// The GPU limit a wired-limit setting gives: the setting itself when non-zero, else macOS's default.
pub fn gpuLimitFor(memory: u64, wired_limit_mb: u64) u64 {
    return if (wired_limit_mb == 0) defaultGpuLimit(memory) else wired_limit_mb << 20;
}

/// Nominal memory bandwidth (bytes a second) for the chips we serve; 0 when unknown (the engine measures it).
pub fn bandwidthOf(chip: []const u8) u64 {
    const table = [_]struct { []const u8, u64 }{
        .{ "M3 Ultra", 819 },  .{ "M2 Ultra", 800 },  .{ "M1 Ultra", 800 }, .{ "M4 Max", 546 }, .{ "M3 Max", 400 },
        .{ "M2 Max", 400 },    .{ "M1 Max", 400 },    .{ "M4 Pro", 273 },   .{ "GB10", 273 },   .{ "RTX PRO 6000", 1792 },
        .{ "RTX 5090", 1792 }, .{ "RTX 4090", 1008 }, .{ "RTX 3090", 936 },
    };
    for (table) |row| if (std.mem.indexOf(u8, chip, row[0]) != null) return row[1] * 1_000_000_000;
    return 0;
}

/// A stable node id from a platform UUID (never zero).
pub fn idFrom(uuid: []const u8) NodeId {
    const h = std.hash.Wyhash.hash(0x54464e4f4445, uuid);
    return if (h == 0) 1 else h;
}

pub const Totals = struct {
    nodes: u32 = 0,
    memory: u64 = 0,
    gpu_limit: u64 = 0,
    free: u64 = 0,
    disk_free: u64 = 0,
    gpu_cores: u32 = 0,
    bandwidth: u64 = 0,
    ports_up: u32 = 0,
};

pub fn totals(nodes: []const Inventory) Totals {
    var t: Totals = .{};
    for (nodes) |n| {
        t.nodes += 1;
        t.memory += n.memory;
        t.gpu_limit += n.gpu_limit;
        t.free += n.free;
        t.disk_free += n.disk_free;
        t.gpu_cores += n.gpu_cores;
        t.bandwidth += n.bandwidth;
        for (n.ports()) |p| t.ports_up += @intFromBool(p.up);
    }
    return t;
}

test "macOS's default GPU limit matches both measured Macs and two thirds on small ones" {
    try std.testing.expectApproxEqRel(@as(f64, 115_448_725_504), @as(f64, @floatFromInt(defaultGpuLimit(128 * gib))), 1e-6);
    try std.testing.expectApproxEqRel(@as(f64, 239_143_780_352), @as(f64, @floatFromInt(defaultGpuLimit(256 * gib))), 1e-6);
    try std.testing.expectEqual(16 * gib / 3 * 2, defaultGpuLimit(16 * gib));
    try std.testing.expectEqual(@as(u64, 491_520) << 20, gpuLimitFor(512 * gib, 491_520));
    try std.testing.expect(defaultGpuLimit(512 * gib) > 450 * gib and defaultGpuLimit(512 * gib) < 455 * gib);
}

test "names, ports, bandwidth and totals" {
    var a: Inventory = .{ .name = .of("node-a"), .chip = .of("Apple M3 Ultra"), .memory = 512 * gib, .gpu_limit = 400 * gib };
    a.addPort(.{ .device = .of("rdma_en3"), .up = true, .gbps = 80, .gid = @splat(3) });
    a.addPort(.{ .device = .of("rdma_en2"), .gid = @splat(2) });
    var b = a;
    b.name = .of("node-b");
    try std.testing.expectEqualStrings("node-a", a.name.str());
    try std.testing.expectEqual(@as(?u8, 1), a.portByGid(@splat(2)));
    try std.testing.expectEqual(@as(u64, 819_000_000_000), bandwidthOf(a.chip.str()));
    try std.testing.expectEqual(@as(u64, 0), bandwidthOf("Apple M9"));
    const t = totals(&.{ a, b });
    try std.testing.expectEqual(@as(u64, 1024 * gib), t.memory);
    try std.testing.expectEqual(@as(u32, 2), t.ports_up);
    try std.testing.expect(idFrom("A") != idFrom("B") and idFrom("A") != 0);
}
