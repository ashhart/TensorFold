//! An inventory from macOS's own tools: sysctl, system_profiler (displays, Thunderbolt), ibv_devinfo -v and df -k.
const std = @import("std");
const node = @import("node.zig");

const Name = node.Name;
const Gid = node.Gid;

pub const Sysctl = struct {
    memsize: u64 = 0,
    pagesize: u64 = 16384,
    wired_limit_mb: u64 = 0,
    free_pages: u64 = 0,
    speculative_pages: u64 = 0,
    chip: Name = .{},
    os: Name = .{},
};

/// The "key: value" lines `sysctl NAME...` prints.
pub fn sysctl(text: []const u8) Sysctl {
    var s: Sysctl = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = split(line) orelse continue;
        if (eql(kv[0], "hw.memsize")) s.memsize = int(kv[1]);
        if (eql(kv[0], "hw.pagesize")) s.pagesize = int(kv[1]);
        if (eql(kv[0], "iogpu.wired_limit_mb")) s.wired_limit_mb = int(kv[1]);
        if (eql(kv[0], "vm.page_free_count")) s.free_pages = int(kv[1]);
        if (eql(kv[0], "vm.page_speculative_count")) s.speculative_pages = int(kv[1]);
        if (eql(kv[0], "machdep.cpu.brand_string")) s.chip = .of(kv[1]);
        if (eql(kv[0], "kern.osproductversion")) s.os = .of(kv[1]);
    }
    return s;
}

pub const Gpu = struct { chip: Name = .{}, cores: u32 = 0 };

/// `system_profiler SPDisplaysDataType`: the GPU's chip and core count.
pub fn displays(text: []const u8) Gpu {
    var g: Gpu = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = split(line) orelse continue;
        if (eql(kv[0], "Chipset Model") and g.chip.len == 0) g.chip = .of(kv[1]);
        if (eql(kv[0], "Total Number of Cores") and g.cores == 0) g.cores = @intCast(int(kv[1]));
    }
    return g;
}

/// One Thunderbolt bus: its own domain UUID, and the far end's when a cable connects a host.
pub const Bus = struct {
    index: u8 = 0,
    receptacle: u8 = 0,
    connected: bool = false,
    gbps: u32 = 0,
    domain: Gid = node.no_gid,
    peer: Gid = node.no_gid,
};

/// `system_profiler SPThunderboltDataType`: every bus, with the connected host's domain UUID.
pub fn thunderbolt(text: []const u8, out: []Bus) usize {
    const head = "Thunderbolt/USB4 Bus ";
    var n: usize = 0;
    var at = std.mem.indexOf(u8, text, head) orelse return 0;
    while (n < out.len) {
        const start = at + head.len;
        const next = std.mem.indexOfPos(u8, text, start, head);
        const block = text[start .. next orelse text.len];
        var b: Bus = .{ .index = @intCast(int(block[0 .. std.mem.indexOfScalar(u8, block, ':') orelse 0])) };
        var domains: u8 = 0;
        var lines = std.mem.splitScalar(u8, block, '\n');
        while (lines.next()) |line| {
            const kv = split(line) orelse continue;
            if (eql(kv[0], "Domain UUID")) {
                const g = uuid(kv[1]) orelse continue;
                if (domains == 0) b.domain = g else if (domains == 1) b.peer = g;
                domains += 1;
            }
            if (eql(kv[0], "Receptacle")) b.receptacle = @intCast(int(kv[1]));
            if (eql(kv[0], "Status")) b.connected = eql(kv[1], "Device connected");
            if (eql(kv[0], "Speed") and !std.mem.startsWith(u8, kv[1], "Up to")) b.gbps = @intCast(int(kv[1]));
        }
        if (!b.connected) b.peer = node.no_gid;
        out[n] = b;
        n += 1;
        at = next orelse break;
    }
    return n;
}

/// One verbs device with the limits a transport must respect.
pub const Device = struct {
    name: Name = .{},
    thunderbolt: bool = false,
    active: bool = false,
    mtu: u32 = 0,
    gbps: u32 = 0,
    gid: Gid = node.no_gid,
    max_mr_size: u64 = 0,
    max_mr: u32 = 0,
    max_qp: u32 = 0,
    max_msg: u64 = 0,
    atomics: bool = false,
};

/// `ibv_devinfo -v`: each device's port state, GID[0] and limits.
pub fn devinfo(text: []const u8, out: []Device) usize {
    var n: usize = 0;
    var width: u32 = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = split(line) orelse continue;
        if (eql(kv[0], "hca_id")) {
            if (n == out.len) break;
            out[n] = .{ .name = .of(kv[1]) };
            n += 1;
            continue;
        }
        if (n == 0) continue;
        const d = &out[n - 1];
        if (eql(kv[0], "transport")) d.thunderbolt = std.mem.startsWith(u8, kv[1], "Thunderbolt");
        if (eql(kv[0], "state")) d.active = std.mem.startsWith(u8, kv[1], "PORT_ACTIVE");
        if (eql(kv[0], "active_mtu")) d.mtu = @intCast(int(kv[1]));
        if (eql(kv[0], "max_mr_size")) d.max_mr_size = int(kv[1]);
        if (eql(kv[0], "max_mr")) d.max_mr = @intCast(int(kv[1]));
        if (eql(kv[0], "max_qp")) d.max_qp = @intCast(int(kv[1]));
        if (eql(kv[0], "max_msg_sz")) d.max_msg = int(kv[1]);
        if (eql(kv[0], "atomic_cap")) d.atomics = !std.mem.startsWith(u8, kv[1], "ATOMIC_NONE");
        if (eql(kv[0], "active_width")) width = @intCast(int(kv[1]));
        if (eql(kv[0], "active_speed")) d.gbps = width * @as(u32, @intCast(int(kv[1])));
        if (std.mem.startsWith(u8, kv[0], "GID[") and int(kv[0][4..]) == 0) d.gid = gid(kv[1]) orelse node.no_gid;
    }
    return n;
}

/// `df -k`: the Available kilobytes of the row mounted at `mount`, in bytes.
pub fn dfAvailable(text: []const u8, mount: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var fields: [9][]const u8 = undefined;
        var count: usize = 0;
        var it = std.mem.tokenizeAny(u8, line, " \t");
        while (it.next()) |f| : (count += 1) {
            if (count < fields.len) fields[count] = f;
        }
        if (count < 9 or !eql(fields[8], mount)) continue;
        return std.fmt.parseInt(u64, fields[3], 10) catch null;
    }
    return null;
}

pub const Texts = struct {
    name: []const u8,
    sysctl: []const u8,
    displays: []const u8,
    thunderbolt: []const u8,
    devinfo: []const u8,
    df: []const u8,
    mount: []const u8 = "/System/Volumes/Data",
};

/// A Mac's inventory from its tools' text; ports join their Thunderbolt bus by GID[0] == the bus's domain UUID.
pub fn inventory(t: Texts) node.Inventory {
    const s = sysctl(t.sysctl);
    const g = displays(t.displays);
    var buses: [16]Bus = undefined;
    const nb = thunderbolt(t.thunderbolt, &buses);
    var devs: [node.max_ports]Device = undefined;
    const nd = devinfo(t.devinfo, &devs);
    const chip = if (g.chip.len > 0) g.chip else s.chip;
    var inv: node.Inventory = .{
        .name = .of(t.name),
        .chip = chip,
        .gpu_cores = g.cores,
        .memory = s.memsize,
        .wired_limit_mb = s.wired_limit_mb,
        .gpu_limit = node.gpuLimitFor(s.memsize, s.wired_limit_mb),
        .free = (s.free_pages + s.speculative_pages) * s.pagesize,
        .bandwidth = node.bandwidthOf(chip.str()),
        .disk_free = (dfAvailable(t.df, t.mount) orelse 0) * 1024,
    };
    var ids = std.hash.Wyhash.init(0);
    for (buses[0..nb]) |b| ids.update(&b.domain);
    inv.id = if (nb > 0) node.idFrom(std.mem.asBytes(&ids.final())) else node.idFrom(t.name);
    for (devs[0..nd]) |d| {
        var p: node.Port = .{ .device = d.name, .kind = if (d.thunderbolt) .tb5 else .cx5, .up = d.active, .gbps = d.gbps, .gid = d.gid };
        for (buses[0..nb]) |b| {
            if (!std.mem.eql(u8, &b.domain, &d.gid)) continue;
            p.gbps = b.gbps;
            p.peer_gid = b.peer;
        }
        inv.addPort(p);
    }
    return inv;
}

/// "DB3DB985-F2E1-48F5-88B8-2E869DC0E9D7" as 16 bytes.
pub fn uuid(text: []const u8) ?Gid {
    var out: Gid = undefined;
    var k: usize = 0;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == '-') continue;
        if (k == 16 or i + 1 >= text.len) return null;
        out[k] = std.fmt.parseInt(u8, text[i .. i + 2], 16) catch return null;
        k += 1;
        i += 1;
    }
    return if (k == 16) out else null;
}

/// An IPv6-style GID ("aca4:89d1:...", "fe80::3498:1bff:fe4a:9540") as 16 bytes.
pub fn gid(text: []const u8) ?Gid {
    var groups: [8]u16 = @splat(0);
    const gap = std.mem.indexOf(u8, text, "::");
    const head = if (gap) |g| text[0..g] else text;
    const tail = if (gap) |g| text[g + 2 ..] else "";
    const nh = fill(head, groups[0..]) orelse return null;
    var back: [8]u16 = undefined;
    const nt = fill(tail, back[0..]) orelse return null;
    if (gap == null and nh != 8) return null;
    if (nh + nt > 8) return null;
    @memcpy(groups[8 - nt ..], back[0..nt]);
    var out: Gid = undefined;
    for (groups, 0..) |v, j| std.mem.writeInt(u16, out[2 * j ..][0..2], v, .big);
    return out;
}

fn fill(text: []const u8, out: []u16) ?usize {
    if (text.len == 0) return 0;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, ':');
    while (it.next()) |part| : (n += 1) {
        if (n == out.len) return null;
        out[n] = std.fmt.parseInt(u16, part, 16) catch return null;
    }
    return n;
}

fn split(line: []const u8) ?[2][]const u8 {
    const colon = std.mem.indexOfScalar(u8, line, ':') orelse return null;
    return .{ std.mem.trim(u8, line[0..colon], " \t\r"), std.mem.trim(u8, line[colon + 1 ..], " \t\r") };
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// The leading number of `text` (decimal, or hex after 0x); 0 when there is none.
fn int(text: []const u8) u64 {
    const t = std.mem.trim(u8, text, " \t[]");
    if (std.mem.startsWith(u8, t, "0x")) {
        var end: usize = 2;
        while (end < t.len and std.ascii.isHex(t[end])) end += 1;
        return std.fmt.parseInt(u64, t[2..end], 16) catch 0;
    }
    var end: usize = 0;
    while (end < t.len and std.ascii.isDigit(t[end])) end += 1;
    return std.fmt.parseInt(u64, t[0..end], 10) catch 0;
}

pub const testing = @import("probe_fixtures.zig");

test "sysctl, displays and df" {
    const s = sysctl(testing.sysctl_a);
    try std.testing.expectEqual(@as(u64, 512 * node.gib), s.memsize);
    try std.testing.expectEqual(@as(u64, 0), s.wired_limit_mb);
    try std.testing.expectEqualStrings("Apple M3 Ultra", s.chip.str());
    const g = displays(testing.displays);
    try std.testing.expectEqual(@as(u32, 80), g.cores);
    try std.testing.expectEqual(@as(?u64, 1_000_000), dfAvailable(testing.df, "/System/Volumes/Data"));
    try std.testing.expectEqual(@as(?u64, null), dfAvailable(testing.df, "/Volumes/none"));
}

test "GIDs and UUIDs parse in both notations" {
    const a = gid("aca4:89d1:0c3b:4215:9512:325f:c1e8:e37a").?;
    try std.testing.expectEqualSlices(u8, &a, &uuid("ACA489D1-0C3B-4215-9512-325FC1E8E37A").?);
    const b = gid("fe80::3498:1bff:fe4a:9540").?;
    try std.testing.expectEqual(@as(u8, 0xfe), b[0]);
    try std.testing.expectEqual(@as(u8, 0x40), b[15]);
    try std.testing.expectEqual(@as(?Gid, null), gid("1:2:3"));
    try std.testing.expectEqual(@as(?Gid, null), uuid("not-a-uuid"));
}

test "Thunderbolt buses and verbs devices join into ports with their peers" {
    var buses: [8]Bus = undefined;
    const nb = thunderbolt(testing.thunderbolt_a, &buses);
    try std.testing.expectEqual(@as(usize, 2), nb);
    try std.testing.expect(buses[0].index == 1 and buses[0].connected and buses[0].gbps == 80 and buses[0].receptacle == 2);
    try std.testing.expect(buses[1].index == 0 and !buses[1].connected and buses[1].gbps == 0);
    var devs: [4]Device = undefined;
    const nd = devinfo(testing.devinfo_a, &devs);
    try std.testing.expectEqual(@as(usize, 2), nd);
    try std.testing.expect(devs[1].active and devs[1].thunderbolt and !devs[1].atomics);
    try std.testing.expectEqual(@as(u64, 0xfa0000), devs[1].max_mr_size);
    try std.testing.expectEqual(@as(u32, 100), devs[1].max_mr);
    const inv = inventory(testing.texts("a", testing.sysctl_a, testing.thunderbolt_a, testing.devinfo_a));
    try std.testing.expectEqual(@as(usize, 2), inv.ports().len);
    const up = inv.ports()[1];
    try std.testing.expect(up.up and up.gbps == 80);
    try std.testing.expectEqualSlices(u8, &uuid(testing.domain_b1).?, &up.peer_gid);
    try std.testing.expect(inv.gpu_limit > 450 * node.gib and inv.bandwidth == 819_000_000_000);
}
