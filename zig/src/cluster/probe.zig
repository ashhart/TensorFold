//! An inventory from: macOS sysctl, system_profiler; Linux meminfo, nvidia-smi, machine-id; both ibv_devinfo -v, df -k.
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
    /// Table index of `gid`. 0 until a RoCE v2 row replaces GID[0].
    gid_index: u8 = 0,
    max_mr_size: u64 = 0,
    max_mr: u32 = 0,
    max_qp: u32 = 0,
    max_msg: u64 = 0,
    atomics: bool = false,
    /// ConnectX-7: the text says so, or vendor_part_id is 4129. Anything else non-Thunderbolt stays ConnectX-5.
    cx7: bool = false,
    /// A RoCE v2 row is already kept, so a later v2 row must not replace it.
    v2: bool = false,
};

/// `ibv_devinfo -v`: each device's port state, its first RoCE v2 GID (or GID[0] when no row says so) and limits.
pub fn devinfo(text: []const u8, out: []Device) usize {
    var n: usize = 0;
    var width: u32 = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = split(line);
        if (kv) |pair| if (eql(pair[0], "hca_id")) {
            if (n == out.len) break;
            out[n] = .{ .name = .of(pair[1]) };
            n += 1;
            continue;
        };
        if (n == 0) continue;
        const d = &out[n - 1];
        if (std.mem.indexOf(u8, line, "ConnectX-7") != null) d.cx7 = true;
        const pair = kv orelse continue;
        if (eql(pair[0], "vendor_part_id") and int(pair[1]) == 4129) d.cx7 = true;
        if (eql(pair[0], "transport")) d.thunderbolt = std.mem.startsWith(u8, pair[1], "Thunderbolt");
        if (eql(pair[0], "state")) d.active = std.mem.startsWith(u8, pair[1], "PORT_ACTIVE");
        if (eql(pair[0], "active_mtu")) d.mtu = @intCast(int(pair[1]));
        if (eql(pair[0], "max_mr_size")) d.max_mr_size = int(pair[1]);
        if (eql(pair[0], "max_mr")) d.max_mr = @intCast(int(pair[1]));
        if (eql(pair[0], "max_qp")) d.max_qp = @intCast(int(pair[1]));
        if (eql(pair[0], "max_msg_sz")) d.max_msg = int(pair[1]);
        if (eql(pair[0], "atomic_cap")) d.atomics = !std.mem.startsWith(u8, pair[1], "ATOMIC_NONE");
        if (eql(pair[0], "active_width")) width = @intCast(int(pair[1]));
        if (eql(pair[0], "active_speed")) d.gbps = width * @as(u32, @intCast(int(pair[1])));
        if (std.mem.startsWith(u8, pair[0], "GID[")) takeGid(d, pair[0], pair[1]);
    }
    return n;
}

/// Keep GID[0] until the first "RoCE v2" row. The comma suffix is not part of the address.
fn takeGid(d: *Device, key: []const u8, value: []const u8) void {
    const index: u8 = @intCast(@min(int(key[4..]), 255));
    const comma = std.mem.indexOfScalar(u8, value, ',');
    const addr = std.mem.trim(u8, if (comma) |c| value[0..c] else value, " \t");
    const note = if (comma) |c| value[c + 1 ..] else "";
    const parsed = gid(addr) orelse return;
    if (std.mem.indexOf(u8, note, "RoCE v2") != null and !d.v2) {
        d.gid = parsed;
        d.gid_index = index;
        d.v2 = true;
    } else if (!d.v2 and index == 0) {
        d.gid = parsed;
        d.gid_index = 0;
    }
}

/// `df -k`: Available kilobytes (4th of macOS's nine columns or Linux's six) in the row whose last field is `mount`.
pub fn dfAvailable(text: []const u8, mount: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var fields: [12][]const u8 = undefined;
        var count: usize = 0;
        var it = std.mem.tokenizeAny(u8, line, " \t");
        while (it.next()) |f| : (count += 1) {
            if (count < fields.len) fields[count] = f;
        }
        if (count < 6 or count > fields.len or !eql(fields[count - 1], mount)) continue;
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
        var p: node.Port = .{ .device = d.name, .kind = portKind(d), .up = d.active, .gbps = d.gbps, .gid = d.gid, .gid_index = d.gid_index };
        for (buses[0..nb]) |b| {
            if (!std.mem.eql(u8, &b.domain, &d.gid)) continue;
            p.gbps = b.gbps;
            p.peer_gid = b.peer;
        }
        inv.addPort(p);
    }
    return inv;
}

fn portKind(d: Device) node.LinkKind {
    if (d.thunderbolt) return .tb5;
    return if (d.cx7) .cx7 else .cx5;
}

pub const MemInfo = struct { total: u64 = 0, available: u64 = 0 };

/// `/proc/meminfo`: MemTotal and MemAvailable, in bytes.
pub fn meminfo(text: []const u8) MemInfo {
    var m: MemInfo = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = split(line) orelse continue;
        if (eql(kv[0], "MemTotal")) m.total = int(kv[1]) * 1024;
        if (eql(kv[0], "MemAvailable")) m.available = int(kv[1]) * 1024;
    }
    return m;
}

pub const Nvidia = struct { name: Name = .{}, memory: u64 = 0, free: u64 = 0, cores: u32 = 0 };

/// `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits` (MiB); `[N/A]` or none is 0.
pub fn nvidiaSmi(text: []const u8) Nvidia {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        if (t.len == 0 or std.mem.indexOfScalar(u8, t, ',') == null) continue;
        var fields: [4][]const u8 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, t, ',');
        while (it.next()) |f| {
            if (n < fields.len) fields[n] = std.mem.trim(u8, f, " \t");
            n += 1;
        }
        if (n < 3) continue;
        return .{
            .name = .of(fields[0]),
            .memory = int(fields[1]) << 20,
            .free = int(fields[2]) << 20,
            .cores = if (n > 3) @intCast(@min(int(fields[3]), std.math.maxInt(u32))) else 0,
        };
    }
    return .{};
}

pub const LinuxTexts = struct {
    name: []const u8,
    meminfo: []const u8,
    nvidia: []const u8,
    devinfo: []const u8,
    df: []const u8,
    machine_id: []const u8 = "",
    mount: []const u8 = "/",
};

/// A Linux host's inventory; gpu_limit is 0 without nvidia-smi's memory, for gpu_limit_gb (a GB10 left 0 is refused).
pub fn inventoryLinux(t: LinuxTexts) node.Inventory {
    const mem = meminfo(t.meminfo);
    const gpu = nvidiaSmi(t.nvidia);
    const chip = if (gpu.name.len > 0) gpu.name else Name.of("CUDA");
    const gb10 = std.mem.indexOf(u8, chip.str(), "GB10") != null;
    var devs: [node.max_ports]Device = undefined;
    const nd = devinfo(t.devinfo, &devs);
    const id_text = std.mem.trim(u8, t.machine_id, " \t\r\n");
    var inv: node.Inventory = .{
        .id = node.idFrom(if (id_text.len > 0) id_text else t.name),
        .name = .of(t.name),
        .backend = .cuda,
        .chip = chip,
        .gpu_cores = gpu.cores,
        .unified = gb10,
        .memory = mem.total,
        .gpu_limit = gpu.memory,
        .free = mem.available,
        .bandwidth = node.bandwidthOf(chip.str()),
        .disk_free = (dfAvailable(t.df, t.mount) orelse 0) * 1024,
    };
    for (devs[0..nd]) |d| {
        inv.addPort(.{ .device = d.name, .kind = portKind(d), .up = d.active, .gbps = d.gbps, .gid = d.gid, .gid_index = d.gid_index });
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

/// An IPv6-style GID ("0011:2233:...", "fe80::aaaa:bbff:fecc:dd40") as 16 bytes.
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
    while (it.next()) |part| {
        if (std.mem.indexOfScalar(u8, part, '.') != null) {
            if (n + 2 > out.len) return null;
            const v4 = ipv4(part) orelse return null;
            out[n] = @intCast(v4 >> 16);
            out[n + 1] = @intCast(v4 & 0xffff);
            n += 2;
        } else {
            if (n == out.len) return null;
            out[n] = std.fmt.parseInt(u16, part, 16) catch return null;
            n += 1;
        }
    }
    return n;
}

/// `192.0.2.10` as the two groups an IPv4-mapped GID stores after `ffff`.
fn ipv4(text: []const u8) ?u32 {
    var o: u32 = 0;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |p| : (n += 1) {
        if (n == 4) return null;
        const b = std.fmt.parseInt(u8, p, 10) catch return null;
        o = (o << 8) | b;
    }
    return if (n == 4) o else null;
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
    const linux_df =
        \\Filesystem     1K-blocks      Used Available Use% Mounted on
        \\/dev/nvme0n1p2 900000000 100000000 750000000  12% /
    ;
    try std.testing.expectEqual(@as(?u64, 750_000_000), dfAvailable(linux_df, "/"));
}

test "a GB10 inventory comes from meminfo, nvidia-smi and ConnectX-7" {
    const mem =
        \\MemTotal:       131072000 kB
        \\MemAvailable:    80000000 kB
    ;
    // a GB10, 2026-11-02: the name query exits 0 and both memory columns are [N/A].
    const nv = "NVIDIA GB10, [N/A], [N/A]\n";
    const dev = @embedFile("testdata/gb10-ibv.txt");
    const df =
        \\Filesystem     1K-blocks      Used Available Use% Mounted on
        \\/dev/nvme0n1p2 900000000 100000000 750000000  12% /
    ;
    var devs: [4]Device = undefined;
    try std.testing.expectEqual(@as(usize, 1), devinfo(dev, &devs));
    try std.testing.expect(devs[0].cx7 and devs[0].active);
    try std.testing.expectEqual(@as(u32, 4096), devs[0].mtu);
    try std.testing.expectEqual(@as(u32, 100), devs[0].gbps);
    try std.testing.expectEqual(@as(u8, 1), devs[0].gid_index);
    const v2 = gid("fe80::1").?;
    try std.testing.expectEqualSlices(u8, &v2, &devs[0].gid);
    const inv = inventoryLinux(.{ .name = "gb10", .meminfo = mem, .nvidia = nv, .devinfo = dev, .df = df, .machine_id = "abc123\n" });
    try std.testing.expectEqual(node.Backend.cuda, inv.backend);
    try std.testing.expect(inv.unified);
    try std.testing.expectEqualStrings("NVIDIA GB10", inv.chip.str());
    try std.testing.expectEqual(@as(u64, 131_072_000 * 1024), inv.memory);
    try std.testing.expectEqual(@as(u64, 80_000_000 * 1024), inv.free);
    try std.testing.expectEqual(@as(u64, 0), inv.gpu_limit);
    try std.testing.expectEqual(@as(u32, 0), inv.gpu_cores);
    try std.testing.expectEqual(@as(u64, 273_000_000_000), inv.bandwidth);
    try std.testing.expectEqual(@as(u64, 750_000_000 * 1024), inv.disk_free);
    try std.testing.expectEqual(node.idFrom("abc123"), inv.id);
    try std.testing.expectEqual(@as(usize, 1), inv.ports().len);
    try std.testing.expect(inv.ports()[0].up and inv.ports()[0].kind == .cx7);
    try std.testing.expectEqual(@as(u8, 1), inv.ports()[0].gid_index);
    try std.testing.expectEqualSlices(u8, &v2, &inv.ports()[0].gid);
    const bare = inventoryLinux(.{ .name = "box", .meminfo = mem, .nvidia = "", .devinfo = "", .df = df });
    try std.testing.expect(!bare.unified and bare.gpu_limit == 0);
    try std.testing.expectEqualStrings("CUDA", bare.chip.str());
    try std.testing.expectEqual(node.idFrom("box"), bare.id);
}

test "GIDs and UUIDs parse in both notations" {
    const a = gid("0011:2233:4455:6677:8899:aabb:ccdd:eeff").?;
    try std.testing.expectEqualSlices(u8, &a, &uuid("00112233-4455-6677-8899-AABBCCDDEEFF").?);
    const b = gid("fe80::aaaa:bbff:fecc:dd40").?;
    try std.testing.expectEqual(@as(u8, 0xfe), b[0]);
    try std.testing.expectEqual(@as(u8, 0x40), b[15]);
    try std.testing.expectEqualSlices(u8, &gid("fe80::1").?, &gid("fe80:0000:0000:0000:0000:0000:0000:0001").?);
    const mapped = gid("::ffff:192.0.2.10").?;
    try std.testing.expectEqualSlices(u8, &mapped, &gid("0000:0000:0000:0000:0000:ffff:c000:020a").?);
    try std.testing.expectEqualSlices(u8, &gid("::ffff:192.0.2.21").?, &gid("0000:0000:0000:0000:0000:ffff:c000:0215").?);
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
