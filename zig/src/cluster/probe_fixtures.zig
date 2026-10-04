//! Synthetic tool output in the exact formats macOS 27 prints, with made-up UUIDs and GIDs, for the probe and topology tests.
const std = @import("std");
const probe = @import("probe.zig");

pub const domain_a0 = "0A000000-0000-4000-8000-000000000000";
pub const domain_a1 = "0A000000-0000-4000-8000-000000000001";
pub const domain_b1 = "0B000000-0000-4000-8000-000000000001";

pub const sysctl_a =
    \\hw.memsize: 549755813888
    \\hw.pagesize: 16384
    \\iogpu.wired_limit_mb: 0
    \\vm.page_free_count: 24000000
    \\vm.page_speculative_count: 200000
    \\machdep.cpu.brand_string: Apple M3 Ultra
    \\kern.osproductversion: 27.0
;

pub const displays =
    \\Graphics/Displays:
    \\
    \\    Apple M3 Ultra:
    \\
    \\      Chipset Model: Apple M3 Ultra
    \\      Type: GPU
    \\      Total Number of Cores: 80
    \\      Metal Support: Metal 4
;

pub const df =
    \\Filesystem     1024-blocks       Used  Available Capacity iused       ifree %iused  Mounted on
    \\/dev/disk3s1s1  7811085600   13333796    1000000     1%  484014  4288821219    0%   /
    \\/dev/disk3s5    7811085600 4385517656    1000000    57% 1501896 33992116560    0%   /System/Volumes/Data
;

pub const thunderbolt_a =
    \\Thunderbolt/USB4:
    \\
    \\    Thunderbolt/USB4 Bus 1:
    \\
    \\      Vendor Name: Apple Inc.
    \\      Device Name: Mac Studio
    \\      Domain UUID: 0A000000-0000-4000-8000-000000000001
    \\      Port:
    \\          Status: Device connected
    \\          Link Status: 0x2
    \\          Speed: 80 Gb/s
    \\          Receptacle: 2
    \\
    \\        Mac Studio:
    \\
    \\          Vendor Name: Apple Inc.
    \\          Device Name: Mac15,14
    \\          Domain UUID: 0B000000-0000-4000-8000-000000000001
    \\
    \\    Thunderbolt/USB4 Bus 0:
    \\
    \\      Vendor Name: Apple Inc.
    \\      Domain UUID: 0A000000-0000-4000-8000-000000000000
    \\      Port:
    \\          Status: No device connected
    \\          Speed: Up to 120 Gb/s
    \\          Receptacle: 1
;

pub const devinfo_a =
    "hca_id:\trdma_en2\n" ++
    "\ttransport:\t\t\tThunderbolt (100)\n" ++
    "\tmax_mr_size:\t\t\t0xfa0000\n" ++
    "\tmax_qp:\t\t\t\t3\n" ++
    "\tmax_mr:\t\t\t\t100\n" ++
    "\tatomic_cap:\t\t\tATOMIC_NONE (0)\n" ++
    "\t\tport:\t1\n" ++
    "\t\t\tstate:\t\t\tPORT_DOWN (1)\n" ++
    "\t\t\tactive_mtu:\t\t4096 (5)\n" ++
    "\t\t\tGID[  0]:\t\t0a00:0000:0000:4000:8000:0000:0000:0000\n" ++
    "\t\t\tGID[  1]:\t\tfe80::3498:1bff:fe4a:9540\n" ++
    "\n" ++
    "hca_id:\trdma_en3\n" ++
    "\ttransport:\t\t\tThunderbolt (100)\n" ++
    "\tmax_mr_size:\t\t\t0xfa0000\n" ++
    "\tmax_qp:\t\t\t\t3\n" ++
    "\tmax_mr:\t\t\t\t100\n" ++
    "\tmax_msg_sz:\t\t0x1000000\n" ++
    "\tatomic_cap:\t\t\tATOMIC_NONE (0)\n" ++
    "\t\tport:\t1\n" ++
    "\t\t\tstate:\t\t\tPORT_ACTIVE (4)\n" ++
    "\t\t\tactive_mtu:\t\t4096 (5)\n" ++
    "\t\t\tGID[  0]:\t\t0a00:0000:0000:4000:8000:0000:0000:0001\n" ++
    "\t\t\tGID[  1]:\t\tfe80::3498:1bff:fe4a:9544\n";

pub fn texts(name: []const u8, sys: []const u8, tb: []const u8, dev: []const u8) probe.Texts {
    return .{ .name = name, .sysctl = sys, .displays = displays, .thunderbolt = tb, .devinfo = dev, .df = df };
}

/// One cable between node `a`'s bus `a_bus` and node `b`'s bus `b_bus`.
pub const Cable = struct { a: u8, a_bus: u8, b: u8, b_bus: u8 };

fn domain(buf: *[36]u8, n: u8, bus: u8) []const u8 {
    return std.fmt.bufPrint(buf, "{X:0>2}000000-0000-4000-8000-0000000000{X:0>2}", .{ n + 1, bus }) catch unreachable;
}

fn peerOf(cables: []const Cable, n: u8, bus: u8) ?[2]u8 {
    for (cables) |c| {
        if (c.a == n and c.a_bus == bus) return .{ c.b, c.b_bus };
        if (c.b == n and c.b_bus == bus) return .{ c.a, c.a_bus };
    }
    return null;
}

/// Thunderbolt and ibv_devinfo text for node `n` with `buses` buses, wired by `cables`.
pub fn render(a: std.mem.Allocator, n: u8, buses: u8, cables: []const Cable) ![2][]u8 {
    var tb: std.Io.Writer.Allocating = .init(a);
    var dev: std.Io.Writer.Allocating = .init(a);
    try tb.writer.writeAll("Thunderbolt/USB4:\n\n");
    var d1: [36]u8 = undefined;
    var d2: [36]u8 = undefined;
    for (0..buses) |i| {
        const bus: u8 = @intCast(i);
        const peer = peerOf(cables, n, bus);
        const status = if (peer != null) "Device connected" else "No device connected";
        const speed = if (peer != null) "80 Gb/s" else "Up to 120 Gb/s";
        try tb.writer.print("    Thunderbolt/USB4 Bus {d}:\n\n      Domain UUID: {s}\n      Port:\n          Status: {s}\n          Speed: {s}\n          Receptacle: {d}\n\n", .{ bus, domain(&d1, n, bus), status, speed, bus + 1 });
        if (peer) |p| try tb.writer.print("        Mac Studio:\n\n          Domain UUID: {s}\n\n", .{domain(&d2, p[0], p[1])});
        const state = if (peer != null) "PORT_ACTIVE (4)" else "PORT_DOWN (1)";
        const g = probe.uuid(domain(&d1, n, bus)).?;
        try dev.writer.print("hca_id:\trdma_en{d}\n\ttransport:\t\t\tThunderbolt (100)\n\tmax_mr_size:\t\t\t0xfa0000\n\tatomic_cap:\t\t\tATOMIC_NONE (0)\n\t\t\tstate:\t\t\t{s}\n\t\t\tGID[  0]:\t\t", .{ bus + 2, state });
        for (0..8) |j| try dev.writer.print("{s}{x:0>4}", .{ if (j == 0) "" else ":", std.mem.readInt(u16, g[2 * j ..][0..2], .big) });
        try dev.writer.writeAll("\n\n");
    }
    return .{ try tb.toOwnedSlice(), try dev.toOwnedSlice() };
}

/// Saved-probe folders for fake hosts: DIR/NAME/{sysctl,displays,thunderbolt,ibv_devinfo,df}.txt, cabled by `cables`.
pub fn writeHosts(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, names: []const []const u8, cables: []const Cable) !void {
    for (names, 0..) |name, i| {
        try dir.createDirPath(io, name);
        var sub = try dir.openDir(io, name, .{});
        defer sub.close(io);
        const t = try render(a, @intCast(i), 4, cables);
        const files = [_]struct { []const u8, []const u8 }{ .{ "sysctl.txt", sysctl_a }, .{ "displays.txt", displays }, .{ "thunderbolt.txt", t[0] }, .{ "ibv_devinfo.txt", t[1] }, .{ "df.txt", df } };
        for (files) |f| try sub.writeFile(io, .{ .sub_path = f[0], .data = f[1] });
    }
}

/// Six cables joining four hosts on buses 1-3: every pair once.
pub const mesh4 = [_]Cable{ .{ .a = 0, .a_bus = 1, .b = 3, .b_bus = 1 }, .{ .a = 0, .a_bus = 2, .b = 2, .b_bus = 2 }, .{ .a = 0, .a_bus = 3, .b = 1, .b_bus = 3 }, .{ .a = 1, .a_bus = 1, .b = 2, .b_bus = 1 }, .{ .a = 1, .a_bus = 2, .b = 3, .b_bus = 2 }, .{ .a = 2, .a_bus = 3, .b = 3, .b_bus = 3 } };
