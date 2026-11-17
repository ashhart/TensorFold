//! Each OS's raw readings as a Machine: macOS sysctl text, Linux's /proc and /sys files, Windows' DXGI and registry.
const std = @import("std");
const machine = @import("machine.zig");
const Allocator = std.mem.Allocator;
const Machine = machine.Machine;
const Gpu = machine.Gpu;

/// "key: value" lines, as `sysctl NAME...` prints them; the readers write the same text from sysctlbyname.
pub fn macos(a: Allocator, arch: []const u8, sysctl: []const u8) !Machine {
    var m: Machine = .{ .os = .macos, .arch = arch };
    var lines = std.mem.splitScalar(u8, sysctl, '\n');
    while (lines.next()) |line| {
        const kv = pair(line, ':') orelse continue;
        if (eql(kv[0], "hw.memsize")) m.ram = int(kv[1]);
        if (eql(kv[0], "hw.ncpu")) m.cores = @intCast(@min(int(kv[1]), 1 << 16));
        if (eql(kv[0], "hw.perflevel0.logicalcpu")) m.performance_cores = @intCast(@min(int(kv[1]), 1 << 16));
        if (eql(kv[0], "hw.perflevel1.logicalcpu")) m.efficiency_cores = @intCast(@min(int(kv[1]), 1 << 16));
        if (eql(kv[0], "machdep.cpu.brand_string")) m.cpu = try a.dupe(u8, kv[1]);
        if (eql(kv[0], "kern.osproductversion")) m.os_version = try a.dupe(u8, kv[1]);
        if (eql(kv[0], "iogpu.wired_limit_mb")) m.gpu_wired_limit = int(kv[1]) << 20;
    }
    // Apple silicon's GPU is the chip's own, on the same memory; an Intel Mac's GPUs are not listed here.
    if (machine.appleChip(m.cpu) != null) m.gpus = try a.dupe(Gpu, &.{.{ .vendor = .apple, .name = m.cpu, .unified = true }});
    return m;
}

/// One PCI function from /sys/bus/pci/devices/SLOT: its class, vendor and device files, and amdgpu's VRAM size.
pub const Pci = struct { slot: []const u8, class: []const u8, vendor: []const u8, device: []const u8, vram: []const u8 = "" };
/// /proc/driver/nvidia/gpus/SLOT/information.
pub const NvidiaInfo = struct { slot: []const u8, text: []const u8 };

pub const LinuxTexts = struct {
    meminfo: []const u8 = "",
    cpuinfo: []const u8 = "",
    /// /etc/os-release.
    os_release: []const u8 = "",
    /// /proc/sys/kernel/osrelease.
    kernel: []const u8 = "",
    pci: []const Pci = &.{},
    nvidia: []const NvidiaInfo = &.{},
    /// /sys/class/kfd/kfd/topology/nodes/N/properties, in node order (AMD's gfx targets).
    kfd: []const []const u8 = &.{},
};

pub fn linux(a: Allocator, arch: []const u8, t: LinuxTexts) !Machine {
    var m: Machine = .{ .os = .linux, .arch = arch };
    var lines = std.mem.splitScalar(u8, t.meminfo, '\n');
    while (lines.next()) |line| {
        const kv = pair(line, ':') orelse continue;
        if (eql(kv[0], "MemTotal")) m.ram = int(kv[1]) * 1024;
    }
    m.cpu, m.cores = try cpuinfo(a, t.cpuinfo);
    var os = std.mem.splitScalar(u8, t.os_release, '\n');
    while (os.next()) |line| {
        const kv = pair(line, '=') orelse continue;
        if (eql(kv[0], "PRETTY_NAME")) m.os_version = try a.dupe(u8, std.mem.trim(u8, kv[1], "\"'"));
    }
    m.wsl = std.ascii.findIgnoreCase(t.kernel, "microsoft") != null or std.ascii.findIgnoreCase(t.kernel, "wsl") != null;
    var gfx: std.ArrayList([]const u8) = .empty;
    for (t.kfd) |props| if (gfxTarget(a, props)) |g| try gfx.append(a, g);
    var gpus: std.ArrayList(Gpu) = .empty;
    var amd: usize = 0;
    for (t.pci) |d| {
        // display controllers only (class 0x03xxxx): VGA, 3D (a data-center card) and other display
        const class = std.mem.trim(u8, d.class, " \r\n");
        if (!std.mem.startsWith(u8, class, "0x03")) continue;
        const vendor = machine.vendorOf(d.vendor);
        if (vendor == .other or vendor == .apple) continue;
        const pci = try std.fmt.allocPrint(a, "{s}:{s}", .{ hex4(d.vendor), hex4(d.device) });
        var g: Gpu = .{ .vendor = vendor, .name = "", .pci = pci };
        switch (vendor) {
            .nvidia => {
                for (t.nvidia) |info| if (eql(info.slot, d.slot)) {
                    const model = field(info.text, "Model") orelse "";
                    if (model.len > 0 and !std.ascii.eqlIgnoreCase(model, "unknown")) g.name = try a.dupe(u8, model);
                };
                if (g.name.len == 0) g.name = machine.pciName(pci) orelse try std.fmt.allocPrint(a, "NVIDIA GPU ({s})", .{pci});
                g = machine.withTable(g);
            },
            .amd => {
                if (amd < gfx.items.len) g.gfx = gfx.items[amd];
                amd += 1;
                g.memory = int(d.vram);
                g.name = if (g.gfx.len > 0) try std.fmt.allocPrint(a, "AMD GPU ({s})", .{g.gfx}) else try std.fmt.allocPrint(a, "AMD GPU ({s})", .{pci});
            },
            else => g.name = try std.fmt.allocPrint(a, "Intel GPU ({s})", .{pci}),
        }
        try gpus.append(a, g);
    }
    m.gpus = gpus.items;
    return m;
}

/// The CPU's name and logical cores: x86's "model name", else Arm's parts by count ("Cortex-X925 x10 + ...").
fn cpuinfo(a: Allocator, text: []const u8) !struct { []const u8, u32 } {
    var name: []const u8 = "";
    var cores: u32 = 0;
    var parts: [8]struct { id: u64 = 0, n: u32 = 0 } = @splat(.{});
    var kinds: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = pair(line, ':') orelse continue;
        if (eql(kv[0], "processor")) cores += 1;
        if (eql(kv[0], "model name") and name.len == 0) name = kv[1];
        if (!eql(kv[0], "CPU part")) continue;
        const id = std.fmt.parseInt(u64, trimHex(kv[1]), 16) catch continue;
        const at = for (parts[0..kinds], 0..) |p, i| {
            if (p.id == id) break i;
        } else blk: {
            if (kinds == parts.len) continue;
            parts[kinds] = .{ .id = id };
            kinds += 1;
            break :blk kinds - 1;
        };
        parts[at].n += 1;
    }
    if (name.len > 0 or kinds == 0) return .{ try a.dupe(u8, name), cores };
    var out: std.Io.Writer.Allocating = .init(a);
    for (parts[0..kinds], 0..) |p, i| {
        if (i > 0) try out.writer.writeAll(" + ");
        if (armPart(p.id)) |n| try out.writer.print("{s} x{d}", .{ n, p.n }) else try out.writer.print("Arm part 0x{x} x{d}", .{ p.id, p.n });
    }
    return .{ out.written(), cores };
}

/// Arm's own core part numbers (the GB10's two kinds, Grace's and the Neoverse server cores).
fn armPart(id: u64) ?[]const u8 {
    return switch (id) {
        0xd87 => "Cortex-X925",
        0xd85 => "Cortex-A725",
        0xd4f => "Neoverse-V2",
        0xd40 => "Neoverse-V1",
        0xd0c => "Neoverse-N1",
        0xd49 => "Neoverse-N2",
        else => null,
    };
}

/// A kfd node's "gfx_target_version 110501" as "gfx1151"; null for a CPU node (0) or none.
fn gfxTarget(a: Allocator, props: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, props, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, " \t\r");
        if (!eql(words.next() orelse continue, "gfx_target_version")) continue;
        const v = std.fmt.parseInt(u32, words.next() orelse return null, 10) catch return null;
        if (v == 0) return null;
        return std.fmt.allocPrint(a, "gfx{d}{x}{x}", .{ v / 10000, (v / 100) % 100, v % 100 }) catch null;
    }
    return null;
}

/// One adapter as DXGI lists it (IDXGIAdapter1::GetDesc1), its description already UTF-8.
pub const Adapter = struct { name: []const u8, vendor: u32, device: u32, memory: u64 = 0, software: bool = false };

pub const WindowsReadings = struct {
    /// The registry's ProcessorNameString.
    cpu: []const u8 = "",
    /// The registry's CurrentBuild ("26100").
    build: []const u8 = "",
    cores: u32 = 0,
    ram: u64 = 0,
    adapters: []const Adapter = &.{},
};

pub fn windows(a: Allocator, arch: []const u8, r: WindowsReadings) !Machine {
    var m: Machine = .{ .os = .windows, .arch = arch, .cpu = try a.dupe(u8, std.mem.trim(u8, r.cpu, " ")), .os_version = r.build, .cores = r.cores, .ram = r.ram };
    var gpus: std.ArrayList(Gpu) = .empty;
    for (r.adapters) |d| {
        if (d.software) continue; // the Basic Render Driver
        const vendor = switch (d.vendor) {
            0x10de => machine.Vendor.nvidia,
            0x1002 => machine.Vendor.amd,
            0x8086 => machine.Vendor.intel,
            else => continue, // remote and virtual display adapters
        };
        const pci = try std.fmt.allocPrint(a, "{x:0>4}:{x:0>4}", .{ d.vendor, d.device });
        // a twin entry for one card (one adapter per output on some drivers) is listed once
        const twin = for (gpus.items) |g| {
            if (std.mem.eql(u8, g.pci, pci) and std.mem.eql(u8, g.name, d.name) and g.memory == d.memory) break true;
        } else false;
        if (twin) continue;
        var g: Gpu = .{ .vendor = vendor, .name = try a.dupe(u8, d.name), .pci = pci, .memory = d.memory };
        if (vendor == .nvidia) {
            if (g.name.len == 0) g.name = machine.pciName(pci) orelse "NVIDIA GPU";
            g = machine.withTable(g);
            if (g.unified) g.memory = 0; // an integrated GB10's small dedicated carve-out is not its memory
        }
        try gpus.append(a, g);
    }
    m.gpus = gpus.items;
    return m;
}

fn pair(line: []const u8, sep: u8) ?[2][]const u8 {
    const at = std.mem.indexOfScalar(u8, line, sep) orelse return null;
    return .{ std.mem.trim(u8, line[0..at], " \t\r"), std.mem.trim(u8, line[at + 1 ..], " \t\r") };
}

/// The value of "Key: value" in a driver's text, tabs and spaces trimmed.
fn field(text: []const u8, key: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const kv = pair(line, ':') orelse continue;
        if (eql(kv[0], key)) return kv[1];
    }
    return null;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

/// The leading digits of `text` as a number ("131072000 kB": 131072000); 0 when it starts with none.
fn int(text: []const u8) u64 {
    const t = std.mem.trim(u8, text, " \t\r\n");
    var end: usize = 0;
    while (end < t.len and std.ascii.isDigit(t[end])) end += 1;
    return std.fmt.parseInt(u64, t[0..end], 10) catch 0;
}

fn trimHex(text: []const u8) []const u8 {
    const t = std.mem.trim(u8, text, " \t\r\n");
    return if (std.mem.startsWith(u8, t, "0x")) t[2..] else t;
}

/// A sysfs id ("0x10de\n") as its four hex digits ("10de").
fn hex4(text: []const u8) []const u8 {
    const t = trimHex(text);
    return t[0..@min(t.len, 4)];
}
