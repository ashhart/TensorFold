//! The machine a `tensorfold device` scan finds: OS, CPU, RAM and GPUs; never a serial, host or user name, or path.
const std = @import("std");

pub const Os = enum { macos, linux, windows, other };
pub const Vendor = enum { nvidia, amd, intel, apple, other };
/// Where a GPU's facts came from: the OS's own listing, the known-card table below, or the GPU's driver.
pub const Source = enum { os, table, driver };

pub const Gpu = struct {
    vendor: Vendor,
    name: []const u8,
    /// Device memory in bytes; 0 for unified memory (the system's RAM) or when unknown.
    memory: u64 = 0,
    unified: bool = false,
    /// NVIDIA's compute capability as 10 * major + minor (121 is sm_121); 0 when unknown or not NVIDIA.
    sm: u32 = 0,
    /// AMD's gfx target ("gfx1151"); empty when unknown or not AMD.
    gfx: []const u8 = "",
    /// The PCI vendor and device ids ("10de:2b85"); empty where the OS lists none (Apple, a driver's list).
    pci: []const u8 = "",
    /// Where `sm`, `memory` and `unified` came from.
    source: Source = .os,
};

pub const Machine = struct {
    os: Os,
    os_version: []const u8 = "",
    arch: []const u8,
    cpu: []const u8 = "",
    cores: u32 = 0,
    performance_cores: u32 = 0,
    efficiency_cores: u32 = 0,
    ram: u64 = 0,
    /// macOS's GPU wired limit (iogpu.wired_limit_mb) in bytes; 0 when unset or not a Mac.
    gpu_wired_limit: u64 = 0,
    /// Linux under WSL 2: the GPUs reach it through the driver alone, so the OS's own listing has none.
    wsl: bool = false,
    gpus: []const Gpu = &.{},
};

pub const Tier = enum { base, pro, max, ultra };
pub const AppleChip = struct { generation: u32, tier: Tier };

/// "Apple M3 Ultra" as generation 3, Ultra; null for any other CPU name (an Intel Mac's included).
pub fn appleChip(name: []const u8) ?AppleChip {
    var words = std.mem.tokenizeScalar(u8, name, ' ');
    if (!std.ascii.eqlIgnoreCase(words.next() orelse return null, "apple")) return null;
    const m = words.next() orelse return null;
    if (m.len < 2 or (m[0] != 'M' and m[0] != 'm')) return null;
    const generation = std.fmt.parseInt(u32, m[1..], 10) catch return null;
    if (generation == 0) return null;
    const tier: Tier = if (words.next()) |t|
        (if (std.ascii.eqlIgnoreCase(t, "ultra")) .ultra else if (std.ascii.eqlIgnoreCase(t, "max")) .max else if (std.ascii.eqlIgnoreCase(t, "pro")) .pro else .base)
    else
        .base;
    return .{ .generation = generation, .tier = tier };
}

/// A card the table knows: its compute capability and, for the named platforms, its memory.
pub const Known = struct { sm: u32, memory_gib: u32 = 0, unified: bool = false };

/// Name patterns, first match wins: every word of `all` must appear in the name as a whole word, in any case.
const Pattern = struct { all: []const []const u8, known: Known };
const patterns = [_]Pattern{
    .{ .all = &.{"GB10"}, .known = .{ .sm = 121, .unified = true } },
    .{ .all = &.{ "RTX", "PRO", "6000", "Blackwell" }, .known = .{ .sm = 120, .memory_gib = 96 } },
    .{ .all = &.{ "RTX", "PRO", "Blackwell" }, .known = .{ .sm = 120 } },
    .{ .all = &.{"Thor"}, .known = .{ .sm = 110, .unified = true } },
    .{ .all = &.{"GB300"}, .known = .{ .sm = 103 } },
    .{ .all = &.{"B300"}, .known = .{ .sm = 103 } },
    .{ .all = &.{"GB200"}, .known = .{ .sm = 100 } },
    .{ .all = &.{"B200"}, .known = .{ .sm = 100 } },
    .{ .all = &.{"B100"}, .known = .{ .sm = 100 } },
    .{ .all = &.{"GH200"}, .known = .{ .sm = 90 } },
    .{ .all = &.{"H100"}, .known = .{ .sm = 90 } },
    .{ .all = &.{"H200"}, .known = .{ .sm = 90 } },
    .{ .all = &.{"H800"}, .known = .{ .sm = 90 } },
    .{ .all = &.{"H20"}, .known = .{ .sm = 90 } },
    .{ .all = &.{ "Quadro", "RTX" }, .known = .{ .sm = 75 } },
    .{ .all = &.{"Ada"}, .known = .{ .sm = 89 } },
    .{ .all = &.{ "RTX", "5090" }, .known = .{ .sm = 120, .memory_gib = 32 } },
    .{ .all = &.{ "RTX", "5080" }, .known = .{ .sm = 120, .memory_gib = 16 } },
    .{ .all = &.{ "RTX", "5070" }, .known = .{ .sm = 120 } },
    .{ .all = &.{ "RTX", "5060" }, .known = .{ .sm = 120 } },
    .{ .all = &.{ "RTX", "4090" }, .known = .{ .sm = 89, .memory_gib = 24 } },
    .{ .all = &.{ "RTX", "4080" }, .known = .{ .sm = 89 } },
    .{ .all = &.{ "RTX", "4070" }, .known = .{ .sm = 89 } },
    .{ .all = &.{ "RTX", "4060" }, .known = .{ .sm = 89 } },
    .{ .all = &.{"L40S"}, .known = .{ .sm = 89 } },
    .{ .all = &.{"L40"}, .known = .{ .sm = 89 } },
    .{ .all = &.{"L4"}, .known = .{ .sm = 89 } },
    .{ .all = &.{ "RTX", "3090" }, .known = .{ .sm = 86, .memory_gib = 24 } },
    .{ .all = &.{ "RTX", "3080" }, .known = .{ .sm = 86 } },
    .{ .all = &.{ "RTX", "3070" }, .known = .{ .sm = 86 } },
    .{ .all = &.{ "RTX", "3060" }, .known = .{ .sm = 86 } },
    .{ .all = &.{"A100"}, .known = .{ .sm = 80 } },
    .{ .all = &.{"A800"}, .known = .{ .sm = 80 } },
    .{ .all = &.{"A30"}, .known = .{ .sm = 80 } },
    .{ .all = &.{"A40"}, .known = .{ .sm = 86 } },
    .{ .all = &.{"A10"}, .known = .{ .sm = 86 } },
    .{ .all = &.{"A6000"}, .known = .{ .sm = 86 } },
    .{ .all = &.{"A5000"}, .known = .{ .sm = 86 } },
    .{ .all = &.{"A4000"}, .known = .{ .sm = 86 } },
    .{ .all = &.{ "RTX", "2080" }, .known = .{ .sm = 75 } },
    .{ .all = &.{ "RTX", "2070" }, .known = .{ .sm = 75 } },
    .{ .all = &.{ "RTX", "2060" }, .known = .{ .sm = 75 } },
    .{ .all = &.{"T4"}, .known = .{ .sm = 75 } },
    .{ .all = &.{"V100"}, .known = .{ .sm = 70 } },
};

/// What the table says of an NVIDIA card by its name ("NVIDIA GeForce RTX 5090"); null for a card it does not know.
pub fn knownNvidia(name: []const u8) ?Known {
    for (patterns) |p| {
        const all = for (p.all) |want| {
            if (!hasWord(name, want)) break false;
        } else true;
        if (all) return p.known;
    }
    return null;
}

/// PCI device ids whose driver may give no name (GB10's open kernel module reports "Model: Unknown").
const pci_names = [_]struct { []const u8, []const u8 }{
    .{ "10de:2e12", "NVIDIA GB10" },
};

/// The name of a card the OS lists only by its PCI ids, when the table knows it.
pub fn pciName(pci: []const u8) ?[]const u8 {
    for (pci_names) |row| if (std.ascii.eqlIgnoreCase(row[0], pci)) return row[1];
    return null;
}

/// `word` in `text` as a whole word, case-insensitive: "L4" is in "NVIDIA L4", not in "NVIDIA L40S".
fn hasWord(text: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeAny(u8, text, " -_(),/");
    while (it.next()) |w| if (std.ascii.eqlIgnoreCase(w, word)) return true;
    return false;
}

/// The vendor of a PCI vendor id ("10de" or "0x10de").
pub fn vendorOf(id: []const u8) Vendor {
    const hex = if (std.mem.startsWith(u8, id, "0x")) id[2..] else id;
    const v = std.fmt.parseInt(u16, std.mem.trim(u8, hex, " \r\n"), 16) catch return .other;
    return switch (v) {
        0x10de => .nvidia,
        0x1002 => .amd,
        0x8086 => .intel,
        0x106b => .apple,
        else => .other,
    };
}

/// A GPU with the table's facts filled in where the OS gave none.
pub fn withTable(g: Gpu) Gpu {
    if (g.vendor != .nvidia) return g;
    const k = knownNvidia(g.name) orelse return g;
    var out = g;
    if (out.sm == 0) out.sm = k.sm;
    if (out.memory == 0 and !k.unified) out.memory = @as(u64, k.memory_gib) << 30;
    out.unified = out.unified or k.unified;
    if (out.source == .os) out.source = .table;
    return out;
}

/// The name a person knows a GPU by: "RTX 5090", "GB10", "Apple M3 Ultra".
pub fn shortName(name: []const u8) []const u8 {
    var s = std.mem.trim(u8, name, " ");
    for ([_][]const u8{ "NVIDIA ", "GeForce " }) |prefix| {
        if (std.ascii.startsWithIgnoreCase(s, prefix)) s = s[prefix.len..];
    }
    return s;
}

/// A hardware fingerprint over the OS's own listing: a rescan follows a new GPU, more RAM or another OS.
pub fn fingerprint(m: Machine) [16]u8 {
    var h = std.hash.Wyhash.init(0x54465f444556);
    h.update(@tagName(m.os));
    h.update(m.arch);
    h.update(m.cpu);
    h.update(std.mem.asBytes(&m.cores));
    const gib: u64 = (m.ram + (1 << 29)) >> 30;
    h.update(std.mem.asBytes(&gib));
    h.update(std.mem.asBytes(&m.wsl));
    for (m.gpus) |g| {
        h.update(@tagName(g.vendor));
        h.update(g.pci);
        h.update(g.name);
    }
    const x = h.final();
    return std.fmt.bytesToHex(std.mem.asBytes(&x), .lower);
}

/// The named platform a machine is, for its one-line summary: "NVIDIA GB10 (DGX Spark class)", "Apple M3 Ultra".
pub fn platform(m: Machine) ?[]const u8 {
    for (m.gpus) |g| if (g.vendor == .nvidia and g.sm == 121 and hasWord(g.name, "GB10")) return "NVIDIA GB10 (DGX Spark class)";
    if (appleChip(m.cpu) != null) return m.cpu;
    return null;
}

/// The OS as a person names it: "macOS 27.2", "Windows 11 (build 26100)", "Ubuntu 24.04.5 LTS".
pub fn osName(buf: []u8, m: Machine) []const u8 {
    const v = m.os_version;
    return switch (m.os) {
        .macos => std.fmt.bufPrint(buf, "macOS{s}{s}", .{ if (v.len > 0) " " else "", v }) catch "macOS",
        .windows => blk: {
            const build = std.fmt.parseInt(u32, v, 10) catch 0;
            const name = if (build >= 22000) "Windows 11" else if (build > 0) "Windows 10" else "Windows";
            if (build == 0) break :blk name;
            break :blk std.fmt.bufPrint(buf, "{s} (build {d})", .{ name, build }) catch name;
        },
        .linux => if (v.len > 0) (std.fmt.bufPrint(buf, "{s}{s}", .{ v, if (m.wsl) " (WSL)" else "" }) catch v) else if (m.wsl) "Linux (WSL)" else "Linux",
        .other => if (v.len > 0) v else "an OS TensorFold does not build for",
    };
}
