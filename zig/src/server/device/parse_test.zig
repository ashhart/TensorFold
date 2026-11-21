//! Each OS's recorded readings parse into the machine they came from.
const std = @import("std");
const parse = @import("parse.zig");
const machine = @import("machine.zig");
const fx = @import("test_fixtures.zig");
const t = std.testing;

test "device scan: a Mac's sysctl text is its chip, cores, unified memory and one Apple GPU" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const m = try parse.macos(arena.allocator(), "aarch64", fx.m3_ultra);
    try t.expectEqual(machine.Os.macos, m.os);
    try t.expectEqualStrings("Apple M3 Ultra", m.cpu);
    try t.expectEqualStrings("27.2.1", m.os_version);
    try t.expectEqual(@as(u64, 256) << 30, m.ram);
    try t.expectEqual(@as(u32, 32), m.cores);
    try t.expectEqual(@as(u32, 24), m.performance_cores);
    try t.expectEqual(@as(u32, 8), m.efficiency_cores);
    try t.expectEqual(@as(u64, 0), m.gpu_wired_limit);
    try t.expectEqual(@as(usize, 1), m.gpus.len);
    try t.expectEqual(machine.Vendor.apple, m.gpus[0].vendor);
    try t.expect(m.gpus[0].unified);
    try t.expectEqual(machine.AppleChip{ .generation = 3, .tier = .ultra }, machine.appleChip(m.cpu).?);
    const m5 = try parse.macos(arena.allocator(), "aarch64", fx.m5_ultra);
    try t.expectEqual(machine.AppleChip{ .generation = 5, .tier = .ultra }, machine.appleChip(m5.cpu).?);
    const wired = try parse.macos(arena.allocator(), "aarch64", fx.m1_max_wired);
    try t.expectEqual(@as(u64, 28672) << 20, wired.gpu_wired_limit);
    try t.expectEqual(machine.AppleChip{ .generation = 1, .tier = .max }, machine.appleChip(wired.cpu).?);
}

test "device scan: an Intel Mac lists no Apple GPU and is no Apple chip" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const m = try parse.macos(arena.allocator(), "x86_64", fx.intel_mac);
    try t.expectEqual(@as(usize, 0), m.gpus.len);
    try t.expect(machine.appleChip(m.cpu) == null);
    try t.expect(machine.appleChip("Apple") == null and machine.appleChip("Apple Mx") == null and machine.appleChip("AMD Radeon Pro") == null);
    try t.expectEqual(machine.AppleChip{ .generation = 4, .tier = .base }, machine.appleChip("Apple M4").?);
    try t.expectEqual(machine.AppleChip{ .generation = 4, .tier = .pro }, machine.appleChip("Apple M4 Pro").?);
}

test "device scan: a GB10's /proc and /sys readings, its driver's Unknown model named from the PCI id" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const m = try parse.linux(arena.allocator(), "aarch64", fx.gb10);
    try t.expectEqual(machine.Os.linux, m.os);
    try t.expectEqual(@as(u64, 127535340) * 1024, m.ram);
    try t.expectEqual(@as(u32, 20), m.cores);
    try t.expectEqualStrings("Cortex-A725 x10 + Cortex-X925 x10", m.cpu);
    try t.expectEqualStrings("Ubuntu 24.04.5 LTS", m.os_version);
    try t.expect(!m.wsl);
    try t.expectEqual(@as(usize, 1), m.gpus.len); // the PCI bridge (class 0x0604) is not a GPU
    const g = m.gpus[0];
    try t.expectEqualStrings("NVIDIA GB10", g.name);
    try t.expectEqualStrings("10de:2e12", g.pci);
    try t.expectEqual(@as(u32, 121), g.sm);
    try t.expect(g.unified);
    try t.expectEqual(@as(u64, 0), g.memory);
    try t.expectEqual(machine.Source.table, g.source);
    try t.expectEqualStrings("NVIDIA GB10 (DGX Spark class)", machine.platform(m).?);
}

test "device scan: an RTX 3090 desktop and a Hopper server, named by the driver's information file" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const m = try parse.linux(arena.allocator(), "x86_64", fx.rtx3090);
    try t.expectEqualStrings("AMD Ryzen 9 5950X 16-Core Processor", m.cpu);
    try t.expectEqual(@as(u32, 2), m.cores);
    try t.expectEqual(@as(usize, 1), m.gpus.len); // the HDMI audio function (class 0x0403) is not a GPU
    try t.expectEqualStrings("NVIDIA GeForce RTX 3090", m.gpus[0].name);
    try t.expectEqual(@as(u32, 86), m.gpus[0].sm);
    try t.expectEqual(@as(u64, 24) << 30, m.gpus[0].memory);
    const h = try parse.linux(arena.allocator(), "x86_64", fx.h100);
    try t.expectEqual(@as(usize, 1), h.gpus.len); // a 3D controller (0x0302) counts
    try t.expectEqual(@as(u32, 90), h.gpus[0].sm);
    try t.expectEqual(@as(u64, 0), h.gpus[0].memory); // the table names no memory for it
}

test "device scan: WSL is told by its kernel, and an AMD APU's gfx target comes from kfd" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const w = try parse.linux(arena.allocator(), "x86_64", fx.wsl);
    try t.expect(w.wsl);
    try t.expectEqual(@as(usize, 0), w.gpus.len);
    const a = try parse.linux(arena.allocator(), "x86_64", fx.strix_halo);
    try t.expectEqual(@as(usize, 1), a.gpus.len);
    try t.expectEqual(machine.Vendor.amd, a.gpus[0].vendor);
    try t.expectEqualStrings("gfx1151", a.gpus[0].gfx);
    try t.expectEqualStrings("AMD GPU (gfx1151)", a.gpus[0].name);
    try t.expectEqual(@as(u64, 96) << 30, a.gpus[0].memory);
}

test "device scan: Windows' DXGI adapters, the software renderer left out, an integrated GB10's carve-out dropped" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const m = try parse.windows(arena.allocator(), "x86_64", fx.rtx5090);
    try t.expectEqual(machine.Os.windows, m.os);
    try t.expectEqualStrings("AMD Ryzen 9 9950X3D 16-Core Processor", m.cpu);
    try t.expectEqual(@as(usize, 1), m.gpus.len);
    try t.expectEqualStrings("NVIDIA GeForce RTX 5090", m.gpus[0].name);
    try t.expectEqualStrings("10de:2b85", m.gpus[0].pci);
    try t.expectEqual(@as(u32, 120), m.gpus[0].sm);
    try t.expectEqual(@as(u64, 34190917632), m.gpus[0].memory); // DXGI's own count, not the table's
    var buf: [64]u8 = undefined;
    try t.expectEqualStrings("Windows 11 (build 26100)", machine.osName(&buf, m));
    const s = try parse.windows(arena.allocator(), "aarch64", fx.windows_gb10);
    try t.expect(s.gpus[0].unified);
    try t.expectEqual(@as(u64, 0), s.gpus[0].memory);
    try t.expectEqual(@as(u32, 121), s.gpus[0].sm);
}

test "device scan: the name table matches whole words and keeps model numbers apart" {
    try t.expectEqual(@as(u32, 120), machine.knownNvidia("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition").?.sm);
    try t.expectEqual(@as(u32, 96), machine.knownNvidia("NVIDIA RTX PRO 6000 Blackwell Server Edition").?.memory_gib);
    try t.expectEqual(@as(u32, 89), machine.knownNvidia("NVIDIA RTX 5000 Ada Generation").?.sm);
    try t.expectEqual(@as(u32, 75), machine.knownNvidia("Quadro RTX 5000").?.sm);
    try t.expectEqual(@as(u32, 86), machine.knownNvidia("NVIDIA RTX A6000").?.sm);
    try t.expectEqual(@as(u32, 80), machine.knownNvidia("NVIDIA A100-SXM4-80GB").?.sm);
    try t.expectEqual(@as(u32, 86), machine.knownNvidia("NVIDIA A10").?.sm);
    try t.expectEqual(@as(u32, 89), machine.knownNvidia("NVIDIA L4").?.sm);
    try t.expectEqual(@as(u32, 89), machine.knownNvidia("NVIDIA L40S").?.sm);
    try t.expectEqual(@as(u32, 89), machine.knownNvidia("NVIDIA GeForce RTX 4090 D").?.sm);
    try t.expectEqual(@as(u32, 103), machine.knownNvidia("NVIDIA GB300").?.sm);
    try t.expect(machine.knownNvidia("NVIDIA A10G") == null);
    try t.expectEqualStrings("RTX 5090", machine.shortName("NVIDIA GeForce RTX 5090"));
}

test "device scan: the fingerprint follows the hardware the OS lists and nothing the driver adds" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parse.linux(a, "aarch64", fx.gb10);
    const fp = machine.fingerprint(m);
    try t.expectEqualSlices(u8, &fp, &machine.fingerprint(try parse.linux(a, "aarch64", fx.gb10)));
    var more = m;
    more.ram += 64 << 30;
    try t.expect(!std.mem.eql(u8, &fp, &machine.fingerprint(more)));
    var other = m;
    other.gpus = &.{};
    try t.expect(!std.mem.eql(u8, &fp, &machine.fingerprint(other)));
    var drifted = m;
    drifted.ram += 100 << 20; // MemTotal moving by a few MiB across kernels keeps the GiB
    try t.expectEqualSlices(u8, &fp, &machine.fingerprint(drifted));
}
