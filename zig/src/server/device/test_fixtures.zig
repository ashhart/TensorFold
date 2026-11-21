//! Raw readings in each source's own format: recorded on our machines (UUIDs and BIOS ids zeroed), or built to match.
const std = @import("std");
const parse = @import("parse.zig");

/// The Studio, an M3 Ultra with 256 GB (recorded 13 Nov 2026 with `sysctl NAME...`).
pub const m3_ultra =
    \\machdep.cpu.brand_string: Apple M3 Ultra
    \\kern.osproductversion: 27.2.1
    \\hw.memsize: 274877906944
    \\hw.ncpu: 32
    \\hw.perflevel0.logicalcpu: 24
    \\hw.perflevel1.logicalcpu: 8
    \\iogpu.wired_limit_mb: 0
;

/// An M5 Ultra with 256 GB (recorded 13 Nov 2026).
pub const m5_ultra =
    \\machdep.cpu.brand_string: Apple M5 Ultra
    \\kern.osproductversion: 27.0.1
    \\hw.memsize: 274877906944
    \\hw.ncpu: 36
    \\hw.perflevel0.logicalcpu: 12
    \\hw.perflevel1.logicalcpu: 24
    \\iogpu.wired_limit_mb: 0
;

/// A 32 GB M1 Max whose GPU wired limit was raised to 28 GB.
pub const m1_max_wired =
    \\machdep.cpu.brand_string: Apple M1 Max
    \\kern.osproductversion: 15.6
    \\hw.memsize: 34359738368
    \\hw.ncpu: 10
    \\hw.perflevel0.logicalcpu: 8
    \\hw.perflevel1.logicalcpu: 2
    \\iogpu.wired_limit_mb: 28672
;

/// An Intel Mac: no perflevel keys, no iogpu key.
pub const intel_mac =
    \\machdep.cpu.brand_string: Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz
    \\kern.osproductversion: 15.6
    \\hw.memsize: 34359738368
    \\hw.ncpu: 16
;

/// A GB10 (DGX Spark class) as recorded 13 Nov 2026: 20 Arm cores, the open kernel module's "Model: Unknown".
pub const gb10: parse.LinuxTexts = .{
    .meminfo = "MemTotal:       127535340 kB\nMemFree:        13053236 kB\nMemAvailable:   12700884 kB\n",
    .cpuinfo = gb10_cpuinfo,
    .os_release = "PRETTY_NAME=\"Ubuntu 24.04.5 LTS\"\nNAME=\"Ubuntu\"\nVERSION_ID=\"24.04\"\n",
    .kernel = "6.17.0-1029-nvidia\n",
    .pci = &.{
        .{ .slot = "0000:00:00.0", .class = "0x060400\n", .vendor = "0x10de\n", .device = "0x22ce\n" },
        .{ .slot = "000f:01:00.0", .class = "0x030000\n", .vendor = "0x10de\n", .device = "0x2e12\n" },
    },
    .nvidia = &.{.{ .slot = "000f:01:00.0", .text = gb10_information }},
};

const gb10_information =
    "Model: \t\t Unknown\nIRQ:   \t\t 485\nGPU UUID: \t GPU-00000000-0000-0000-0000-000000000000\n" ++
    "Video BIOS: \t ??.??.??.??.??\nBus Type: \t PCIe\nDMA Size: \t 41 bits\nDMA Mask: \t 0x1ffffffffff\n" ++
    "Bus Location: \t 000f:01:00.0\nDevice Minor: \t 0\nGPU Firmware: \t 580.173.02\nGPU Excluded:\t No\n";

const gb10_cpuinfo = blk: {
    var text: []const u8 = "";
    for (0..20) |i| text = text ++ std.fmt.comptimePrint(
        "processor\t: {d}\nBogoMIPS\t: 2000.00\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU part\t: 0x{s}\n\n",
        .{ i, if (i < 10) "d85" else "d87" },
    );
    break :blk text;
};

/// An x86_64 desktop with an RTX 3090 (its HDMI audio function alongside), in the driver's formats.
pub const rtx3090: parse.LinuxTexts = .{
    .meminfo = "MemTotal:       65756196 kB\nMemAvailable:   60123456 kB\n",
    .cpuinfo = "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 5950X 16-Core Processor\n\n" ++
        "processor\t: 1\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 9 5950X 16-Core Processor\n\n",
    .os_release = "NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 22.04.4 LTS\"\n",
    .kernel = "6.8.0-45-generic\n",
    .pci = &.{
        .{ .slot = "0000:0b:00.0", .class = "0x030000\n", .vendor = "0x10de\n", .device = "0x2204\n" },
        .{ .slot = "0000:0b:00.1", .class = "0x040300\n", .vendor = "0x10de\n", .device = "0x1aef\n" },
    },
    .nvidia = &.{.{ .slot = "0000:0b:00.0", .text = "Model: \t\t NVIDIA GeForce RTX 3090\nIRQ:   \t\t 112\nBus Type: \t PCIe\n" }},
};

/// A data-center Hopper card: a 3D controller (class 0x0302), as servers list them.
pub const h100: parse.LinuxTexts = .{
    .meminfo = "MemTotal:       2113529856 kB\n",
    .cpuinfo = "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum 8480+\n\n",
    .os_release = "PRETTY_NAME=\"Ubuntu 22.04.4 LTS\"\n",
    .pci = &.{.{ .slot = "0000:18:00.0", .class = "0x030200\n", .vendor = "0x10de\n", .device = "0x2330\n" }},
    .nvidia = &.{.{ .slot = "0000:18:00.0", .text = "Model: \t\t NVIDIA H100 80GB HBM3\nBus Type: \t PCIe\n" }},
};

/// WSL 2: the GPU reaches Linux through /dev/dxg, so the PCI listing has none.
pub const wsl: parse.LinuxTexts = .{
    .meminfo = "MemTotal:       32768000 kB\n",
    .cpuinfo = "processor\t: 0\nmodel name\t: AMD Ryzen 9 9950X 16-Core Processor\n\n",
    .os_release = "PRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\n",
    .kernel = "6.6.87-microsoft-standard-WSL2\n",
};

/// An AMD Ryzen AI Max (Strix Halo) APU: a kfd CPU node, then its gfx1151 GPU node.
pub const strix_halo: parse.LinuxTexts = .{
    .meminfo = "MemTotal:       131072000 kB\n",
    .cpuinfo = "processor\t: 0\nmodel name\t: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S\n\n",
    .os_release = "PRETTY_NAME=\"Fedora Linux 42 (Workstation Edition)\"\n",
    .pci = &.{.{ .slot = "0000:c5:00.0", .class = "0x038000\n", .vendor = "0x1002\n", .device = "0x1586\n", .vram = "103079215104\n" }},
    .kfd = &.{ "cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\n", "cpu_cores_count 0\nsimd_count 80\ngfx_target_version 110501\n" },
};

/// Windows 11 with an RTX 5090, as DXGI lists it: the card, then the Basic Render Driver (software).
pub const rtx5090: parse.WindowsReadings = .{
    .cpu = "AMD Ryzen 9 9950X3D 16-Core Processor           ",
    .build = "26100",
    .cores = 32,
    .ram = 68451041280,
    .adapters = &.{
        .{ .name = "NVIDIA GeForce RTX 5090", .vendor = 0x10de, .device = 0x2b85, .memory = 34190917632 },
        .{ .name = "Microsoft Basic Render Driver", .vendor = 0x1414, .device = 0x8c, .software = true },
    },
};

/// Windows on Arm with an integrated GB10, built to match DXGI's fields: its small dedicated carve-out.
pub const windows_gb10: parse.WindowsReadings = .{
    .cpu = "Arm processor",
    .build = "26200",
    .cores = 20,
    .ram = 137438953472,
    .adapters = &.{.{ .name = "NVIDIA GB10", .vendor = 0x10de, .device = 0x2e12, .memory = 536870912 }},
};
