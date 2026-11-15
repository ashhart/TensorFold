//! This machine as the model commands see it: chip, memory, the bytes one GPU may hold and its memory bandwidth.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

pub const gib: u64 = 1 << 30;

pub const Backend = enum { metal, cuda, none };

pub const Machine = struct {
    backend: Backend = .none,
    chip_buf: [64]u8 = @splat(0),
    chip_len: u8 = 0,
    gpu_cores: u32 = 0,
    /// CUDA GPUs nvidia-smi lists; the limit is the first one's.
    gpus: u32 = 0,
    memory: u64 = 0,
    /// Bytes one GPU may keep resident: the Metal wired limit, a card's VRAM, or a share of a GB10's unified memory.
    gpu_limit: u64 = 0,
    /// iogpu.wired_limit_mb as set on a Mac; 0 means macOS chooses.
    wired_limit_mb: u64 = 0,
    unified: bool = false,
    /// Nominal memory bandwidth in bytes a second; 0 for a chip outside the table.
    bandwidth: u64 = 0,

    pub fn chip(m: *const Machine) []const u8 {
        return m.chip_buf[0..m.chip_len];
    }

    pub fn setChip(m: *Machine, name: []const u8) void {
        const t = std.mem.trim(u8, name, " \t\r\n\x00");
        m.chip_len = @intCast(@min(t.len, m.chip_buf.len));
        @memcpy(m.chip_buf[0..m.chip_len], t[0..m.chip_len]);
    }
};

/// macOS GPU limit when iogpu.wired_limit_mb is 0: fit to 107.5 of 128 GiB and 222.7 of 256 GiB, two thirds below that.
pub fn defaultGpuLimit(memory: u64) u64 {
    const fit = memory / 10 * 9 -| 768 * gib / 100;
    return @max(memory / 3 * 2, fit);
}

/// The GPU limit a wired-limit setting gives: the setting itself when non-zero, else macOS's default.
pub fn gpuLimitFor(memory: u64, wired_limit_mb: u64) u64 {
    return if (wired_limit_mb == 0) defaultGpuLimit(memory) else wired_limit_mb << 20;
}

/// Published memory bandwidth in GB/s; the first row whose name the chip contains wins, so variants precede base chips.
const bandwidth_gbs = [_]struct { []const u8, u64 }{
    .{ "M5 Ultra", 1200 },     .{ "M3 Ultra", 819 },  .{ "M2 Ultra", 800 },  .{ "M1 Ultra", 800 },
    .{ "M5 Max", 614 },        .{ "M4 Max", 546 },    .{ "M3 Max", 400 },    .{ "M2 Max", 400 },
    .{ "M1 Max", 400 },        .{ "M5 Pro", 307 },    .{ "M4 Pro", 273 },    .{ "M3 Pro", 150 },
    .{ "M2 Pro", 200 },        .{ "M1 Pro", 200 },    .{ "M5", 153 },        .{ "M4", 120 },
    .{ "M3", 100 },            .{ "M2", 100 },        .{ "M1", 68 },         .{ "GB10", 273 },
    .{ "RTX PRO 6000", 1792 }, .{ "RTX 5090", 1792 }, .{ "RTX 4090", 1008 }, .{ "RTX 3090", 936 },
};

/// The smaller-GPU Max chips have a narrower memory bus than the table's top configuration.
const smaller_max = [_]struct { []const u8, u32, u64 }{ .{ "M5 Max", 32, 460 }, .{ "M4 Max", 32, 410 }, .{ "M3 Max", 30, 300 } };

/// Nominal memory bandwidth (bytes a second) for the chips we serve; 0 when unknown (the engine measures it).
pub fn bandwidthOf(chip_name: []const u8) u64 {
    return bandwidthFor(chip_name, 0);
}

/// As bandwidthOf, with the GPU core count picking a Max chip's configuration when it is known.
pub fn bandwidthFor(chip_name: []const u8, gpu_cores: u32) u64 {
    for (smaller_max) |row| if (gpu_cores == row[1] and std.mem.indexOf(u8, chip_name, row[0]) != null) return row[2] * 1_000_000_000;
    for (bandwidth_gbs) |row| if (std.mem.indexOf(u8, chip_name, row[0]) != null) return row[1] * 1_000_000_000;
    return 0;
}

/// This machine: sysctl and ioreg on a Mac, nvidia-smi elsewhere; a field it cannot read stays 0.
pub fn detect(a: Allocator, io: std.Io) Machine {
    var m: Machine = .{ .memory = std.process.totalSystemMemory() catch 0 };
    if (builtin.os.tag == .macos) {
        m.backend = .metal;
        m.unified = true;
        var buf: [128]u8 = undefined;
        m.setChip(sysctlString("machdep.cpu.brand_string", &buf) orelse "");
        m.wired_limit_mb = sysctlInt("iogpu.wired_limit_mb");
        m.gpu_cores = gpuCores(run(a, io, &.{ "/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1" }) orelse "");
        m.gpu_limit = gpuLimitFor(m.memory, m.wired_limit_mb);
        m.bandwidth = bandwidthFor(m.chip(), m.gpu_cores);
        return m;
    }
    applyNvidia(&m, run(a, io, &.{ "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits" }) orelse "");
    return m;
}

/// nvidia-smi's `name, memory.total` rows (MiB); a GB10 or a GPU without its own memory takes macOS's share of RAM.
pub fn applyNvidia(m: *Machine, csv: []const u8) void {
    var lines = std.mem.splitScalar(u8, csv, '\n');
    while (lines.next()) |line| {
        const t = std.mem.trim(u8, line, " \t\r");
        const comma = std.mem.indexOfScalar(u8, t, ',') orelse continue;
        m.gpus += 1;
        if (m.gpus > 1) continue;
        m.backend = .cuda;
        m.setChip(t[0..comma]);
        const mib = std.fmt.parseInt(u64, std.mem.trim(u8, t[comma + 1 ..], " "), 10) catch 0;
        m.unified = mib == 0 or std.mem.indexOf(u8, m.chip(), "GB10") != null;
        m.gpu_limit = if (m.unified) defaultGpuLimit(m.memory) else mib << 20;
    }
    m.bandwidth = bandwidthOf(m.chip());
}

/// `"gpu-core-count" = N` from ioreg's AGXAccelerator entry; 0 when absent.
pub fn gpuCores(text: []const u8) u32 {
    const key = "\"gpu-core-count\" = ";
    const at = std.mem.indexOf(u8, text, key) orelse return 0;
    const rest = text[at + key.len ..];
    var end: usize = 0;
    while (end < rest.len and std.ascii.isDigit(rest[end])) end += 1;
    return std.fmt.parseInt(u32, rest[0..end], 10) catch 0;
}

fn run(a: Allocator, io: std.Io, argv: []const []const u8) ?[]const u8 {
    const r = std.process.run(a, io, .{
        .argv = argv,
        .stdout_limit = .limited(4 << 20),
        .stderr_limit = .limited(64 << 10),
        .timeout = .{ .duration = .{ .raw = .fromSeconds(10), .clock = .awake } },
    }) catch return null;
    return if (r.term.success()) r.stdout else null;
}

fn sysctlString(comptime name: [:0]const u8, buf: []u8) ?[]const u8 {
    var len: usize = buf.len;
    if (std.c.sysctlbyname(name, buf.ptr, &len, null, 0) != 0 or len == 0) return null;
    return std.mem.trimEnd(u8, buf[0..len], "\x00");
}

fn sysctlInt(comptime name: [:0]const u8) u64 {
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    return if (std.c.sysctlbyname(name, &v, &len, null, 0) == 0) v else 0;
}

test "bandwidth by chip name, with the smaller Max configurations and unknown chips at zero" {
    try std.testing.expectEqual(@as(u64, 819_000_000_000), bandwidthOf("Apple M3 Ultra"));
    try std.testing.expectEqual(@as(u64, 614_000_000_000), bandwidthFor("Apple M5 Max", 40));
    try std.testing.expectEqual(@as(u64, 460_000_000_000), bandwidthFor("Apple M5 Max", 32));
    try std.testing.expectEqual(@as(u64, 614_000_000_000), bandwidthFor("Apple M5 Max", 0));
    try std.testing.expectEqual(@as(u64, 273_000_000_000), bandwidthOf("Apple M4 Pro"));
    try std.testing.expectEqual(@as(u64, 120_000_000_000), bandwidthOf("Apple M4"));
    try std.testing.expectEqual(@as(u64, 153_000_000_000), bandwidthOf("Apple M5"));
    try std.testing.expectEqual(@as(u64, 273_000_000_000), bandwidthOf("NVIDIA GB10"));
    try std.testing.expectEqual(@as(u64, 0), bandwidthOf("Apple M9"));
}

test "nvidia-smi rows: a card's own memory, a GB10's share of RAM, and the GPU count" {
    var card: Machine = .{ .memory = 64 * gib };
    applyNvidia(&card, "NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 3090, 24576\n");
    try std.testing.expectEqualStrings("NVIDIA GeForce RTX 3090", card.chip());
    try std.testing.expectEqual(@as(u64, 24 * gib), card.gpu_limit);
    try std.testing.expectEqual(@as(u32, 2), card.gpus);
    try std.testing.expect(card.backend == .cuda and !card.unified and card.bandwidth == 936_000_000_000);

    var spark: Machine = .{ .memory = 128 * gib };
    applyNvidia(&spark, "NVIDIA GB10, [N/A]\n");
    try std.testing.expect(spark.unified and spark.gpu_limit == defaultGpuLimit(128 * gib));

    var none: Machine = .{};
    applyNvidia(&none, "");
    try std.testing.expect(none.backend == .none and none.gpus == 0);
}

test "detect reads this host: its memory, and on a Mac its chip and GPU limit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const m = detect(arena.allocator(), std.testing.io);
    try std.testing.expect(m.memory > 0);
    if (builtin.os.tag == .macos) try std.testing.expect(m.backend == .metal and m.chip().len > 0 and m.gpu_limit > 0);
}

test "ioreg's GPU core count" {
    try std.testing.expectEqual(@as(u32, 40), gpuCores("  | |   \"gpu-core-count\" = 40\n  | |   \"other\" = 1"));
    try std.testing.expectEqual(@as(u32, 0), gpuCores("no such key"));
}
