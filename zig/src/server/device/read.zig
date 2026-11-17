//! The quick scan: what the OS lists without any GPU driver (sysctl, /proc and /sys, DXGI and the registry).
const std = @import("std");
const builtin = @import("builtin");
const sys = @import("sys");
const machine = @import("machine.zig");
const parse = @import("parse.zig");
const Allocator = std.mem.Allocator;

pub const arch = @tagName(builtin.cpu.arch);

/// This machine as its OS lists it, in milliseconds: no driver is opened and no GPU is touched.
pub fn quick(a: Allocator, io: std.Io) !machine.Machine {
    return switch (builtin.os.tag) {
        .macos => parse.macos(a, arch, try macosText(a)),
        .linux => parse.linux(a, arch, try linuxTexts(a, io)),
        .windows => parse.windows(a, arch, try windowsReadings(a)),
        else => .{ .os = .other, .arch = arch, .ram = std.process.totalSystemMemory() catch 0 },
    };
}

/// The sysctl values a Mac's scan reads, written as `sysctl NAME...` prints them.
fn macosText(a: Allocator) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    var buf: [256]u8 = undefined;
    for ([_][:0]const u8{ "machdep.cpu.brand_string", "kern.osproductversion" }) |key| {
        var n = buf.len;
        if (std.c.sysctlbyname(key, &buf, &n, null, 0) != 0 or n == 0 or n > buf.len) continue;
        try w.print("{s}: {s}\n", .{ key, std.mem.trimEnd(u8, buf[0..n], "\x00") });
    }
    for ([_][:0]const u8{ "hw.memsize", "hw.ncpu", "hw.perflevel0.logicalcpu", "hw.perflevel1.logicalcpu", "iogpu.wired_limit_mb" }) |key| {
        var v: [8]u8 = @splat(0);
        var n: usize = v.len;
        if (std.c.sysctlbyname(key, &v, &n, null, 0) != 0) continue;
        const value: u64 = switch (n) {
            4 => std.mem.readInt(u32, v[0..4], .little),
            8 => std.mem.readInt(u64, &v, .little),
            else => continue,
        };
        try w.print("{s}: {d}\n", .{ key, value });
    }
    return out.written();
}

/// The /proc and /sys files a Linux scan reads; a missing one reads as empty.
fn linuxTexts(a: Allocator, io: std.Io) !parse.LinuxTexts {
    var t: parse.LinuxTexts = .{
        .meminfo = text(a, io, "/proc/meminfo"),
        .cpuinfo = text(a, io, "/proc/cpuinfo"),
        .os_release = text(a, io, "/etc/os-release"),
        .kernel = text(a, io, "/proc/sys/kernel/osrelease"),
    };
    var pci: std.ArrayList(parse.Pci) = .empty;
    var nvidia: std.ArrayList(parse.NvidiaInfo) = .empty;
    if (std.Io.Dir.cwd().openDir(io, "/sys/bus/pci/devices", .{ .iterate = true })) |dir| {
        var d = dir;
        defer d.close(io);
        var it = d.iterate();
        while (it.next(io) catch null) |entry| {
            const slot = try a.dupe(u8, entry.name);
            const class = text(a, io, try std.fmt.allocPrint(a, "/sys/bus/pci/devices/{s}/class", .{slot}));
            if (!std.mem.startsWith(u8, class, "0x03")) continue;
            const f: parse.Pci = .{
                .slot = slot,
                .class = class,
                .vendor = text(a, io, try std.fmt.allocPrint(a, "/sys/bus/pci/devices/{s}/vendor", .{slot})),
                .device = text(a, io, try std.fmt.allocPrint(a, "/sys/bus/pci/devices/{s}/device", .{slot})),
                .vram = text(a, io, try std.fmt.allocPrint(a, "/sys/bus/pci/devices/{s}/mem_info_vram_total", .{slot})),
            };
            try pci.append(a, f);
            if (machine.vendorOf(f.vendor) == .nvidia) try nvidia.append(a, .{ .slot = slot, .text = text(a, io, try std.fmt.allocPrint(a, "/proc/driver/nvidia/gpus/{s}/information", .{slot})) });
        }
    } else |_| {}
    std.mem.sort(parse.Pci, pci.items, {}, struct {
        fn less(_: void, x: parse.Pci, y: parse.Pci) bool {
            return std.mem.order(u8, x.slot, y.slot) == .lt;
        }
    }.less);
    t.pci = pci.items;
    t.nvidia = nvidia.items;
    var kfd: std.ArrayList([]const u8) = .empty;
    for (0..64) |node| {
        const props = text(a, io, try std.fmt.allocPrint(a, "/sys/class/kfd/kfd/topology/nodes/{d}/properties", .{node}));
        if (props.len == 0) break;
        try kfd.append(a, props);
    }
    t.kfd = kfd.items;
    return t;
}

/// A /proc or /sys file's text, streamed (procfs reports size 0); empty when it cannot be read.
fn text(a: Allocator, io: std.Io, path: []const u8) []const u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return "";
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch "";
}

const win = struct {
    const HKEY = std.os.windows.HKEY;
    const GUID = std.os.windows.GUID;
    const hklm = std.os.windows.HKEY_LOCAL_MACHINE;
    const rrf_rt_reg_sz: u32 = 0x2;
    extern "advapi32" fn RegGetValueW(key: HKEY, sub: [*:0]const u16, value: [*:0]const u16, flags: u32, kind: ?*u32, data: ?*anyopaque, size: ?*u32) callconv(.winapi) i32;

    /// DXGI_ADAPTER_DESC1.
    const Desc = extern struct {
        description: [128]u16,
        vendor: u32,
        device: u32,
        subsys: u32,
        revision: u32,
        dedicated_video: usize,
        dedicated_system: usize,
        shared_system: usize,
        luid_low: u32,
        luid_high: i32,
        flags: u32,
    };
    /// IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}.
    const iid_factory1: GUID = .{ .Data1 = 0x770aae78, .Data2 = 0xf26f, .Data3 = 0x4dba, .Data4 = .{ 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87 } };
    const Create = *const fn (*const GUID, *?*anyopaque) callconv(.winapi) i32;
    /// A COM object: its vtable's entries (Release 2, IDXGIFactory1's EnumAdapters1 12, IDXGIAdapter1's GetDesc1 10).
    const Com = extern struct { vtable: [*]const *const anyopaque };
    const Release = *const fn (*Com) callconv(.winapi) u32;
    const EnumAdapters1 = *const fn (*Com, u32, *?*Com) callconv(.winapi) i32;
    const GetDesc1 = *const fn (*Com, *Desc) callconv(.winapi) i32;

    fn regString(a: Allocator, comptime sub: []const u8, comptime value: []const u8) []const u8 {
        var buf: [256]u16 = undefined;
        var size: u32 = @sizeOf(@TypeOf(buf));
        const sub_w = std.unicode.utf8ToUtf16LeStringLiteral(sub);
        const value_w = std.unicode.utf8ToUtf16LeStringLiteral(value);
        if (RegGetValueW(hklm, sub_w, value_w, rrf_rt_reg_sz, null, &buf, &size) != 0 or size < 2) return "";
        const units = std.mem.sliceTo(buf[0 .. size / 2], 0);
        return std.unicode.utf16LeToUtf8Alloc(a, units) catch "";
    }

    /// Every adapter DXGI lists, present ones only (the registry also keeps removed cards' keys).
    fn adapters(a: Allocator) ![]const parse.Adapter {
        var lib = sys.DynLib.openRestricted("dxgi.dll") catch return &.{};
        defer lib.close();
        const create = lib.lookup(Create, "CreateDXGIFactory1") orelse return &.{};
        var raw: ?*anyopaque = null;
        if (create(&iid_factory1, &raw) < 0) return &.{};
        const factory: *Com = @ptrCast(@alignCast(raw orelse return &.{}));
        defer _ = @as(Release, @ptrCast(factory.vtable[2]))(factory);
        var out: std.ArrayList(parse.Adapter) = .empty;
        var i: u32 = 0;
        while (i < 16) : (i += 1) {
            var adapter: ?*Com = null;
            if (@as(EnumAdapters1, @ptrCast(factory.vtable[12]))(factory, i, &adapter) < 0) break;
            const ad = adapter orelse break;
            defer _ = @as(Release, @ptrCast(ad.vtable[2]))(ad);
            var d: Desc = undefined;
            if (@as(GetDesc1, @ptrCast(ad.vtable[10]))(ad, &d) < 0) continue;
            const name = std.unicode.utf16LeToUtf8Alloc(a, std.mem.sliceTo(&d.description, 0)) catch "";
            try out.append(a, .{ .name = name, .vendor = d.vendor, .device = d.device, .memory = d.dedicated_video, .software = (d.flags & 2) != 0 });
        }
        return out.items;
    }
};

fn windowsReadings(a: Allocator) !parse.WindowsReadings {
    return .{
        .cpu = win.regString(a, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString"),
        .build = win.regString(a, "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", "CurrentBuild"),
        .cores = @intCast(@min(std.Thread.getCpuCount() catch 0, 1 << 16)),
        .ram = std.process.totalSystemMemory() catch 0,
        .adapters = try win.adapters(a),
    };
}
