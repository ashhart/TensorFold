//! Prototype-style runtime view: one Runtime over a Context and its Stream, with plain buffers, modules and kernels.

const std = @import("std");
const abi = @import("abi.zig");
const xpu = @import("driver.zig");
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;

/// Per-kernel launch timing (XPU_PROF=1): every launch is synced and timed, so entries include the sync round trip.
pub var prof_on: bool = false;
const ProfEntry = struct { name: ?[*:0]const u8 = null, ns: u64 = 0, calls: u64 = 0 };
var prof_tab: [160]ProfEntry = @splat(.{});

fn profNow() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Prints and clears the launch profile (calls, total ms, us per call), largest first.
pub fn profDump(label: []const u8) void {
    var used: usize = 0;
    var total: u64 = 0;
    for (prof_tab) |e| {
        if (e.name == null) break;
        used += 1;
        total += e.ns;
    }
    std.debug.print("-- launch profile {s}: {d} kernels, {d:.2} ms total\n", .{ label, used, @as(f64, @floatFromInt(total)) / 1e6 });
    var done: usize = 0;
    while (done < used) : (done += 1) {
        var best: usize = used;
        for (prof_tab[0..used], 0..) |e, i| if (e.calls > 0 and (best == used or e.ns > prof_tab[best].ns)) {
            best = i;
        };
        if (best == used) break;
        const e = prof_tab[best];
        std.debug.print("   {s:<28} {d:>6} calls {d:>9.3} ms {d:>8.1} us/call\n", .{ std.mem.span(e.name.?), e.calls, @as(f64, @floatFromInt(e.ns)) / 1e6, @as(f64, @floatFromInt(e.ns)) / 1e3 / @as(f64, @floatFromInt(e.calls)) });
        prof_tab[best].calls = 0;
    }
    prof_tab = @splat(.{});
}

/// Device bytes allocated through Runtime.alloc (now and peak), for memory reports.
pub var alloc_now: u64 = 0;
pub var alloc_peak: u64 = 0;
/// ARC_ALLOC_TRACE=1 prints every allocation of 32 MB or more.
var alloc_trace: bool = false;
/// Pinned host bytes from Runtime.allocHost (tables the kernels read over PCIe instead of holding them in VRAM).
pub var host_now: u64 = 0;
/// ARC_VRAM_CAP_GB: allocations taking counter + allowance (ARC_VRAM_OVERHEAD_GB, default 0.5) past the cap fail.
var cap_bytes: u64 = 0;
var cap_overhead: u64 = 0;
/// Device memory free for this process, bytes: cap (ARC_VRAM_CAP_GB or ARC_VRAM_TOTAL_GB, default 31) minus allocated.
pub fn vramFree() u64 {
    const total: u64 = if (cap_bytes > 0) cap_bytes else blk: {
        const g = if (std.c.getenv("ARC_VRAM_TOTAL_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 31) else 31;
        break :blk @intFromFloat(g * 1e9);
    };
    const oh: u64 = if (cap_bytes > 0) cap_overhead else 500_000_000;
    return total -| (alloc_now + oh);
}

/// Largest drm-total-vram0 sampled by this process (ARC_VRAM_DEBUG=1), bytes.
var drm_peak: u64 = 0;
var drm_sampled_at: u64 = 0;

extern "c" fn opendir(path: [*:0]const u8) ?*anyopaque;
extern "c" fn readdir(d: *anyopaque) ?*const Dirent;
extern "c" fn closedir(d: *anyopaque) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn getpid() c_int;
extern "c" fn atexit(f: *const fn () callconv(.c) void) c_int;
const Dirent = extern struct { ino: u64, off: i64, reclen: u16, typ: u8, name: [256]u8 };

fn zpath(buf: []u8, comptime f: []const u8, args: anytype) ?[*:0]const u8 {
    const t = std.fmt.bufPrint(buf, f ++ "\x00", args) catch return null;
    return @ptrCast(t.ptr);
}

fn fieldKiB(t: []const u8, key: []const u8) ?u64 {
    const i = std.mem.indexOf(u8, t, key) orelse return null;
    var j = i + key.len;
    while (j < t.len and (t[j] == ' ' or t[j] == '\t')) j += 1;
    var e = j;
    while (e < t.len and t[e] >= '0' and t[e] <= '9') e += 1;
    return std.fmt.parseInt(u64, t[j..e], 10) catch null;
}

/// KiB of device memory held on the xe GPU by the DRM clients of one process (drm-total-vram0 of its fds).
pub fn pidVramKiB(pid: c_int) u64 {
    var total: u64 = 0;
    var dp: [48]u8 = undefined;
    const dir = opendir(zpath(&dp, "/proc/{d}/fdinfo", .{pid}) orelse return 0) orelse return 0;
    var seen: [32]u64 = undefined;
    var nseen: usize = 0;
    while (readdir(dir)) |fe| {
        const fname = std.mem.sliceTo(&fe.name, 0);
        if (fname.len == 0 or fname[0] == '.') continue;
        var fp: [96]u8 = undefined;
        const fd = open(zpath(&fp, "/proc/{d}/fdinfo/{s}", .{ pid, fname }) orelse continue, 0);
        if (fd < 0) continue;
        var buf: [2048]u8 = undefined;
        const n = read(fd, &buf, buf.len);
        _ = close(fd);
        if (n <= 0) continue;
        const t = buf[0..@intCast(n)];
        if (std.mem.indexOf(u8, t, "drm-driver:\txe") == null) continue;
        const cid = fieldKiB(t, "drm-client-id:") orelse continue;
        if (std.mem.indexOfScalar(u64, seen[0..nseen], cid) != null) continue;
        if (nseen < seen.len) {
            seen[nseen] = cid;
            nseen += 1;
        }
        total += fieldKiB(t, "drm-total-vram0:") orelse 0;
    }
    _ = closedir(dir);
    return total;
}

/// KiB of device memory that OTHER processes of this user hold on the xe GPU.
pub fn otherVramKiB() u64 {
    const me = getpid();
    var total: u64 = 0;
    const proc = opendir("/proc") orelse return 0;
    defer _ = closedir(proc);
    while (readdir(proc)) |pe| {
        const pid = std.fmt.parseInt(c_int, std.mem.sliceTo(&pe.name, 0), 10) catch continue;
        if (pid != me) total += pidVramKiB(pid);
    }
    return total;
}

/// KiB of device memory this process holds per the kernel driver (counted allocations plus context, kernels, queues).
pub fn ownVramKiB() u64 {
    return pidVramKiB(getpid());
}

/// The live runtime queue, for the exit and panic paths (Runtime is returned by value, so only the handles are kept).
var live_list: abi.CommandListHandle = null;
var live_sync: ?@FieldType(abi.Api, "zeCommandListHostSynchronize") = null;
const drain_timeout_ns: u64 = 10_000_000_000;

/// Waits at most 10 s for the queue to go idle, then proceeds: an unbounded wait on a wedged GPU would hang exit.
fn drainBounded() void {
    const f = live_sync orelse return;
    const l = live_list;
    live_sync = null;
    if (l == null) return;
    const rc = f(l, drain_timeout_ns);
    if (rc != abi.success) std.debug.print("device queue not idle after the bounded wait (result 0x{x}), exiting anyway\n", .{rc});
}

/// Panic handler for program roots (`pub const panic = std.debug.FullPanic(rt.panicDrain);`): drains the queue first.
pub fn panicDrain(msg: []const u8, first_trace_addr: ?usize) noreturn {
    drainBounded();
    std.debug.defaultPanic(msg, first_trace_addr);
}

fn atExit() callconv(.c) void {
    drainBounded();
    reportPeak();
}

fn reportPeak() void {
    if (alloc_peak > (2 << 30)) std.debug.print("device allocated {d:.2} GB (peak {d:.2} GB)\n", .{ @as(f64, @floatFromInt(alloc_now)) / 1e9, @as(f64, @floatFromInt(alloc_peak)) / 1e9 });
    if (alloc_trace) std.debug.print("kernel driver count (drm-total-vram0): {d:.2} GB now, {d:.2} GB largest sample; pinned host tables {d:.2} GB\n", .{ @as(f64, @floatFromInt(ownVramKiB() << 10)) / 1e9, @as(f64, @floatFromInt(drm_peak)) / 1e9, @as(f64, @floatFromInt(host_now)) / 1e9 });
}

/// Refuses to start while others hold over ARC_OTHERS_MAX_GB (default 8) of VRAM; ARC_ALLOW_SHARED=1 overrides.
fn vramGuard() Error!void {
    if (std.c.getenv("ARC_ALLOW_SHARED") != null) return;
    var lim: u64 = 8;
    if (std.c.getenv("ARC_OTHERS_MAX_GB")) |v| lim = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 8;
    const held = otherVramKiB();
    if (std.c.getenv("ARC_VRAM_DEBUG") != null) std.debug.print("other processes hold {d} KiB of device memory\n", .{held});
    if (held > lim << 20) {
        std.debug.print("refusing to start: other processes hold {d:.1} GB of device memory (limit {d} GB). Run one model program at a time under `flock /tmp/b70.lock` (ARC_ALLOW_SHARED=1 overrides).\n", .{ @as(f64, @floatFromInt(held)) / 1048576.0, lim });
        return error.VramBusy;
    }
}

pub const Error = xpu.Error || error{VramBusy};

pub const Runtime = struct {
    drv: *const xpu.Driver,
    device: abi.DeviceHandle,
    ctx: abi.ContextHandle,
    list: abi.CommandListHandle,

    /// Borrows the context and stream; both must outlive the Runtime.
    pub fn init(c: *const Context, s: Stream) Error!Runtime {
        try vramGuard();
        live_list = s.list;
        live_sync = c.d.api.zeCommandListHostSynchronize;
        _ = atexit(&atExit); // exit handlers run last-in first-out: after the loader is open, so ours runs before it tears down
        alloc_trace = std.c.getenv("ARC_ALLOC_TRACE") != null;
        if (std.c.getenv("ARC_VRAM_CAP_GB")) |v| {
            cap_bytes = @intFromFloat((std.fmt.parseFloat(f64, std.mem.span(v)) catch 0) * 1e9);
            const oh = if (std.c.getenv("ARC_VRAM_OVERHEAD_GB")) |o| (std.fmt.parseFloat(f64, std.mem.span(o)) catch 0.5) else 0.5;
            cap_overhead = @intFromFloat(oh * 1e9);
        }
        if (std.c.getenv("XPU_PROF") != null) prof_on = true;
        return .{ .drv = c.d, .device = c.entry.device, .ctx = c.handle, .list = s.list };
    }

    /// Drains the queue (bounded wait); the device handles belong to the Context and Stream.
    pub fn deinit(_: *Runtime) void {
        drainBounded();
    }

    pub fn stream(self: *const Runtime, c: *const Context) Stream {
        return .{ .ctx = c, .list = self.list };
    }

    pub fn alloc(self: *Runtime, bytes: usize) Error!Buffer {
        var p: ?*anyopaque = null;
        if (cap_bytes != 0 and alloc_now + bytes + cap_overhead > cap_bytes) {
            std.debug.print("device allocation of {d:.1} MB refused: {d:.2} GB allocated + {d:.2} GB allowance + this exceeds the cap of {d:.2} GB (ARC_VRAM_CAP_GB)\n", .{ @as(f64, @floatFromInt(bytes)) / 1048576.0, @as(f64, @floatFromInt(alloc_now)) / 1e9, @as(f64, @floatFromInt(cap_overhead)) / 1e9, @as(f64, @floatFromInt(cap_bytes)) / 1e9 });
            return error.OutOfMemory;
        }
        try self.drv.check(self.drv.api.zeMemAllocDevice(self.ctx, &.{}, bytes, 64, self.device, &p), "zeMemAllocDevice");
        alloc_now += bytes;
        if (alloc_trace and bytes >= (32 << 20)) std.debug.print("alloc {d:.1} MB (now {d:.2} GB)\n", .{ @as(f64, @floatFromInt(bytes)) / 1048576.0, @as(f64, @floatFromInt(alloc_now)) / 1e9 });
        alloc_peak = @max(alloc_peak, alloc_now);
        if (alloc_trace and alloc_now > drm_sampled_at + (64 << 20)) { // ARC_ALLOC_TRACE: sample the kernel's own count of this process as the peak grows
            drm_sampled_at = alloc_now;
            drm_peak = @max(drm_peak, ownVramKiB() << 10);
        }
        return .{ .rt = self, .ptr = p, .len = bytes };
    }

    /// Pinned host memory the kernels read directly over PCIe; for large tables read a few rows at a time.
    pub fn allocHost(self: *Runtime, bytes: usize) Error!Buffer {
        var p: ?*anyopaque = null;
        try self.drv.check(self.drv.api.zeMemAllocHost(self.ctx, &.{}, bytes, 64, &p), "zeMemAllocHost");
        host_now += bytes;
        return .{ .rt = self, .ptr = p, .len = bytes, .host = true };
    }

    pub fn upload(self: *Runtime, dst: Buffer, src: []const u8) Error!void {
        if (src.len > dst.len) return error.Invalid;
        try self.drv.check(self.drv.api.zeCommandListAppendMemoryCopy(self.list, dst.ptr, src.ptr, src.len, null, 0, null), "upload");
    }

    pub fn download(self: *Runtime, dst: []u8, src: Buffer) Error!void {
        if (dst.len > src.len) return error.Invalid;
        try self.drv.check(self.drv.api.zeCommandListAppendMemoryCopy(self.list, dst.ptr, src.ptr, dst.len, null, 0, null), "download");
    }

    pub fn sync(self: *Runtime) Error!void {
        try self.drv.check(self.drv.api.zeCommandListHostSynchronize(self.list, std.math.maxInt(u64)), "sync");
    }

    /// Builds a SPIR-V module; a refused build logs the driver text.
    pub fn module(self: *Runtime, spv: []const u8) Error!Module {
        return self.moduleWith(spv, null);
    }

    /// As `module` with driver build flags (e.g. "-cl-intel-128-GRF-per-thread").
    pub fn moduleWith(self: *Runtime, spv: []const u8, flags: ?[*:0]const u8) Error!Module {
        var m: abi.ModuleHandle = null;
        var log: abi.BuildLogHandle = null;
        const md: abi.ModuleDesc = .{ .format = abi.module_format_spirv, .input_size = spv.len, .input = spv.ptr, .build_flags = flags };
        const res = self.drv.api.zeModuleCreate(self.ctx, self.device, &md, &m, &log);
        if (res != abi.success) {
            var sz: usize = 0;
            _ = self.drv.api.zeModuleBuildLogGetString(log, &sz, null);
            const buf = std.heap.page_allocator.alloc(u8, sz) catch return error.OutOfMemory;
            defer std.heap.page_allocator.free(buf);
            _ = self.drv.api.zeModuleBuildLogGetString(log, &sz, buf.ptr);
            std.log.err("module build: {s}", .{buf});
            return error.BuildFailed;
        }
        return .{ .rt = self, .handle = m };
    }
};

pub const Buffer = struct {
    rt: *Runtime,
    ptr: ?*anyopaque,
    len: usize,
    host: bool = false, // pinned host memory (allocHost)

    pub fn free(self: *Buffer) void {
        _ = self.rt.drv.api.zeMemFree(self.rt.ctx, self.ptr);
        if (self.host) host_now -|= self.len else alloc_now -|= self.len;
    }
};

pub const Module = struct {
    rt: *Runtime,
    handle: abi.ModuleHandle,

    pub fn deinit(self: *Module) void {
        _ = self.rt.drv.api.zeModuleDestroy(self.handle);
    }

    pub fn kernel(self: *Module, name: [*:0]const u8, local: [3]u32) Error!Kernel {
        var k: abi.KernelHandle = null;
        const api = &self.rt.drv.api;
        try self.rt.drv.check(api.zeKernelCreate(self.handle, &.{ .name = name }, &k), "zeKernelCreate");
        try self.rt.drv.check(api.zeKernelSetGroupSize(k, local[0], local[1], local[2]), "zeKernelSetGroupSize");
        return .{ .rt = self.rt, .handle = k, .name = name };
    }
};

pub const Kernel = struct {
    rt: *Runtime,
    handle: abi.KernelHandle,
    name: [*:0]const u8 = "?",

    pub fn deinit(self: *Kernel) void {
        _ = self.rt.drv.api.zeKernelDestroy(self.handle);
    }

    pub fn setBuffer(self: *Kernel, index: u32, b: Buffer) Error!void {
        var p = b.ptr;
        try self.rt.drv.check(self.rt.drv.api.zeKernelSetArgumentValue(self.handle, index, @sizeOf(?*anyopaque), @ptrCast(&p)), "setBuffer");
    }

    pub fn setU32(self: *Kernel, index: u32, v: u32) Error!void {
        var x = v;
        try self.rt.drv.check(self.rt.drv.api.zeKernelSetArgumentValue(self.handle, index, 4, @ptrCast(&x)), "setU32");
    }

    pub fn setF32(self: *Kernel, index: u32, v: f32) Error!void {
        var x = v;
        try self.rt.drv.check(self.rt.drv.api.zeKernelSetArgumentValue(self.handle, index, 4, @ptrCast(&x)), "setF32");
    }

    /// Queues the kernel over `groups` work-groups.
    pub fn launch(self: *Kernel, groups: [3]u32) Error!void {
        const g: abi.GroupCount = .{ .x = groups[0], .y = groups[1], .z = groups[2] };
        if (prof_on) try self.rt.sync();
        const t0 = if (prof_on) profNow() else 0;
        try self.rt.drv.check(self.rt.drv.api.zeCommandListAppendLaunchKernel(self.rt.list, self.handle, &g, null, 0, null), "launch");
        if (prof_on) {
            try self.rt.sync();
            const dt = profNow() - t0;
            for (&prof_tab) |*e| {
                if (e.name == null) e.name = self.name;
                if (std.mem.eql(u8, std.mem.span(e.name.?), std.mem.span(self.name))) {
                    e.ns += dt;
                    e.calls += 1;
                    break;
                }
            }
        }
    }
};
