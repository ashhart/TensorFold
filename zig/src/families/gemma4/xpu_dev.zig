//! Device plumbing of the Gemma 4 E4B XPU runner: safetensors shards (mmap), Level Zero device, launches, GPU time.
const std = @import("std");
const ze = @import("ze");
const safetensors = @import("safetensors");

pub const alloc = std.heap.c_allocator;

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
extern "c" fn creat(path: [*:0]const u8, mode: c_uint) c_int;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;

pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

pub const Tensor = safetensors.Tensor;

/// A whole file mapped read-only (the caller never unmaps it: the process is short-lived).
pub fn mapFile(path: []const u8) ![]const u8 {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = open(z.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    const end = lseek(fd, 0, 2);
    if (end <= 0) return error.EmptyFile;
    const size: usize = @intCast(end);
    const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed;
    if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed;
    return @as([*]const u8, @ptrCast(base))[0..size];
}

pub fn writeFile(path: []const u8, parts: []const []const u8) !void {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = creat(z.ptr, 0o644);
    if (fd < 0) return error.CannotCreate;
    defer _ = close(fd);
    for (parts) |p| {
        var done: usize = 0;
        while (done < p.len) {
            const n = write(fd, p.ptr + done, p.len - done);
            if (n <= 0) return error.ShortWrite;
            done += @intCast(n);
        }
    }
}

pub const Checkpoint = struct {
    tensors: std.ArrayList(Tensor) = .empty,
    maps: std.ArrayList([]const u8) = .empty,
    headers: std.ArrayList(std.json.Parsed(std.json.Value)) = .empty,

    pub fn addShard(c: *Checkpoint, path: []const u8) !void {
        const bytes = try mapFile(path);
        try c.maps.ensureUnusedCapacity(alloc, 1);
        try c.headers.ensureUnusedCapacity(alloc, 1);
        c.headers.appendAssumeCapacity(try safetensors.parse(alloc, bytes, &c.tensors));
        c.maps.appendAssumeCapacity(bytes);
    }

    /// Every shard the checkpoint's index names.
    pub fn open(c: *Checkpoint, dir: []const u8) !void {
        const index_path = try std.fmt.allocPrint(alloc, "{s}/model.safetensors.index.json", .{dir});
        defer alloc.free(index_path);
        const parsed = std.json.parseFromSlice(std.json.Value, alloc, try mapFile(index_path), .{}) catch return error.BadIndex;
        defer parsed.deinit();
        const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
        if (map == null or map.? != .object) return error.BadIndex;
        var seen: std.ArrayList([]const u8) = .empty;
        defer seen.deinit(alloc);
        var it = map.?.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadIndex;
            const f = e.value_ptr.string;
            for (seen.items) |s| {
                if (std.mem.eql(u8, s, f)) break;
            } else {
                try seen.append(alloc, f);
                const shard = try std.fmt.allocPrint(alloc, "{s}/{s}", .{ dir, f });
                defer alloc.free(shard);
                try c.addShard(shard);
            }
        }
    }

    pub fn deinit(c: *Checkpoint) void {
        for (c.headers.items) |h| h.deinit();
        for (c.maps.items) |m| _ = munmap(@ptrCast(@constCast(m.ptr)), m.len);
        c.tensors.deinit(alloc);
        c.maps.deinit(alloc);
        c.headers.deinit(alloc);
        c.* = .{};
    }

    pub fn get(c: *const Checkpoint, name: []const u8) !Tensor {
        for (c.tensors.items) |t| if (std.mem.eql(u8, t.name, name)) return t;
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    }
};

pub fn off(p: *anyopaque, bytes: usize) *anyopaque {
    return @ptrFromInt(@intFromPtr(p) + bytes);
}

pub const Dev = struct {
    g: ze.Gpu,
    list: ze.H,
    staging: []u8 = &.{},
    allocated: usize = 0,
    names: std.ArrayList(struct { k: ze.H, name: []const u8 }) = .empty,
    // XPU_PROFILE: GPU time per kernel
    prof_on: bool = false,
    events: [4096]ze.H = @splat(null),
    used: usize = 0,
    ev_kernel: [4096]ze.H = @splat(null),
    total: std.AutoHashMapUnmanaged(usize, struct { s: f64, n: u64 }) = .empty,

    pub fn open(device_id: u32) !*Dev {
        const d = try alloc.create(Dev);
        errdefer alloc.destroy(d);
        d.* = .{ .g = try ze.Gpu.open(device_id), .list = undefined };
        errdefer d.g.close();
        d.list = try d.g.immediate(0, 0);
        return d;
    }

    pub fn deinit(d: *Dev) void {
        d.settle();
        d.g.close();
        alloc.destroy(d);
    }

    pub fn settle(d: *Dev) void {
        const rc = d.g.api.zeCommandListHostSynchronize(d.list, std.math.maxInt(u64));
        if (rc != 0) {
            std.log.err("XPU: a command list did not drain (Level Zero 0x{x}); the process ends", .{rc});
            std.process.exit(1);
        }
    }

    pub fn kernel(d: *Dev, m: ze.H, name: [:0]const u8, local: u32) !ze.H {
        const k = try d.g.kernel(m, name.ptr);
        try ze.check(d.g.api.zeKernelSetGroupSize(k, local, 1, 1));
        try d.names.append(alloc, .{ .k = k, .name = name });
        return k;
    }

    pub fn alloc_(d: *Dev, bytes: usize) !*anyopaque {
        d.allocated += @max(bytes, 64);
        return d.g.deviceAlloc(@max(bytes, 64));
    }

    pub fn upload(d: *Dev, bytes: []const u8) !*anyopaque {
        const p = try d.alloc_(bytes.len);
        try d.copyIn(p, bytes);
        return p;
    }

    /// Host bytes (mmap included) to the device through the USM staging buffer.
    pub fn copyIn(d: *Dev, dst: *anyopaque, bytes: []const u8) !void {
        if (d.staging.len == 0) d.staging = @as([*]u8, @ptrCast(try d.g.hostAlloc(256 << 20)))[0 .. 256 << 20];
        var done: usize = 0;
        while (done < bytes.len) {
            const n = @min(d.staging.len, bytes.len - done);
            @memcpy(d.staging[0..n], bytes[done .. done + n]);
            try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, off(dst, done), d.staging.ptr, n, null, 0, null));
            try d.sync();
            done += n;
        }
    }

    pub fn fill(d: *Dev, p: *anyopaque, pattern: anytype, bytes: usize) !void {
        const T = @TypeOf(pattern);
        const host = try alloc.alloc(T, bytes / @sizeOf(T));
        defer alloc.free(host);
        @memset(host, pattern);
        try d.copyIn(p, std.mem.sliceAsBytes(host));
    }

    /// Device bytes to the host, waiting for the queue.
    pub fn read(d: *Dev, dst: []u8, src: *anyopaque) !void {
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst.ptr, src, dst.len, null, 0, null));
        try d.sync();
    }

    pub fn copy(d: *Dev, dst: *anyopaque, src: *anyopaque, n: usize) !void {
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, dst, src, n, null, 0, null));
    }

    pub fn launch(d: *Dev, k: ze.H, groups: ze.GroupCount, args: anytype) !void {
        inline for (args, 0..) |a, i| try d.g.arg(k, i, a);
        var ev: ze.H = null;
        if (d.prof_on) {
            if (d.used == d.events.len) try d.sync();
            ev = d.events[d.used];
            d.ev_kernel[d.used] = k;
            d.used += 1;
        }
        try ze.check(d.g.api.zeCommandListAppendLaunchKernel(d.list, k, &groups, ev, 0, null));
    }

    pub fn sync(d: *Dev) !void {
        d.settle();
        if (d.prof_on) {
            for (d.events[0..d.used], d.ev_kernel[0..d.used]) |e, k| {
                const t = try d.g.seconds(e);
                try ze.check(d.g.api.zeEventHostReset(e));
                const slot = try d.total.getOrPut(alloc, @intFromPtr(k));
                if (!slot.found_existing) slot.value_ptr.* = .{ .s = 0, .n = 0 };
                slot.value_ptr.s += t;
                slot.value_ptr.n += 1;
            }
            d.used = 0;
        }
    }

    pub fn profOn(d: *Dev) !void {
        d.events = try d.g.events(4096);
        d.prof_on = true;
    }

    pub fn profReport(d: *Dev, label: []const u8) void {
        if (!d.prof_on) return;
        var sum: f64 = 0;
        var it = d.total.valueIterator();
        while (it.next()) |v| sum += v.s;
        std.debug.print("GPU time, {s}: {d:.1} ms\n", .{ label, sum * 1e3 });
        for (d.names.items) |n| if (d.total.get(@intFromPtr(n.k))) |v| if (v.s > 0)
            std.debug.print("  {s:<18} {d:8.2} ms {d:5.1} %  {d:6} launches {d:7.1} us each\n", .{ n.name, v.s * 1e3, v.s / sum * 100, v.n, v.s / @as(f64, @floatFromInt(v.n)) * 1e6 });
        d.total.clearRetainingCapacity();
    }
};
