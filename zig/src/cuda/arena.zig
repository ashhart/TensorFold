//! A device sub-allocator under DeviceBuffer.alloc: buffers carved from a few large chunks, not one cuMemAlloc each.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, Unsupported, CudaFailed, OutOfMemory };

/// Where chunks come from: `chunk` maps `len` bytes (a multiple of `granularity`), `release` gives a chunk back.
pub const Backend = struct {
    ctx: *anyopaque,
    granularity: usize,
    chunk: *const fn (ctx: *anyopaque, len: usize) ?u64,
    release: *const fn (ctx: *anyopaque, base: u64, len: usize) void,
    name: []const u8,
};

pub const Options = struct {
    small_max: usize = 4 << 20, // requests up to this many bytes share chunks
    small_chunk: usize = 64 << 20, // a shared chunk's bytes, rounded up to the granularity
    alignment: usize = 512, // every range starts at a multiple of this
};

pub const Stats = struct { small: u64 = 0, large: u64 = 0, reused: u64 = 0, fallback: u64 = 0, chunk_bytes: u64 = 0 };

/// Small requests bump 512-byte aligned through shared chunks and are reused by size; large ones get their own chunk.
pub const Arena = struct {
    gpa: std.mem.Allocator,
    be: Backend,
    o: Options,
    mutex: std.atomic.Mutex = .unlocked,
    small: std.ArrayList([2]u64) = .empty, // shared chunks (base, len)
    cursor: u64 = 0, // the current shared chunk's next free byte
    limit: u64 = 0,
    free_small: std.AutoHashMapUnmanaged(u64, std.ArrayList(u64)) = .empty, // freed small ranges by rounded size
    live: std.AutoHashMapUnmanaged(u64, u64) = .empty, // live small ranges: address to rounded size
    large: std.AutoHashMapUnmanaged(u64, u64) = .empty, // large chunks: address to mapped size
    stats: Stats = .{},

    pub fn init(gpa: std.mem.Allocator, be: Backend, o: Options) Arena {
        return .{ .gpa = gpa, .be = be, .o = o };
    }

    pub fn deinit(a: *Arena) void {
        var it = a.large.iterator();
        while (it.next()) |e| a.be.release(a.be.ctx, e.key_ptr.*, e.value_ptr.*);
        for (a.small.items) |c| a.be.release(a.be.ctx, c[0], c[1]);
        var f = a.free_small.valueIterator();
        while (f.next()) |l| l.deinit(a.gpa);
        a.free_small.deinit(a.gpa);
        a.live.deinit(a.gpa);
        a.large.deinit(a.gpa);
        a.small.deinit(a.gpa);
        a.* = undefined;
    }

    fn lock(a: *Arena) void {
        while (!a.mutex.tryLock()) std.Thread.yield() catch {};
    }

    fn roundUp(x: usize, m: usize) usize {
        return std.mem.alignForward(usize, x, m);
    }

    /// An address for `len` bytes, or null (the caller then allocates as before).
    pub fn alloc(a: *Arena, len: usize) ?u64 {
        if (len == 0) return null;
        a.lock();
        defer a.mutex.unlock();
        return a.allocLocked(len) catch null;
    }

    fn allocLocked(a: *Arena, len: usize) !?u64 {
        if (len > a.o.small_max) {
            const n = roundUp(len, a.be.granularity);
            const p = a.be.chunk(a.be.ctx, n) orelse {
                a.stats.fallback += 1;
                return null;
            };
            errdefer a.be.release(a.be.ctx, p, n);
            try a.large.put(a.gpa, p, n);
            a.stats.large += 1;
            a.stats.chunk_bytes += n;
            return p;
        }
        const n: u64 = roundUp(len, a.o.alignment);
        if (a.free_small.getPtr(n)) |l| if (l.pop()) |p| {
            try a.live.put(a.gpa, p, n);
            a.stats.reused += 1;
            return p;
        };
        if (a.limit - a.cursor < n) {
            const cl = roundUp(@max(a.o.small_chunk, n), a.be.granularity);
            const base = a.be.chunk(a.be.ctx, cl) orelse {
                a.stats.fallback += 1;
                return null;
            };
            errdefer a.be.release(a.be.ctx, base, cl);
            try a.small.append(a.gpa, .{ base, cl });
            a.cursor = base;
            a.limit = base + cl;
            a.stats.chunk_bytes += cl;
        }
        const p = a.cursor;
        try a.live.put(a.gpa, p, n);
        a.cursor += n;
        a.stats.small += 1;
        return p;
    }

    /// True when `p` was the arena's (and is now free); false: not ours, the caller frees it as before.
    pub fn free(a: *Arena, p: u64) bool {
        a.lock();
        defer a.mutex.unlock();
        if (a.large.fetchRemove(p)) |e| {
            a.be.release(a.be.ctx, p, e.value);
            return true;
        }
        const e = a.live.fetchRemove(p) orelse return false;
        const g = a.free_small.getOrPut(a.gpa, e.value) catch return true; // leaked range, still ours
        if (!g.found_existing) g.value_ptr.* = .empty;
        g.value_ptr.append(a.gpa, p) catch {};
        return true;
    }

    // -- memory.zig's hook ----------------------------------------------------------------------------------------

    fn hookAlloc(ctx: *anyopaque, len: usize) ?u64 {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        return a.alloc(len);
    }

    fn hookFree(ctx: *anyopaque, p: u64) bool {
        const a: *Arena = @ptrCast(@alignCast(ctx));
        return a.free(p);
    }

    /// Every later DeviceBuffer.alloc comes from the arena, which must outlive its buffers; only addresses change.
    pub fn install(a: *Arena) void {
        memory.setHook(.{ .ctx = a, .alloc = hookAlloc, .free = hookFree });
    }
};

// -- backends ---------------------------------------------------------------------------------------------------------

/// One cuMemAlloc a chunk (the driver's own 2 MiB-aligned allocations).
pub const Plain = struct {
    d: *const Driver,

    pub fn backend(p: *Plain) Backend {
        return .{ .ctx = p, .granularity = 2 << 20, .chunk = chunk, .release = release, .name = "plain" };
    }

    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const p: *Plain = @ptrCast(@alignCast(ctx));
        var q: abi.DevicePtr = 0;
        if (p.d.api.cuMemAlloc_v2(&q, len) != abi.success) return null;
        return q;
    }

    fn release(ctx: *anyopaque, base: u64, _: usize) void {
        const p: *Plain = @ptrCast(@alignCast(ctx));
        _ = p.d.api.cuMemFree_v2(base);
    }
};

/// CUDA's virtual memory management: one reserved virtual range, chunks of the recommended granularity mapped in order.
pub const Vmm = struct {
    lib: std.DynLib,
    device: abi.Device,
    prop: Prop,
    granularity: usize,
    va: u64 = 0,
    va_len: usize = 0,
    next: u64 = 0,
    api: Api,

    const Api = struct {
        reserve: *const fn (*u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        addressFree: *const fn (u64, usize) callconv(.c) abi.Result,
        create: *const fn (*u64, usize, *const Prop, c_ulonglong) callconv(.c) abi.Result,
        release: *const fn (u64) callconv(.c) abi.Result,
        map: *const fn (u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        unmap: *const fn (u64, usize) callconv(.c) abi.Result,
        setAccess: *const fn (u64, usize, *const AccessDesc, usize) callconv(.c) abi.Result,
        granularity: *const fn (*usize, *const Prop, c_int) callconv(.c) abi.Result,
    };

    pub const Prop = extern struct { // CUmemAllocationProp
        type: c_int = 1, // CU_MEM_ALLOCATION_TYPE_PINNED
        handle_types: c_int = 0,
        loc_type: c_int = 1, // CU_MEM_LOCATION_TYPE_DEVICE
        loc_id: c_int = 0,
        win32: ?*anyopaque = null,
        compression: u8 = 0,
        rdma: u8 = 0,
        usage: u16 = 0,
        reserved: [4]u8 = .{ 0, 0, 0, 0 },
    };
    const AccessDesc = extern struct { loc_type: c_int = 1, loc_id: c_int, flags: c_int = 3 }; // CUmemAccessDesc, read-write

    comptime {
        std.debug.assert(@sizeOf(Prop) == 32 and @offsetOf(Prop, "compression") == 24 and @sizeOf(AccessDesc) == 12);
    }

    /// `va_bytes` of virtual addresses reserved (cheap: no memory behind them until a chunk maps).
    pub fn open(d: *const Driver, device: abi.Device, va_bytes: usize) Error!Vmm {
        var vmm_ok: c_int = 0;
        if (d.api.cuDeviceGetAttribute(&vmm_ok, .virtual_memory_management_supported, device) != abi.success or vmm_ok == 0) return error.Unsupported; // VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const names = .{ .{ "reserve", "cuMemAddressReserve" }, .{ "addressFree", "cuMemAddressFree" }, .{ "create", "cuMemCreate" }, .{ "release", "cuMemRelease" }, .{ "map", "cuMemMap" }, .{ "unmap", "cuMemUnmap" }, .{ "setAccess", "cuMemSetAccess" }, .{ "granularity", "cuMemGetAllocationGranularity" } };
        inline for (names) |n| @field(api, n[0]) = lib.lookup(@FieldType(Api, n[0]), n[1]) orelse return error.MissingSymbol;
        var prop: Prop = .{ .loc_id = device };
        var rdma: c_int = 0; // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED: keep the memory registrable as cuMemAlloc's is
        if (d.api.cuDeviceGetAttribute(&rdma, .gpu_direct_rdma_with_cuda_vmm_supported, device) == abi.success and rdma != 0) prop.rdma = 1;
        var g: usize = 0;
        if (api.granularity(&g, &prop, 1) != abi.success or g == 0) return error.Unsupported; // RECOMMENDED
        g = @max(g, 2 << 20);
        const len = std.mem.alignForward(usize, va_bytes, g);
        var va: u64 = 0;
        if (api.reserve(&va, len, g, 0, 0) != abi.success) return error.CudaFailed;
        return .{ .lib = lib, .device = device, .prop = prop, .granularity = g, .va = va, .va_len = len, .next = va, .api = api };
    }

    pub fn close(v: *Vmm) void {
        _ = v.api.addressFree(v.va, v.va_len);
        v.lib.close();
    }

    pub fn backend(v: *Vmm) Backend {
        return .{ .ctx = v, .granularity = v.granularity, .chunk = chunk, .release = release, .name = "vmm" };
    }

    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const v: *Vmm = @ptrCast(@alignCast(ctx));
        if (v.va + v.va_len - v.next < len) return null;
        var h: u64 = 0;
        if (v.api.create(&h, len, &v.prop, 0) != abi.success) return null;
        defer _ = v.api.release(h); // the mapping holds the memory until it is unmapped
        const at = v.next;
        if (v.api.map(at, len, 0, h, 0) != abi.success) return null;
        const desc: AccessDesc = .{ .loc_id = v.device };
        if (v.api.setAccess(at, len, &desc, 1) != abi.success) {
            _ = v.api.unmap(at, len);
            return null;
        }
        v.next += len;
        return at;
    }

    fn release(ctx: *anyopaque, base: u64, len: usize) void {
        const v: *Vmm = @ptrCast(@alignCast(ctx));
        _ = v.api.unmap(base, len); // the virtual range is not reused: the reservation is sized for the process
    }
};

// -- tests (a fake backend: addresses only) ---------------------------------------------------------------------------

const Fake = struct {
    next: u64 = 1 << 40,
    chunks: u32 = 0,
    released: u32 = 0,
    fail: bool = false,

    fn backend(f: *Fake, g: usize) Backend {
        return .{ .ctx = f, .granularity = g, .chunk = chunk, .release = release, .name = "fake" };
    }
    fn chunk(ctx: *anyopaque, len: usize) ?u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        if (f.fail) return null;
        const p = f.next;
        f.next += len;
        f.chunks += 1;
        return p;
    }
    fn release(ctx: *anyopaque, _: u64, _: usize) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }
};

test "arena: small requests pack 512-aligned into shared chunks, are reused by size, large ones get their own" {
    const t = std.testing;
    var f: Fake = .{};
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{ .small_max = 1 << 20, .small_chunk = 4 << 20 });
    defer a.deinit();
    const p1 = a.alloc(20480).?; // a norm weight
    const p2 = a.alloc(983040).?; // an mHC fn16
    const p3 = a.alloc(100).?;
    try t.expectEqual(p1 + 20480, p2);
    try t.expectEqual(p2 + 983040, p3);
    for ([_]u64{ p1, p2, p3 }) |p| try t.expectEqual(@as(u64, 0), p % 512);
    try t.expectEqual(@as(u32, 1), f.chunks);
    // a freed range comes back for the same rounded size
    try t.expect(a.free(p3));
    try t.expectEqual(p3, a.alloc(17).?);
    try t.expectEqual(@as(u64, 1), a.stats.reused);
    // three more 1 MiB ranges fit the 4 MiB chunk, the fourth opens a new shared chunk
    for (0..3) |_| _ = a.alloc(1 << 20).?;
    try t.expectEqual(@as(u32, 1), f.chunks);
    _ = a.alloc(1 << 20).?;
    try t.expectEqual(@as(u32, 2), f.chunks);
    // large: its own chunk, rounded to the granularity, released when freed
    const big = a.alloc(5 << 20).?;
    try t.expectEqual(@as(u32, 3), f.chunks);
    try t.expectEqual(@as(u64, 6 << 20), a.large.get(big).?);
    try t.expect(a.free(big));
    try t.expectEqual(@as(u32, 1), f.released);
    // not ours
    try t.expect(!a.free(12345));
    // a backend that cannot map: null, the caller allocates as before
    f.fail = true;
    try t.expectEqual(@as(?u64, null), a.alloc(64 << 20));
    try t.expectEqual(@as(u64, 1), a.stats.fallback);
}

test "arena: under DeviceBuffer.alloc through memory.setHook" {
    const t = std.testing;
    var f: Fake = .{};
    var a = Arena.init(t.allocator, f.backend(2 << 20), .{});
    defer a.deinit();
    a.install();
    defer memory.setHook(null);
    const h = memory.currentHook().?;
    const p = h.alloc(h.ctx, 4096).?;
    try t.expect(h.free(h.ctx, p));
    try t.expect(!h.free(h.ctx, 7));
}
