//! Level Zero opened with dlopen and no C header: only the entries the XPU backend uses; a missing symbol fails closed.
const std = @import("std");

pub const H = ?*anyopaque;
pub const Error = error{ LoaderUnavailable, MissingSymbol, ZeFailed, NoDevice, OutOfMemory };

pub const Api = struct {
    zeInit: *const fn (u32) callconv(.c) u32,
    zeDriverGet: *const fn (*u32, ?[*]H) callconv(.c) u32,
    zeDeviceGet: *const fn (H, *u32, ?[*]H) callconv(.c) u32,
    zeDeviceGetProperties: *const fn (H, *DeviceProperties) callconv(.c) u32,
    zeContextCreate: *const fn (H, *const ContextDesc, *H) callconv(.c) u32,
    zeCommandQueueCreate: *const fn (H, H, *const QueueDesc, *H) callconv(.c) u32,
    zeCommandQueueExecuteCommandLists: *const fn (H, u32, [*]H, H) callconv(.c) u32,
    zeCommandQueueSynchronize: *const fn (H, u64) callconv(.c) u32,
    zeCommandListCreate: *const fn (H, H, *const ListDesc, *H) callconv(.c) u32,
    zeCommandListCreateImmediate: *const fn (H, H, *const QueueDesc, *H) callconv(.c) u32,
    zeCommandListClose: *const fn (H) callconv(.c) u32,
    zeCommandListReset: *const fn (H) callconv(.c) u32,
    zeCommandListHostSynchronize: *const fn (H, u64) callconv(.c) u32,
    zeCommandListAppendLaunchKernel: *const fn (H, H, *const GroupCount, H, u32, ?[*]H) callconv(.c) u32,
    zeCommandListAppendMemoryCopy: *const fn (H, ?*anyopaque, ?*const anyopaque, usize, H, u32, ?[*]H) callconv(.c) u32,
    zeCommandListAppendBarrier: *const fn (H, H, u32, ?[*]H) callconv(.c) u32,
    zeMemAllocDevice: *const fn (H, *const DeviceAllocDesc, usize, usize, H, *?*anyopaque) callconv(.c) u32,
    zeMemAllocHost: *const fn (H, *const HostAllocDesc, usize, usize, *?*anyopaque) callconv(.c) u32,
    zeMemFree: *const fn (H, ?*anyopaque) callconv(.c) u32,
    zeModuleCreate: *const fn (H, H, *const ModuleDesc, *H, ?*H) callconv(.c) u32,
    zeKernelCreate: *const fn (H, *const KernelDesc, *H) callconv(.c) u32,
    zeKernelSetGroupSize: *const fn (H, u32, u32, u32) callconv(.c) u32,
    zeKernelSetArgumentValue: *const fn (H, u32, usize, ?*const anyopaque) callconv(.c) u32,
    zeKernelSetIndirectAccess: *const fn (H, u32) callconv(.c) u32,
    zeKernelGetProperties: *const fn (H, *KernelProperties) callconv(.c) u32,
    zeEventPoolCreate: *const fn (H, *const EventPoolDesc, u32, ?[*]H, *H) callconv(.c) u32,
    zeEventCreate: *const fn (H, *const EventDesc, *H) callconv(.c) u32,
    zeEventHostSynchronize: *const fn (H, u64) callconv(.c) u32,
    zeEventHostReset: *const fn (H) callconv(.c) u32,
    zeEventQueryKernelTimestamp: *const fn (H, *KernelTimestampResult) callconv(.c) u32,
    zeEventDestroy: *const fn (H) callconv(.c) u32,
    zeEventPoolDestroy: *const fn (H) callconv(.c) u32,
    zeKernelDestroy: *const fn (H) callconv(.c) u32,
    zeModuleDestroy: *const fn (H) callconv(.c) u32,
    zeCommandListDestroy: *const fn (H) callconv(.c) u32,
    zeContextDestroy: *const fn (H) callconv(.c) u32,
};

/// Sysman, for the free memory of the device's own memory modules (Gpu.freeMemory).
pub const SysApi = struct {
    zesInit: *const fn (u32) callconv(.c) u32,
    zesDriverGet: *const fn (*u32, ?[*]H) callconv(.c) u32,
    zesDeviceGet: *const fn (H, *u32, ?[*]H) callconv(.c) u32,
    zesDeviceGetProperties: *const fn (H, *SysDeviceProperties) callconv(.c) u32,
    zesDeviceEnumMemoryModules: *const fn (H, *u32, ?[*]H) callconv(.c) u32,
    zesMemoryGetProperties: *const fn (H, *MemProperties) callconv(.c) u32,
    zesMemoryGetState: *const fn (H, *MemState) callconv(.c) u32,
};
/// zes_device_properties_t: the core properties, then strings this backend does not read.
pub const SysDeviceProperties = extern struct {
    stype: u32 = 0x1, // ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
    pNext: ?*anyopaque = null,
    core: DeviceProperties = .{ .stype = 0x3 }, // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
    num_subdevices: u32 = 0,
    strings: [6][64]u8 = @splat(@splat(0)), // serial, board, brand, model, vendor, driver version
};
pub const MemProperties = extern struct {
    stype: u32 = 0xb, // ZES_STRUCTURE_TYPE_MEM_PROPERTIES
    pNext: ?*anyopaque = null,
    type: u32 = 0,
    on_subdevice: u8 = 0,
    subdevice_id: u32 = 0,
    location: u32 = 0, // 1: on-board device memory
    physical_size: u64 = 0,
    bus_width: i32 = 0,
    channels: i32 = 0,
};
pub const MemState = extern struct {
    stype: u32 = 0x1e, // ZES_STRUCTURE_TYPE_MEM_STATE
    pNext: ?*const anyopaque = null,
    health: u32 = 0,
    free: u64 = 0,
    size: u64 = 0,
};

/// ZE_extension_record_replay_graph: capture an immediate list into a graph, then replay it.
pub const GraphApi = struct {
    zeCommandListBeginGraphCaptureExt: *const fn (H, ?*const anyopaque) callconv(.c) u32,
    zeCommandListEndGraphCaptureExt: *const fn (H, ?*const anyopaque, *H) callconv(.c) u32,
    zeGraphInstantiateExt: *const fn (H, ?*const anyopaque, *H) callconv(.c) u32,
    zeCommandListAppendGraphExt: *const fn (H, H, ?*const anyopaque, H, u32, ?[*]H) callconv(.c) u32,
};

pub const DeviceProperties = extern struct {
    stype: u32 = 0x00020006, // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2: timerResolution in cycles per second
    pNext: ?*anyopaque = null,
    type: u32 = 0,
    vendorId: u32 = 0,
    deviceId: u32 = 0,
    flags: u32 = 0,
    subdeviceId: u32 = 0,
    coreClockRate: u32 = 0,
    maxMemAllocSize: u64 = 0,
    maxHardwareContexts: u32 = 0,
    maxCommandQueuePriority: u32 = 0,
    numThreadsPerEU: u32 = 0,
    physicalEUSimdWidth: u32 = 0,
    numEUsPerSubslice: u32 = 0,
    numSubslicesPerSlice: u32 = 0,
    numSlices: u32 = 0,
    timerResolution: u64 = 0,
    timestampValidBits: u32 = 0,
    kernelTimestampValidBits: u32 = 0,
    uuid: [16]u8 = @splat(0),
    name: [256]u8 = @splat(0),
};
pub const ContextDesc = extern struct { stype: u32 = 0xd, pNext: ?*const anyopaque = null, flags: u32 = 0 };
pub const QueueDesc = extern struct {
    stype: u32 = 0xe,
    pNext: ?*const anyopaque = null,
    ordinal: u32 = 0,
    index: u32 = 0,
    flags: u32 = 0,
    mode: u32 = 0, // 1 synchronous, 2 asynchronous
    priority: u32 = 0, // 0 normal, 1 low, 2 high
};
pub const queue_in_order: u32 = 1 << 1;
pub const ListDesc = extern struct { stype: u32 = 0xf, pNext: ?*const anyopaque = null, ordinal: u32 = 0, flags: u32 = 0 };
pub const list_in_order: u32 = 1 << 3;
pub const DeviceAllocDesc = extern struct { stype: u32 = 0x15, pNext: ?*const anyopaque = null, flags: u32 = 0, ordinal: u32 = 0 };
pub const HostAllocDesc = extern struct { stype: u32 = 0x16, pNext: ?*const anyopaque = null, flags: u32 = 0 };
pub const ModuleDesc = extern struct {
    stype: u32 = 0x1b,
    pNext: ?*const anyopaque = null,
    format: u32 = 1, // ZE_MODULE_FORMAT_NATIVE
    inputSize: usize,
    pInputModule: [*]const u8,
    pBuildFlags: ?[*:0]const u8 = null,
    pConstants: ?*const anyopaque = null,
};
pub const KernelDesc = extern struct { stype: u32 = 0x1d, pNext: ?*const anyopaque = null, flags: u32 = 0, pKernelName: [*:0]const u8 };
pub const KernelProperties = extern struct {
    stype: u32 = 0x1e,
    pNext: ?*anyopaque = null,
    numKernelArgs: u32 = 0,
    requiredGroupSize: [3]u32 = @splat(0),
    requiredNumSubGroups: u32 = 0,
    requiredSubgroupSize: u32 = 0,
    maxSubgroupSize: u32 = 0,
    maxNumSubgroups: u32 = 0,
    localMemSize: u32 = 0,
    privateMemSize: u32 = 0,
    spillMemSize: u32 = 0,
    uuid: [32]u8 = @splat(0),
};
pub const GroupCount = extern struct { x: u32, y: u32 = 1, z: u32 = 1 };
pub const EventPoolDesc = extern struct { stype: u32 = 0x10, pNext: ?*const anyopaque = null, flags: u32 = 0, count: u32 };
pub const pool_host_visible: u32 = 1 << 0;
pub const pool_kernel_timestamp: u32 = 1 << 2;
pub const EventDesc = extern struct { stype: u32 = 0x11, pNext: ?*const anyopaque = null, index: u32, signal: u32 = 1 << 2, wait: u32 = 1 << 2 };
pub const KernelTimestampResult = extern struct { global_start: u64 = 0, global_end: u64 = 0, context_start: u64 = 0, context_end: u64 = 0 };

pub fn check(r: u32) Error!void {
    if (r != 0) {
        std.debug.print("Level Zero : 0x{x}\n", .{r});
        return error.ZeFailed;
    }
}

pub fn resolve(comptime Table: type, lib: *std.DynLib) Error!Table {
    var api: Table = undefined;
    const info = @typeInfo(Table).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        @field(api, name) = lib.lookup(T, name) orelse {
            std.debug.print("missing Level Zero symbol: {s}\n", .{name});
            return error.MissingSymbol;
        };
    }
    return api;
}

/// The loader, a context and the first device whose PCI id is `device_id`.
pub const Gpu = struct {
    lib: std.DynLib,
    api: Api,
    driver: H,
    device: H,
    context: H,
    props: DeviceProperties,
    owned: Owned = .{}, // what close() gives back

    /// Every allocation, kernel, module, list and event made through this Gpu, destroyed by close().
    const Owned = struct {
        mems: std.AutoHashMapUnmanaged(usize, void) = .empty,
        kernels: std.ArrayList(H) = .empty,
        modules: std.ArrayList(H) = .empty,
        lists: std.ArrayList(H) = .empty,
        events: std.ArrayList(H) = .empty,
        pools: std.ArrayList(H) = .empty,
    };
    const own_alloc = std.heap.c_allocator;

    fn keep(list: *std.ArrayList(H), h: H, destroy: *const fn (H) callconv(.c) u32) Error!H {
        list.append(own_alloc, h) catch {
            _ = destroy(h);
            return error.OutOfMemory;
        };
        return h;
    }

    fn keepMem(g: *Gpu, p: *anyopaque) Error!*anyopaque {
        g.owned.mems.put(own_alloc, @intFromPtr(p), {}) catch {
            _ = g.api.zeMemFree(g.context, p);
            return error.OutOfMemory;
        };
        return p;
    }

    /// Destroys what this Gpu made, then the context; the caller has waited for its queues.
    pub fn close(g: *Gpu) void {
        const o = &g.owned;
        for (o.events.items) |e| _ = g.api.zeEventDestroy(e);
        for (o.pools.items) |p| _ = g.api.zeEventPoolDestroy(p);
        for (o.kernels.items) |k| _ = g.api.zeKernelDestroy(k);
        for (o.modules.items) |m| _ = g.api.zeModuleDestroy(m);
        for (o.lists.items) |l| _ = g.api.zeCommandListDestroy(l);
        var it = o.mems.keyIterator();
        while (it.next()) |p| _ = g.api.zeMemFree(g.context, @ptrFromInt(p.*));
        o.mems.deinit(own_alloc);
        inline for (.{ &o.kernels, &o.modules, &o.lists, &o.events, &o.pools }) |l| l.deinit(own_alloc);
        _ = g.api.zeContextDestroy(g.context);
        g.lib.close();
    }

    pub fn open(device_id: u32) Error!Gpu {
        var lib = std.DynLib.open("libze_loader.so.1") catch return error.LoaderUnavailable;
        errdefer lib.close();
        const api = try resolve(Api, &lib);
        try check(api.zeInit(1)); // ZE_INIT_FLAG_GPU_ONLY
        var nd: u32 = 8;
        var drivers: [8]H = @splat(null);
        try check(api.zeDriverGet(&nd, &drivers));
        for (drivers[0..nd]) |drv| {
            var nv: u32 = 8;
            var devs: [8]H = @splat(null);
            try check(api.zeDeviceGet(drv, &nv, &devs));
            for (devs[0..nv]) |dev| {
                var p: DeviceProperties = .{};
                try check(api.zeDeviceGetProperties(dev, &p));
                if (p.deviceId != device_id) continue;
                var ctx: H = null;
                try check(api.zeContextCreate(drv, &.{}, &ctx));
                return .{ .lib = lib, .api = api, .driver = drv, .device = dev, .context = ctx, .props = p };
            }
        }
        return error.NoDevice;
    }

    /// On-board memory by Sysman (matched by UUID): free and total; `free` misses allocations not yet written.
    pub fn memory(g: *Gpu) Error!struct { free: u64, size: u64 } {
        const sys = try resolve(SysApi, &g.lib);
        try check(sys.zesInit(0));
        var nd: u32 = 8;
        var drivers: [8]H = @splat(null);
        try check(sys.zesDriverGet(&nd, &drivers));
        for (drivers[0..nd]) |drv| {
            var nv: u32 = 8;
            var devs: [8]H = @splat(null);
            try check(sys.zesDeviceGet(drv, &nv, &devs));
            for (devs[0..nv]) |dev| {
                var p: SysDeviceProperties = .{};
                try check(sys.zesDeviceGetProperties(dev, &p));
                if (!std.mem.eql(u8, &p.core.uuid, &g.props.uuid)) continue;
                var nm: u32 = 16;
                var mods: [16]H = @splat(null);
                try check(sys.zesDeviceEnumMemoryModules(dev, &nm, &mods));
                var left: u64 = 0;
                var all: u64 = 0;
                for (mods[0..nm]) |m| {
                    var mp: MemProperties = .{};
                    try check(sys.zesMemoryGetProperties(m, &mp));
                    if (mp.location != 1) continue;
                    var st: MemState = .{};
                    try check(sys.zesMemoryGetState(m, &st));
                    left += st.free;
                    all += if (mp.physical_size > 0) mp.physical_size else st.size;
                }
                return .{ .free = left, .size = all };
            }
        }
        return error.NoDevice;
    }

    pub fn deviceAlloc(g: *Gpu, size: usize) Error!*anyopaque {
        var p: ?*anyopaque = null;
        try check(g.api.zeMemAllocDevice(g.context, &.{}, size, 4096, g.device, &p));
        return g.keepMem(p.?);
    }

    pub fn hostAlloc(g: *Gpu, size: usize) Error!*anyopaque {
        var p: ?*anyopaque = null;
        try check(g.api.zeMemAllocHost(g.context, &.{}, size, 4096, &p));
        return g.keepMem(p.?);
    }

    pub fn free(g: *Gpu, p: *anyopaque) void {
        _ = g.owned.mems.remove(@intFromPtr(p));
        _ = g.api.zeMemFree(g.context, p);
    }

    /// In-order immediate list on queue group `ordinal` (0 compute, 1 copy on the B70).
    pub fn immediate(g: *Gpu, ordinal: u32, priority: u32) Error!H {
        var l: H = null;
        try check(g.api.zeCommandListCreateImmediate(g.context, g.device, &.{ .ordinal = ordinal, .flags = queue_in_order, .mode = 2, .priority = priority }, &l));
        return keep(&g.owned.lists, l, g.api.zeCommandListDestroy);
    }

    /// Immediate list without the in-order flag: commands with no barrier or event between them may overlap.
    pub fn immediateUnordered(g: *Gpu, ordinal: u32) Error!H {
        var l: H = null;
        try check(g.api.zeCommandListCreateImmediate(g.context, g.device, &.{ .ordinal = ordinal, .flags = 0, .mode = 2, .priority = 0 }, &l));
        return keep(&g.owned.lists, l, g.api.zeCommandListDestroy);
    }

    pub fn module(g: *Gpu, bin: []const u8) Error!H {
        var m: H = null;
        try check(g.api.zeModuleCreate(g.context, g.device, &.{ .inputSize = bin.len, .pInputModule = bin.ptr }, &m, null));
        return keep(&g.owned.modules, m, g.api.zeModuleDestroy);
    }

    pub fn kernel(g: *Gpu, m: H, name: [*:0]const u8) Error!H {
        var k: H = null;
        try check(g.api.zeKernelCreate(m, &.{ .pKernelName = name }, &k));
        _ = try keep(&g.owned.kernels, k, g.api.zeKernelDestroy);
        var kp: KernelProperties = .{};
        try check(g.api.zeKernelGetProperties(k, &kp));
        if (kp.spillMemSize != 0) std.debug.print("warning: {s} spills {d} bytes of registers\n", .{ name, kp.spillMemSize });
        return k;
    }

    /// The kernel reads device memory through addresses held in buffers: the driver keeps every allocation resident
    pub fn indirect(g: *Gpu, k: H) Error!void {
        try check(g.api.zeKernelSetIndirectAccess(k, 1 << 1)); // ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE
    }

    pub fn arg(g: *Gpu, k: H, i: u32, v: anytype) Error!void {
        try check(g.api.zeKernelSetArgumentValue(k, i, @sizeOf(@TypeOf(v)), @ptrCast(&v)));
    }

    /// Host-visible timestamped events (GPU time of each kernel).
    pub fn events(g: *Gpu, comptime n: usize) Error![n]H {
        @setEvalBranchQuota(100000);
        var pool: H = null;
        try check(g.api.zeEventPoolCreate(g.context, &.{ .flags = pool_host_visible | pool_kernel_timestamp, .count = n }, 1, @ptrCast(&g.device), &pool));
        _ = try keep(&g.owned.pools, pool, g.api.zeEventPoolDestroy);
        var out: [n]H = @splat(null);
        for (&out, 0..) |*e, i| {
            try check(g.api.zeEventCreate(pool, &.{ .index = @intCast(i) }, e));
            _ = try keep(&g.owned.events, e.*, g.api.zeEventDestroy);
        }
        return out;
    }

    /// GPU duration of a kernel signalled on `ev`, in seconds.
    pub fn seconds(g: *Gpu, ev: H) Error!f64 {
        var r: KernelTimestampResult = .{};
        try check(g.api.zeEventQueryKernelTimestamp(ev, &r));
        return @as(f64, @floatFromInt(r.global_end -% r.global_start)) / @as(f64, @floatFromInt(g.props.timerResolution));
    }
};
