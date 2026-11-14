//! HIP's `stage.Device`: the proxy's copies on a stream of its own, and 64-bit words written and awaited on a stream.
const std = @import("std");
const sys = @import("sys");
const collective = @import("collective.zig");
const log = @import("log.zig");
const stage = @import("stage.zig");
const sock = @import("sock.zig");
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;
const Error = collective.Error;

const Result = c_int;
const P = ?*anyopaque;

/// What staging calls, each the HIP runtime's exact export.
const Api = struct {
    hipGetDevice: *const fn (*c_int) callconv(.c) Result,
    hipDeviceGetAttribute: *const fn (*c_int, c_int, c_int) callconv(.c) Result,
    hipStreamCreateWithFlags: *const fn (*Stream, c_uint) callconv(.c) Result,
    hipStreamDestroy: *const fn (Stream) callconv(.c) Result,
    hipStreamQuery: *const fn (Stream) callconv(.c) Result,
    hipMemcpyDtoHAsync: *const fn (P, P, usize, Stream) callconv(.c) Result,
    hipMemcpyHtoDAsync: *const fn (P, ?*const anyopaque, usize, Stream) callconv(.c) Result,
    hipStreamWriteValue64: *const fn (Stream, P, u64, c_uint) callconv(.c) Result,
    hipStreamWaitValue64: *const fn (Stream, P, u64, c_uint, u64) callconv(.c) Result,
    hipStreamIsCapturing: *const fn (Stream, *c_int) callconv(.c) Result,
    hipHostMalloc: *const fn (*P, usize, c_uint) callconv(.c) Result,
    hipHostFree: *const fn (P) callconv(.c) Result,
};

/// hipDeviceAttributeCanUseStreamWaitValue.
const can_wait_value: c_int = 10013;
/// hipHostMallocMapped | hipHostMallocCoherent: the GPU sees the proxy's words as the proxy writes them.
const pinned: c_uint = 0x2 | 0x40000000;
/// hipStreamWaitValueGte.
const at_least: c_uint = 0;
/// hipStreamNonBlocking: the proxy's copies never wait for the legacy null stream.
const non_blocking: c_uint = 1;
/// hipErrorNotReady: the stream still has work queued.
const not_ready: Result = 600;
/// The proxy asks the runtime about a copy every copy_look_ns, spinning between asks, then naps once copy_spin_ns pass.
const copy_look_ns = 3 * std.time.ns_per_us;
const copy_spin_ns = 500 * std.time.ns_per_us;
const copy_nap_ns = 10 * std.time.ns_per_us;

/// The proxy copies on a stream of its own and waits for each copy; the caller's stream only writes and waits.
pub const Hip = struct {
    api: Api,
    own: Stream = null,
    /// A copy still queued past this fails its call and frees every stream: its queue may be one a parked stream holds.
    limit_ns: u64 = 30 * std.time.ns_per_s,

    /// The current GPU, through the HIP runtime `lib` the caller opened; refuses one whose streams cannot wait.
    pub fn init(lib: *sys.DynLib) Error!Hip {
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                log.warn("[tensorfold] tp: the HIP runtime has no {s}; TCP staging needs it", .{name});
                return error.Unsupported;
            };
        }
        var dev: c_int = 0;
        var can: c_int = 0;
        try call(api.hipGetDevice(&dev), "hipGetDevice");
        try call(api.hipDeviceGetAttribute(&can, can_wait_value, dev), "hipDeviceGetAttribute");
        if (can == 0) {
            log.warn("[tensorfold] tp: HIP device {d} cannot wait on a stream value; TCP staging refuses it", .{dev});
            return error.Unsupported;
        }
        var h: Hip = .{ .api = api };
        try call(api.hipStreamCreateWithFlags(&h.own, non_blocking), "hipStreamCreateWithFlags");
        return h;
    }

    pub fn deinit(h: *Hip) void {
        _ = h.api.hipStreamDestroy(h.own);
    }

    pub fn device(h: *Hip) stage.Device {
        return .{ .ptr = h, .vtable = &vtable };
    }

    const vtable: stage.Device.VTable = .{ .download = download, .upload = upload, .write = write, .wait = wait, .capturing = capturing, .alloc = alloc, .free = free };

    fn of(ptr: *anyopaque) *Hip {
        return @ptrCast(@alignCast(ptr));
    }

    fn call(res: Result, comptime what: []const u8) Error!void {
        if (res == 0) return;
        log.err("[tensorfold] tp: " ++ what ++ " failed (HIP error {d})", .{res});
        return error.BackendFailed;
    }

    fn download(ptr: *anyopaque, host: [*]u8, dev: DevicePtr, n: usize) Error!void {
        const h = of(ptr);
        try call(h.api.hipMemcpyDtoHAsync(host, @ptrFromInt(dev), n, h.own), "hipMemcpyDtoHAsync");
        return h.settle();
    }

    fn upload(ptr: *anyopaque, dev: DevicePtr, host: [*]const u8, n: usize) Error!void {
        const h = of(ptr);
        try call(h.api.hipMemcpyHtoDAsync(@ptrFromInt(dev), host, n, h.own), "hipMemcpyHtoDAsync");
        return h.settle();
    }

    /// The proxy's stream has finished its copy, or the copy failed, or `limit_ns` passed without it.
    fn settle(h: *Hip) Error!void {
        const t0 = sock.nowNs();
        while (true) {
            const res = h.api.hipStreamQuery(h.own);
            if (res != not_ready) return call(res, "hipStreamQuery");
            const waited = sock.nowNs() - t0;
            if (waited > h.limit_ns) {
                log.err("[tensorfold] tp: a staged copy was still queued after {d} ms", .{waited / std.time.ns_per_ms});
                return error.BackendFailed;
            }
            if (waited >= copy_spin_ns) {
                sock.sleepNs(copy_nap_ns);
                continue;
            }
            const next = sock.nowNs() + copy_look_ns; // asking too often slows the copy itself
            while (sock.nowNs() < next) std.atomic.spinLoopHint();
        }
    }

    fn write(ptr: *anyopaque, word: *u64, value: u64, s: Stream) Error!void {
        return call(of(ptr).api.hipStreamWriteValue64(s, word, value, 0), "hipStreamWriteValue64");
    }

    fn wait(ptr: *anyopaque, word: *u64, value: u64, s: Stream) Error!void {
        return call(of(ptr).api.hipStreamWaitValue64(s, word, value, at_least, std.math.maxInt(u64)), "hipStreamWaitValue64");
    }

    fn capturing(ptr: *anyopaque, s: Stream) Error!bool {
        var status: c_int = 0;
        try call(of(ptr).api.hipStreamIsCapturing(s, &status), "hipStreamIsCapturing");
        return status != 0;
    }

    fn alloc(ptr: *anyopaque, n: usize) Error![]align(64) u8 {
        var p: P = null;
        try call(of(ptr).api.hipHostMalloc(&p, n, pinned), "hipHostMalloc");
        const base: [*]align(64) u8 = @ptrCast(@alignCast(p orelse return error.BackendFailed));
        return base[0..n];
    }

    fn free(ptr: *anyopaque, slab: []align(64) u8) void {
        _ = of(ptr).api.hipHostFree(slab.ptr);
    }
};
