//! What the HIP staging tests and `tf-tp-test tcp-hip` call themselves: the runtime, streams, buffers and copies.
const std = @import("std");
const sys = @import("sys");
const collective = @import("collective.zig");
const DevicePtr = collective.DevicePtr;
const Stream = collective.Stream;

const Result = c_int;
const P = ?*anyopaque;

const Api = struct {
    hipInit: *const fn (c_uint) callconv(.c) Result,
    hipGetDeviceCount: *const fn (*c_int) callconv(.c) Result,
    hipSetDevice: *const fn (c_int) callconv(.c) Result,
    hipStreamCreateWithFlags: *const fn (*Stream, c_uint) callconv(.c) Result,
    hipStreamDestroy: *const fn (Stream) callconv(.c) Result,
    hipStreamSynchronize: *const fn (Stream) callconv(.c) Result,
    hipStreamBeginCapture: *const fn (Stream, c_int) callconv(.c) Result,
    hipStreamEndCapture: *const fn (Stream, *P) callconv(.c) Result,
    hipGraphDestroy: *const fn (P) callconv(.c) Result,
    hipMalloc: *const fn (*P, usize) callconv(.c) Result,
    hipFree: *const fn (P) callconv(.c) Result,
    hipMemcpyHtoD: *const fn (P, ?*const anyopaque, usize) callconv(.c) Result,
    hipMemcpyDtoH: *const fn (P, P, usize) callconv(.c) Result,
};

pub const Error = error{ NoHip, HipFailed };

/// hipStreamNonBlocking: a test's copies on the null stream never wait for a stream a test holds.
const non_blocking: c_uint = 1;

pub const Rig = struct {
    lib: sys.DynLib,
    api: Api,

    /// The HIP runtime by its usual name, with a GPU; `NoHip` when either is absent.
    pub fn open() Error!Rig {
        var lib = sys.DynLib.open("libamdhip64.so") catch return error.NoHip;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| @field(api, name) = lib.lookup(T, name) orelse return error.NoHip;
        var n: c_int = 0;
        if (api.hipInit(0) != 0 or api.hipGetDeviceCount(&n) != 0 or n == 0) return error.NoHip;
        return .{ .lib = lib, .api = api };
    }

    pub fn close(r: *Rig) void {
        r.lib.close();
    }

    fn ok(res: Result) Error!void {
        if (res != 0) return error.HipFailed;
    }

    pub fn use(r: *Rig, device: u32) Error!void {
        return ok(r.api.hipSetDevice(@intCast(device)));
    }

    pub fn stream(r: *Rig) Error!Stream {
        var s: Stream = null;
        try ok(r.api.hipStreamCreateWithFlags(&s, non_blocking));
        return s;
    }

    pub fn destroy(r: *Rig, s: Stream) void {
        _ = r.api.hipStreamDestroy(s);
    }

    pub fn sync(r: *Rig, s: Stream) Error!void {
        return ok(r.api.hipStreamSynchronize(s));
    }

    /// Starts capturing `s` (relaxed mode); `endCapture` drops the graph.
    pub fn capture(r: *Rig, s: Stream) Error!void {
        return ok(r.api.hipStreamBeginCapture(s, 2));
    }

    pub fn endCapture(r: *Rig, s: Stream) Error!void {
        var g: P = null;
        try ok(r.api.hipStreamEndCapture(s, &g));
        if (g != null) _ = r.api.hipGraphDestroy(g);
    }

    pub fn alloc(r: *Rig, n: usize) Error!DevicePtr {
        var p: P = null;
        try ok(r.api.hipMalloc(&p, @max(n, 1)));
        return @intFromPtr(p);
    }

    pub fn free(r: *Rig, p: DevicePtr) void {
        _ = r.api.hipFree(@ptrFromInt(p));
    }

    /// dev <- host, landed: a copy from pageable memory may return before its DMA ends, so the null stream syncs too.
    pub fn put(r: *Rig, dev: DevicePtr, host: []const u8) Error!void {
        if (host.len == 0) return;
        try ok(r.api.hipMemcpyHtoD(@ptrFromInt(dev), host.ptr, host.len));
        try ok(r.api.hipStreamSynchronize(null));
    }

    pub fn get(r: *Rig, host: []u8, dev: DevicePtr) Error!void {
        if (host.len > 0) try ok(r.api.hipMemcpyDtoH(host.ptr, @ptrFromInt(dev), host.len));
    }
};
