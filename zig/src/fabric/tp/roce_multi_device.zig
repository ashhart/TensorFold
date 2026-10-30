//! Device setup owns every partial allocation until the CUDA context drains.
const cuda = @import("cuda");
pub fn open(r: anytype, d: *const cuda.Driver, image: []const u8) !bool {
    if (image.len == 0) return error.NoKernel;
    errdefer close(r);
    r.d = d;
    r.hr = try @import("mailbox.zig").HostRegister.open();
    try d.check(r.hr.?.register(r.region.ptr, r.region.len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
    r.registered = true;
    try d.check(d.api.cuMemHostGetDevicePointer_v2(&r.dev, r.region.ptr, 0), "cuMemHostGetDevicePointer");
    r.state = try cuda.DeviceBuffer.alloc(d, 16);
    try r.state.?.fill8(0, null);
    r.module = try cuda.Module.load(d, image);
    _ = try r.module.?.function("tp_roce_multi_abi_v1");
    r.func = try r.module.?.function("tp_roce_multi");
    r.chunk_abi = if (r.module.?.function("tp_roce_multi_chunk_abi")) |_| true else |_| false;
    return true;
}

pub fn close(r: anytype) void {
    if (r.retained) return;
    if (r.d) |d| {
        if ((r.registered or r.state != null or r.module != null) and d.api.cuCtxSynchronize() != cuda.abi.success) {
            r.retained = true;
            _ = @TypeOf(r.*).kept.fetchAdd(1, .monotonic);
            return;
        }
    }
    if (r.module) |*m| m.unload();
    if (r.state) |*s| s.free();
    if (r.hr) |*h| {
        if (r.registered) _ = h.unregister(r.region.ptr);
        h.lib.close();
    }
    r.module = null;
    r.state = null;
    r.hr = null;
    r.registered = false;
}
