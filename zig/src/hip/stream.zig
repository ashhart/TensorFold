//! HIP nonblocking streams; callers synchronize before releasing in-flight buffers.
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");

pub const Stream = struct {
    r: *const runtime.Runtime,
    handle: abi.Stream,

    pub fn init(r: *const runtime.Runtime) runtime.Error!Stream {
        var handle: abi.Stream = null;
        try runtime.check(r.api.hipStreamCreateWithFlags(&handle, 1));
        if (handle == null) return error.Invalid;
        return .{ .r = r, .handle = handle };
    }

    pub fn synchronize(self: Stream) runtime.Error!void {
        try runtime.check(self.r.api.hipStreamSynchronize(self.handle));
    }

    pub fn deinit(self: *Stream) void {
        _ = self.r.api.hipStreamDestroy(self.handle);
        self.* = undefined;
    }
};
