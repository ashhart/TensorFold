//! CUDA work whose completion neither its stream nor the context proves ends the process before anything is freed.
const std = @import("std");
const cuda = @import("cuda");

/// Waits for `s`, then for the context; when neither completes, the ranks hear why and this process exits.
pub fn drain(d: *const cuda.Driver, s: cuda.Stream) void {
    if (d.api.cuStreamSynchronize(s.handle) == 0) return;
    if (d.api.cuCtxSynchronize() == 0) return;
    cuda.tp_watch.fail("a CUDA stream and its context did not drain; nothing the GPU may still use is freed");
    std.c._exit(1);
}

/// DeviceBuffer.upload, whose wait is the default stream's; a failed wait drains it and the context before returning.
pub fn upload(buf: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    buf.upload(offset, bytes) catch |e| {
        drain(buf.d, .{ .d = buf.d, .handle = null });
        return e;
    };
}
