//! A tower call whose completion neither its stream nor the context proves ends the process before anything is freed.
const std = @import("std");
const cuda = @import("cuda");

/// Waits for `s`, then for the context; when neither completes, the ranks hear why and this process exits.
pub fn drain(d: *const cuda.Driver, s: cuda.Stream) void {
    if (d.api.cuStreamSynchronize(s.handle) == 0) return;
    if (d.api.cuCtxSynchronize() == 0) return;
    cuda.tp_watch.fail("Qwen vision stream and context did not drain");
    std.c._exit(1);
}
