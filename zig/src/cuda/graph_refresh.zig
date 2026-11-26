//! graph_cache's refresh step on the driver: a hit's body recorded again, its executable updated or replaced.
const std = @import("std");
const graph = @import("graph.zig");
const Stream = @import("stream.zig").Stream;
const gc = @import("graph_cache.zig");
const Held = gc.CudaEngine.Held;

/// The body recorded now and its parameters taken in place; launches of another shape get a new executable.
pub fn refresh(ctx: *anyopaque, exec: *anyopaque, stream: Stream, body: gc.Body) anyerror!*anyopaque {
    const h: *Held = @ptrCast(@alignCast(exec));
    try graph.beginCapture(stream, .thread_local);
    body.run(body.ctx, stream) catch |err| {
        // end the capture so the stream is usable again; the partial graph is dropped
        if (graph.endCapture(stream)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    var g = try graph.endCapture(stream);
    defer g.deinit();
    if (try h.exec.update(g) == .success) return h;
    const fresh = try std.heap.c_allocator.create(Held);
    errdefer std.heap.c_allocator.destroy(fresh);
    fresh.exec = try g.instantiate();
    errdefer fresh.exec.deinit();
    try fresh.exec.upload(stream);
    gc.CudaEngine.free(ctx, exec);
    return fresh;
}
