const cuda = @import("cuda");

/// The compute stream as a pass's pinned staging uses it: copies from it, an event behind them, and a drain.
pub const Uploads = struct {
    stream: cuda.Stream,
    event: cuda.Event,

    pub fn copy(u: Uploads, dst: u64, bytes: []const u8) !void {
        if (@import("exl3/graph/cuda_exl3_graph.zig").capturing) return @import("exl3/graph/cuda_exl3_graph.zig").pinnedCopy(u.stream, dst, bytes); // a copy node reads the pinned bytes at each replay
        try cuda.copy_source.enqueue(u.stream.d, dst, bytes, u.stream);
    }
    pub fn record(u: Uploads) !void {
        try u.event.record(u.stream);
    }
    pub fn wait(u: Uploads) !void {
        try u.event.synchronize();
    }
    pub fn drain(u: Uploads) !void {
        try u.stream.synchronize();
    }
};
