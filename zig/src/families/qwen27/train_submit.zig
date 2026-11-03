//! One row's unchanged native stages in one command buffer, with host encoding and completion measured separately.
const std = @import("std");

pub const Stats = struct {
    encode: f64 = 0,
    wait: f64 = 0,
    gpu: f64 = 0,
    buffers: usize = 0,
    waits: usize = 0,
};
pub fn now(io: ?std.Io) f64 {
    return if (io) |i| @as(f64, @floatFromInt(std.Io.Clock.awake.now(i).toNanoseconds())) / 1e9 else 0;
}
pub fn run(queue: anytype, program: anytype, stats: *Stats, io: ?std.Io) !f64 {
    const start = now(io);
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    var ended = false;
    defer if (!ended) e.end();
    try program.encode(e);
    e.end();
    ended = true;
    const encoded = now(io);
    cb.commit();
    cb.wait();
    stats.encode += encoded - start;
    stats.wait += now(io) - encoded;
    stats.buffers += 1;
    stats.waits += 1;
    if (cb.failure() != null) return error.LearningGpuFailure;
    const gpu = cb.gpuSeconds();
    stats.gpu += gpu;
    return gpu;
}
