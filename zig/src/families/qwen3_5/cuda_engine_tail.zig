//! Fallible post-load allocations, synchronization and warmup before the engine takes ownership.
const std = @import("std");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;
const tables_len = @import("cuda_round.zig").bytes;

pub fn finish(e: anytype, io: std.Io, started: i96) !void {
    errdefer if (e.kit) |k| k.deinit(e.gpa);
    _ = try e.scratch(.allocate);
    e.tables = try DeviceBuffer.alloc(e.d, tables_len);
    errdefer e.tables.free();
    e.staged = try cuda.HostBuffer.alloc(e.d, tables_len);
    errdefer e.staged.free();
    e.host_logits = try cuda.HostBuffer.alloc(e.d, e.verify_rows * e.g.vocab * 2);
    errdefer e.host_logits.free();
    try e.stream.synchronize();
    try e.finishWarm(io, started);
}
