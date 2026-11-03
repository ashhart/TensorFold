//! Fallible post-load allocations, synchronization and warmup before the engine takes ownership.
const std = @import("std");
const cuda = @import("cuda");
const DeviceBuffer = cuda.DeviceBuffer;
const profile = @import("cuda_startup_profile.zig");
const recurrent = @import("cuda_recurrent.zig");
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

pub fn finishWarm(e: anytype, io: std.Io, started: i96) !void {
    const warm_start = profile.now(io);
    e.startup = .{ .load = warm_start - started };
    errdefer recurrent.shutdown(e);
    e.startup.add(try profile.withCache(e.gpa, io, e, @import("cuda_cache.zig").Cache));
}

/// Every path a request takes, run once now: WDDM stalls a process's first forwards for seconds (a 20 s launch).
pub fn warmCache(e: anytype, warm_cache: *@import("cuda_cache.zig").Cache) !void {
    const ids: [32]u32 = @splat(0);
    const greedy: @import("cuda_engine.zig").Draw = .{ .rule = null, .position = 33 };
    try e.forward(&.{.{ .cache = warm_cache, .rows = ids.len }}, &ids, .last); // a prompt chunk: the prompt GEMMs
    try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{greedy} });
    try e.forward(&.{.{ .cache = warm_cache, .rows = 1 }}, ids[0..1], .{ .draws = &.{.{ .rule = .{ .seed = 1, .inv_t = 1.0 / 0.7, .top_p = 0.95, .near = 20.0, .min_log = -std.math.inf(f32), .top_k = 20 }, .position = 34 }} });
    // a tree window and a branch of it kept: the keep kernels, and the forward's branch path
    try e.forward(&.{.{ .cache = warm_cache, .rows = 3, .parents = &.{ -1, 0, 0 } }}, ids[0..3], .{ .draws = &.{ greedy, greedy, greedy } });
    try warm_cache.keep(e, &.{ 0, 2 });
    try e.stream.synchronize();
    recurrent.completed(e);
}

pub fn tapBytes(e: anytype) usize {
    return @import("cuda_cache.zig").tap_rows * e.n_taps * e.g.hidden * 2;
}
