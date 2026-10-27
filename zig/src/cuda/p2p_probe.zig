//! TENSORFOLD_P2P_PROBE: one timed fixed-size NCCL all-gather after init, reported as bus bandwidth (off by default).
const std = @import("std");
const platform = @import("sys");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");

pub const variable = "TENSORFOLD_P2P_PROBE";
pub const bytes_per_rank: usize = 32 << 20;
const warmups = 2;
const iterations = 5;
const enqueue_ms = 60_000;

/// Set to anything but empty or 0. Every rank must agree: a rank that skips it leaves the others in the gather.
pub fn wanted(value: ?[]const u8) bool {
    const v = std.mem.trim(u8, value orelse return false, " ");
    return v.len > 0 and !std.mem.eql(u8, v, "0");
}

pub const Result = struct { ms: f64, bus_gbps: f64 };

/// nccl-tests' all-gather bus bandwidth: world * bytes gathered in `ms`, times (world - 1) / world.
pub fn busGbps(bytes: usize, world: usize, ms: f64) f64 {
    if (ms <= 0 or world < 2) return 0;
    const w: f64 = @floatFromInt(world);
    const total = @as(f64, @floatFromInt(bytes)) * w;
    return total / (ms * 1e-3) * (w - 1) / w / 1e9;
}

/// In-place all-gathers of `bytes_per_rank` a rank on a stream of its own: two warmups, then five timed.
pub fn run(d: *const driver.Driver, lib: *const nccl.Library, comm: nccl.Comm, rank: usize, world: usize) !Result {
    var buf = try memory.DeviceBuffer.alloc(d, bytes_per_rank * world);
    defer buf.free();
    try buf.fill8(0, null);
    var s = try stream_.Stream.init(d, true);
    defer s.deinit();
    var t0 = try stream_.Event.init(d, true);
    defer t0.deinit();
    var t1 = try stream_.Event.init(d, true);
    defer t1.deinit();
    const mine = buf.ptr + rank * bytes_per_rank;
    for (0..warmups) |_| try gather(lib, comm, s, mine, buf.ptr);
    try s.synchronize();
    try t0.record(s);
    for (0..iterations) |_| try gather(lib, comm, s, mine, buf.ptr);
    try t1.record(s);
    try t1.synchronize();
    const ms = @as(f64, try stream_.Event.elapsedMs(t0, t1)) / iterations;
    return .{ .ms = ms, .bus_gbps = busGbps(bytes_per_rank, world, ms) };
}

/// One all-gather enqueued; a nonblocking communicator's in-progress call is waited for before the next.
fn gather(lib: *const nccl.Library, comm: nccl.Comm, s: stream_.Stream, mine: u64, all: u64) !void {
    const r = lib.api.ncclAllGather(mine, all, bytes_per_rank, .u8, comm, s.handle);
    if (r != 0 and r != nccl.in_progress) return error.ProbeFailed;
    var state: nccl.Result = r;
    var waited: u32 = 0;
    while (state == nccl.in_progress) : (waited += 1) {
        if (waited >= enqueue_ms) return error.ProbeTimedOut;
        if (lib.api.ncclCommGetAsyncError(comm, &state) != 0) return error.ProbeFailed;
        if (state != 0 and state != nccl.in_progress) return error.ProbeFailed;
        const ms: platform.c.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
        _ = platform.c.nanosleep(&ms, null);
    }
}

test "probe: unset, empty and 0 are off; anything else is on" {
    try std.testing.expect(!wanted(null));
    try std.testing.expect(!wanted(""));
    try std.testing.expect(!wanted("0"));
    try std.testing.expect(wanted("1"));
}

test "probe: bus bandwidth is nccl-tests' all-gather figure" {
    // 2 ranks x 32 MiB gathered in 10 ms: 6.71 GB/s total, half of it crossing the bus
    try std.testing.expectApproxEqAbs(@as(f64, 3.355), busGbps(32 << 20, 2, 10), 0.001);
    try std.testing.expectEqual(@as(f64, 0), busGbps(32 << 20, 1, 10));
}
