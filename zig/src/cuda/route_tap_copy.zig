//! Async device-to-host copy of one layer's global top-k into the route-tap ring (rank 0, when live).
const tap = @import("core").route_tap;
const Stream = @import("stream.zig").Stream;

/// Copy `rows * slots` int32 picks and fp32 weights from device pointers, before any localisation.
pub fn afterPick(s: Stream, picks: u64, wts: u64, rows: usize, slots: usize) void {
    if (!tap.live()) return;
    const n: u16 = @intCast(@min(rows * slots, tap.max_ids));
    if (n == 0) return;
    const sl = tap.reserve(n) orelse return;
    const nb = @as(usize, n) * 4;
    s.d.check(s.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(sl.ids.ptr), picks, nb, s.handle), "cuMemcpyDtoHAsync") catch {};
    s.d.check(s.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(sl.weights.ptr), wts, nb, s.handle), "cuMemcpyDtoHAsync") catch {};
}
