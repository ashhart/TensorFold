//! Async device-to-host copy of the global top-k before `tf_moe_localize` rewrites local ids.
const tap = @import("core").route_tap;
const runtime = @import("runtime.zig");
const abi = @import("abi.zig");

pub fn afterSelect(r: *const runtime.Runtime, stream: abi.Stream, picks: u64, wts: u64, rows: usize, slots: usize) void {
    if (!tap.live()) return;
    const n: u16 = @intCast(@min(rows * slots, tap.max_ids));
    if (n == 0) return;
    const sl = tap.reserve(n) orelse return;
    const nb = @as(usize, n) * 4;
    runtime.check(r.api.hipMemcpyDtoHAsync(@ptrCast(sl.ids.ptr), @ptrFromInt(picks), nb, stream)) catch {};
    runtime.check(r.api.hipMemcpyDtoHAsync(@ptrCast(sl.weights.ptr), @ptrFromInt(wts), nb, stream)) catch {};
}
