//! What every synchronous download lands in first: this thread's page-locked memory, at most `most`
//! bytes, copied on into the caller's by the CPU. On GB10 a `cuMemcpyDtoH` straight into pageable
//! memory (the host allocator's, freshly mapped) can miss pages with no CUDA error: some keep what
//! they held, the bytes land on other pages of the process. Calls to `DeviceBuffer.download` stage
//! through here; `downloadAsync` still needs pinned memory, as documented.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// The largest piece one copy takes; larger downloads go through in pieces.
pub const most: usize = 4 << 20;

const Held = struct { d: *const Driver, bytes: []align(16) u8 };
threadlocal var held: ?Held = null;

/// This thread's page-locked buffer of at least `len` bytes, reallocated when a larger piece is asked
/// for. The driver allocates it directly, so it is not counted in `memory.usage`: it is one per
/// thread, bounded by `most`, and kept for that thread's next download.
pub fn take(d: *const Driver, len: usize) Error![]align(16) u8 {
    if (held) |h| {
        if (h.d == d and h.bytes.len >= len) return h.bytes[0..len];
        drop();
    }
    const size = @min(most, std.math.ceilPowerOfTwoAssert(usize, @max(len, 64 << 10)));
    var p: ?*anyopaque = null;
    try d.check(d.api.cuMemHostAlloc(&p, size, abi.host_alloc_portable), "cuMemHostAlloc");
    const base: [*]align(16) u8 = @ptrCast(@alignCast(p.?));
    held = .{ .d = d, .bytes = base[0..size] };
    return held.?.bytes[0..len];
}

/// Frees this thread's buffer, if one is held.
pub fn drop() void {
    if (held) |h| {
        _ = h.d.api.cuMemFreeHost(h.bytes.ptr);
        held = null;
    }
}
