//! A thread's page-locked bounce for synchronous downloads on integrated GPUs.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const memory = @import("memory.zig");

/// The largest piece one copy takes; larger downloads go through in pieces.
pub const most: usize = 4 << 20;

const Held = struct { d: *const Driver, bytes: []align(16) u8 };
const Device = struct { d: *const Driver, integrated: bool };
threadlocal var held: ?Held = null;
threadlocal var device: ?Device = null;

/// Whether this thread's current device shares memory with the host; read once per driver.
pub fn integrated(d: *const Driver) Error!bool {
    if (device) |one| if (one.d == d) return one.integrated;
    var dev: abi.Device = 0;
    try d.check(d.api.cuCtxGetDevice(&dev), "cuCtxGetDevice");
    var v: c_int = 0;
    try d.check(d.api.cuDeviceGetAttribute(&v, .integrated, dev), "cuDeviceGetAttribute");
    device = .{ .d = d, .integrated = v != 0 };
    return v != 0;
}

/// This thread's page-locked buffer of at least `len` bytes; it counts in `memory.usage`.
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
    memory.retainHost(size);
    return held.?.bytes[0..len];
}

/// Frees this thread's buffer, if one is held.
pub fn drop() void {
    if (held) |h| {
        _ = h.d.api.cuMemFreeHost(h.bytes.ptr);
        memory.releaseRetainedHost(h.bytes.len);
        held = null;
    }
}

/// Forgets the cached device reading (tests use distinct drivers, but a fake may reuse one).
pub fn forget() void {
    device = null;
}
