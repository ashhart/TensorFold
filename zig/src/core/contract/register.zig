//! Publishing a contract probe: its list and the kernel list take it together or not at all.
const std = @import("std");
const gates = @import("gates.zig");

/// Both lists take the probe or neither does: a failure leaves the caller's errdefer owning it, success the lists.
pub fn register(a: std.mem.Allocator, probes: anytype, kernels: anytype, probe: anytype, kernel: gates.Kernel) error{OutOfMemory}!void {
    const ps = list(probes); // a list, or a pointer field holding one
    const ks = list(kernels);
    try ps.ensureUnusedCapacity(a, 1);
    try ks.ensureUnusedCapacity(a, 1);
    ps.appendAssumeCapacity(probe);
    ks.appendAssumeCapacity(kernel);
}

fn List(comptime P: type) type {
    const child = @typeInfo(P).pointer.child;
    return if (@typeInfo(child) == .pointer) child else P;
}

fn list(p: anytype) List(@TypeOf(p)) {
    return if (comptime @typeInfo(@typeInfo(@TypeOf(p)).pointer.child) == .pointer) p.* else p;
}

test {
    _ = @import("register_test.zig");
}
