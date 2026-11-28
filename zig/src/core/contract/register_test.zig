//! register against the two-append pattern it replaces, under a failing allocator at each list's growth.
const std = @import("std");
const gates = @import("gates.zig");
const register = @import("register.zig").register;

const Probe = struct {
    frees: *usize,
    fn kernel(_: *Probe) gates.Kernel {
        return .{ .spec = undefined, .ctx = undefined, .vt = undefined };
    }
};

const Suite = struct {
    probes: std.ArrayList(*Probe) = .empty,
    kernels: std.ArrayList(gates.Kernel) = .empty,
    // the suites' teardown: every listed probe freed once
    fn deinit(s: *Suite, a: std.mem.Allocator) void {
        for (s.probes.items) |p| p.frees.* += 1;
        s.probes.deinit(a);
        s.kernels.deinit(a);
    }
};

const Way = enum { two_appends_errdefer, two_appends_bare, helper };

// one probe added the way `way` adds it; its errdefer, when it has one, counts a free (as the suites' frees `p`)
fn add(way: Way, a: std.mem.Allocator, s: *Suite, p: *Probe) !void {
    switch (way) {
        .two_appends_errdefer => {
            errdefer p.frees.* += 1;
            try s.probes.append(a, p);
            try s.kernels.append(a, p.kernel());
        },
        .two_appends_bare => {
            try s.probes.append(a, p);
            try s.kernels.append(a, p.kernel());
        },
        .helper => {
            errdefer p.frees.* += 1;
            try register(a, &s.probes, &s.kernels, p, p.kernel());
        },
    }
}

// the probe's frees after `add` under an allocator failing at `fail_index`, and the suite's teardown
fn frees(way: Way, fail_index: usize) usize {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
    const a = failing.allocator();
    var count: usize = 0;
    var probe: Probe = .{ .frees = &count };
    var s: Suite = .{};
    add(way, a, &s, &probe) catch {};
    s.deinit(a);
    return count;
}

test "register: a probe is freed exactly once whichever list's growth fails; two appends free it twice or never" {
    // fail_index 1: the kernel list's growth fails after the probe list took the probe
    try std.testing.expectEqual(@as(usize, 2), frees(.two_appends_errdefer, 1)); // errdefer and teardown: double free
    // fail_index 0: the probe list's growth fails; without an errdefer nothing frees the probe
    try std.testing.expectEqual(@as(usize, 0), frees(.two_appends_bare, 0)); // leak
    for ([_]usize{ 0, 1 }) |fail_index| try std.testing.expectEqual(@as(usize, 1), frees(.helper, fail_index));
    // no failure: the lists own the probe, and teardown frees it once
    try std.testing.expectEqual(@as(usize, 1), frees(.helper, std.math.maxInt(usize)));
}

test "register: takes the lists as pointers or as pointer fields" {
    const a = std.testing.allocator;
    var count: usize = 0;
    var probe: Probe = .{ .frees = &count };
    var probes: std.ArrayList(*Probe) = .empty;
    defer probes.deinit(a);
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(a);
    const field: struct { kernels: *std.ArrayList(gates.Kernel) } = .{ .kernels = &kernels };
    try register(a, &probes, &field.kernels, &probe, probe.kernel());
    try register(a, &probes, field.kernels, &probe, probe.kernel());
    try std.testing.expectEqual(@as(usize, 2), probes.items.len);
    try std.testing.expectEqual(@as(usize, 2), kernels.items.len);
}
