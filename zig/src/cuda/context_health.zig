//! Sticky device faults belong to the retained CUDA context, shared across its streams and host threads.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Entry = struct { driver: usize = 0, context: abi.Context = null, refs: usize = 0, fault: abi.Result = 0 };
var entries: [64]Entry = @splat(.{});
var mutex: std.atomic.Mutex = .unlocked;

/// CUDA reports these asynchronous execution faults as requiring context destruction before further work.
pub fn sticky(code: abi.Result) bool {
    return switch (code) {
        214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 => true,
        else => false,
    };
}

/// Register before the context escapes; a second retain cannot clear an existing fault.
pub fn retain(d: *const Driver, context: abi.Context) !void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) {
        e.refs += 1;
        return;
    };
    for (&entries) |*e| if (e.refs == 0) {
        e.* = .{ .driver = @intFromPtr(d), .context = context, .refs = 1 };
        return;
    };
    return error.OutOfMemory;
}

/// Only the last owner removes a context; an engine close/reopen must still pass the driver's load checks.
pub fn release(d: *const Driver, context: abi.Context) void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) {
        e.refs -= 1;
        if (e.refs == 0) e.* = .{};
        return;
    };
}

pub fn observe(d: *const Driver, context: abi.Context, code: abi.Result) void {
    if (!sticky(code)) return;
    while (!mutex.tryLock()) std.Thread.yield() catch {};
    defer mutex.unlock();
    for (&entries) |*e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) {
        if (e.fault == 0) e.fault = code;
        return;
    };
}

/// Host-only readiness: no context binding, driver call or GPU wait on the HTTP thread.
pub fn healthy(d: *const Driver, context: abi.Context) bool {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
    defer mutex.unlock();
    for (entries) |e| if (e.refs > 0 and e.driver == @intFromPtr(d) and e.context == context) return e.fault == 0;
    return true;
}

test "CUDA context health: sticky faults survive success and another retain; other contexts and 800 stay healthy" {
    var d: Driver = undefined;
    const first: abi.Context = @ptrFromInt(0x3251);
    const other: abi.Context = @ptrFromInt(0x3252);
    try retain(&d, first);
    defer release(&d, first);
    try retain(&d, other);
    defer release(&d, other);
    const old = @import("copy_source.zig").current();
    defer @import("copy_source.zig").setCurrent(old);
    @import("copy_source.zig").setCurrent(first);
    try std.testing.expectError(error.CudaFailed, d.result(800));
    try std.testing.expect(healthy(&d, first));
    try std.testing.expectError(error.OutOfDeviceMemory, d.result(2));
    try std.testing.expect(healthy(&d, first));
    try std.testing.expectError(error.CudaContextLost, d.result(700));
    try d.result(0);
    try retain(&d, first);
    try std.testing.expect(!healthy(&d, first));
    release(&d, first);
    try std.testing.expect(!healthy(&d, first));
    try std.testing.expect(healthy(&d, other));
    for ([_]abi.Result{ 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 }) |code| try std.testing.expect(sticky(code));
    for ([_]abi.Result{ 0, 1, 2, 600, 701, 800, 801 }) |code| try std.testing.expect(!sticky(code));
}

test "CUDA context health: exhausted custody refuses another context before it can escape" {
    var d: Driver = undefined;
    var held: usize = 0;
    defer for (0..held) |i| release(&d, @ptrFromInt(0x10000 + i));
    for (0..64) |i| {
        try retain(&d, @ptrFromInt(0x10000 + i));
        held += 1;
    }
    try std.testing.expectError(error.OutOfMemory, retain(&d, @ptrFromInt(0x20000)));
}
