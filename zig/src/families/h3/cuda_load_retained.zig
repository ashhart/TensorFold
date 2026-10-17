//! Failed weight construction keeps its destinations and all CUDA parents together until recovery.
const std = @import("std");
const Model = @import("cuda.zig").Model;
var head: ?*Model = null;
var mutex: std.atomic.Mutex = .unlocked;
fn lock() void {
    while (!mutex.tryLock()) std.Thread.yield() catch {};
}
pub fn keep(m: *Model) void {
    lock();
    defer mutex.unlock();
    m.retained_next = head;
    head = m;
}
var recovery: std.atomic.Mutex = .unlocked;
/// A retry open or another model's close retries every failed loader in its own context.
pub fn recover() usize {
    // A detached transaction remains a pending owner at every parent/admission boundary.
    if (!recovery.tryLock()) return 1;
    defer recovery.unlock();
    lock();
    var at = head;
    head = null;
    mutex.unlock();
    if (comptime @import("builtin").is_test) if (Testing.detached) |hook| {
        Testing.detached = null;
        hook();
    };
    while (at) |m| {
        at = m.retained_next;
        m.retained_next = null;
        m.tryDeinit() catch keep(m);
    }
    lock();
    defer mutex.unlock();
    var n: usize = 0;
    at = head;
    while (at) |m| {
        n += 1;
        at = m.retained_next;
    }
    return n;
}

pub const Testing = if (@import("builtin").is_test) struct {
    pub var detached: ?*const fn () void = null;
} else struct {};

pub fn admit() !void {
    if (recover() != 0) return error.LoadPending;
}
