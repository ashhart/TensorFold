//! Process-wide custody admission includes active, quarantined and callback-disposal owners.
const custody_sync = @import("copy_source_sync.zig");
const std = @import("std");
const builtin = @import("builtin");
pub const Limits = struct { bytes: usize = 1 << 30, nodes: usize = 4096 };
var limits: Limits = .{};
var bytes: usize = 0;
var nodes: usize = 0;
var mutex: custody_sync.Mutex = .{};
pub fn acquire(size: usize, count: usize) error{CopySourceLimit}!void {
    mutex.lock();
    defer mutex.unlock();
    if (size > limits.bytes - bytes or count > limits.nodes - nodes) return error.CopySourceLimit;
    bytes += size;
    nodes += count;
}
/// Custody that already exists (a buffer a failed wait left in use) counts even past the limit.
pub fn charge(size: usize) void {
    mutex.lock();
    defer mutex.unlock();
    bytes += size;
}
pub fn release(size: usize, count: usize) void {
    mutex.lock();
    defer mutex.unlock();
    std.debug.assert(size <= bytes and count <= nodes);
    bytes -= size;
    nodes -= count;
}
pub fn usage() struct { bytes: usize, nodes: usize } {
    mutex.lock();
    defer mutex.unlock();
    return .{ .bytes = bytes, .nodes = nodes };
}
pub fn setForTest(value: Limits) void {
    if (!builtin.is_test) @compileError("test only");
    mutex.lock();
    defer mutex.unlock();
    std.debug.assert(bytes == 0 and nodes == 0);
    limits = value;
}
