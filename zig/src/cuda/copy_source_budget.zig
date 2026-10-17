//! Process-wide custody admission includes active, quarantined and callback-disposal owners.
const std = @import("std");
const builtin = @import("builtin");
pub const Limits = struct { bytes: usize = 1 << 30, nodes: usize = 4096 };
var limits: Limits = .{};
var bytes: usize = 0;
var nodes: usize = 0;
var mutex: std.c.pthread_mutex_t = .{};
pub fn acquire(size: usize, count: usize) error{CopySourceLimit}!void {
    _ = std.c.pthread_mutex_lock(&mutex);
    defer _ = std.c.pthread_mutex_unlock(&mutex);
    if (size > limits.bytes - bytes or count > limits.nodes - nodes) return error.CopySourceLimit;
    bytes += size;
    nodes += count;
}
/// Custody that already exists (a buffer a failed wait left in use) counts even past the limit.
pub fn charge(size: usize) void {
    _ = std.c.pthread_mutex_lock(&mutex);
    defer _ = std.c.pthread_mutex_unlock(&mutex);
    bytes += size;
}
pub fn release(size: usize, count: usize) void {
    _ = std.c.pthread_mutex_lock(&mutex);
    defer _ = std.c.pthread_mutex_unlock(&mutex);
    std.debug.assert(size <= bytes and count <= nodes);
    bytes -= size;
    nodes -= count;
}
pub fn usage() struct { bytes: usize, nodes: usize } {
    _ = std.c.pthread_mutex_lock(&mutex);
    defer _ = std.c.pthread_mutex_unlock(&mutex);
    return .{ .bytes = bytes, .nodes = nodes };
}
pub fn setForTest(value: Limits) void {
    if (!builtin.is_test) @compileError("test only");
    _ = std.c.pthread_mutex_lock(&mutex);
    defer _ = std.c.pthread_mutex_unlock(&mutex);
    std.debug.assert(bytes == 0 and nodes == 0);
    limits = value;
}
