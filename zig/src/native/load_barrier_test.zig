//! Deterministic scheduling at the registry's detached-list boundary.
const std = @import("std");
var arrived = std.atomic.Value(bool).init(false);
var released = std.atomic.Value(bool).init(false);
pub fn reset() void {
    arrived.store(false, .release);
    released.store(false, .release);
}
pub fn pause() void {
    arrived.store(true, .release);
    while (!released.load(.acquire)) std.Thread.yield() catch {};
}
pub fn wait() void {
    while (!arrived.load(.acquire)) std.Thread.yield() catch {};
}
pub fn release() void {
    released.store(true, .release);
}
