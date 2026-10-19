//! A partially built device owner remains reachable if completion cannot be established during unwind.
const std = @import("std");
const Forward = @import("forward.zig").Forward;
pub const Owner = struct { ptr: *anyopaque, drop: *const fn (*anyopaque) void };
pub fn attach(f: *Forward, ptr: *anyopaque, drop: *const fn (*anyopaque) void) !void {
    try f.init_owners.append(f.gpa, .{ .ptr = ptr, .drop = drop });
}
pub fn detach(f: *Forward, ptr: *anyopaque) void {
    for (f.init_owners.items, 0..) |o, i| if (o.ptr == ptr) {
        _ = f.init_owners.orderedRemove(i);
        return;
    };
}
pub fn failed(f: *Forward, ptr: *anyopaque) void {
    f.runner.d.synchronize() catch return;
    for (f.init_owners.items) |o| if (o.ptr == ptr) {
        detach(f, ptr);
        o.drop(ptr);
        return;
    };
}
pub fn deinit(f: *Forward) void {
    while (f.init_owners.pop()) |o| o.drop(o.ptr);
    f.init_owners.deinit(f.gpa);
}
