//! Synthetic loaded topology shared by portable HTTP and POSIX benchmark tests.
const std = @import("std");
const json = @import("json");
const core = @import("core");
const Value = json.Value;

pub fn topology(a: std.mem.Allocator, world: u32, multi: bool) !Value {
    var all: [8]core.benchmark_group.Wire = undefined;
    for (all[0..world], 0..) |*w, i| {
        w.* = .{ .rank = @intCast(i), .world = world, .memory = 128 << 30, .gpu_memory = 120 << 30, .device_known = 1 };
        w.node[0] = if (multi) @intCast(i + 1) else 1;
        w.device[0] = @intCast(i + 1);
        @memcpy(w.cpu[0..11], "Example CPU");
        @memcpy(w.gpu[0..11], "Example GPU");
    }
    return (try json.parse(a, (try core.benchmark_group.assemble(a, all[0..world])).?)).ok;
}
