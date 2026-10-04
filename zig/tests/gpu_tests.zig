//! GPU runtime tests, model-free and small. Run them under the machine's GPU lock.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");
const basic = @import("basic.zig");
const ordering = @import("ordering.zig");
const overlap = @import("overlap.zig");

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gpu = try common.Gpu.init();
    defer gpu.deinit();
    std.debug.print("device: {s}\n", .{gpu.device.name()});
    const tests = .{
        .{ "vector add", basic.vectorAdd },
        .{ "zero-copy buffer over page-aligned mmap", basic.zeroCopy },
        .{ "residency set", basic.residencySet },
        .{ "concurrent dispatch with barriers", ordering.concurrentBarriers },
        .{ "indirect command buffer replay with new arguments", ordering.icbReplay },
        .{ "shared-event host/GPU overlap", overlap.eventOverlap },
    };
    var failed: usize = 0;
    inline for (tests) |t| {
        const inner = mtl.objc.Pool.push();
        defer inner.pop();
        if (t[1](gpu)) {
            std.debug.print("PASS {s}\n", .{t[0]});
        } else |err| {
            failed += 1;
            std.debug.print("FAIL {s}: {s}\n", .{ t[0], @errorName(err) });
        }
    }
    std.debug.print("{d} of {d} passed\n", .{ tests.len - failed, tests.len });
    if (failed > 0) std.process.exit(1);
}
