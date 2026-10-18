//! Per-compiler CUDA record sets: each later set rises by release; a compiler takes its own set or the one below.
const std = @import("std");
const limits = @import("limits.zig");
const registry = @import("registry.zig");
const checkCuda = registry.checkCuda;
const cuda_release = registry.cuda_release;
const cuda_sets = registry.cuda_sets;
const newer = registry.newer;
const setFor = registry.setFor;
const usageFor = registry.usageFor;
const CudaRecord = registry.CudaRecord;

fn named(name: []const u8) ?CudaRecord {
    for (registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.name, name)) return r;
    return null;
}

test "each later nvcc's set rises by release and names registered launches that fit every SM" {
    const a = std.testing.allocator;
    var below = cuda_release;
    for (cuda_sets) |s| {
        try std.testing.expect(newer(s.release, below));
        below = s.release;
        for (s.usage) |o| {
            var r = named(o.name) orelse return error.UnregisteredOverride;
            r.usage = o.usage;
            try std.testing.expectEqual(@as(usize, 0), try checkCuda(a, &.{r}, false));
        }
    }
}

test "a compiler's records: its own set, else the newest below it, else 13.0's" {
    const topk = named("router tf_nemo_topk").?;
    const rms = named("norms tf_rms_rows").?;
    const cases = [_]struct { ?limits.Nvcc, limits.Nvcc, u32, u32 }{
        .{ null, .{ .major = 13, .minor = 0 }, 91, 45 },
        .{ .{ .major = 12, .minor = 8 }, .{ .major = 13, .minor = 0 }, 91, 45 },
        .{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 0 }, 91, 45 },
        .{ .{ .major = 13, .minor = 1 }, .{ .major = 13, .minor = 0 }, 91, 45 },
        .{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 2 }, 91, 43 },
        .{ .{ .major = 13, .minor = 3 }, .{ .major = 13, .minor = 3 }, 96, 43 },
        .{ .{ .major = 14, .minor = 0 }, .{ .major = 13, .minor = 3 }, 96, 43 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c[1], setFor(c[0]));
        try std.testing.expectEqual(c[2], usageFor(topk, c[0])[3].regs);
        try std.testing.expectEqual(c[3], usageFor(rms, c[0])[4].regs);
        try std.testing.expectEqual(topk.usage[0], usageFor(topk, c[0])[0]);
    }
}

test {
    _ = @import("pending_132_test.zig");
}
