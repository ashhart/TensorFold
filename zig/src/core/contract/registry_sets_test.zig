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
    const topk = named("flashnext_attn fn_attn_fused").?;
    const rms = named("norms tf_rms_rows").?;
    const cases = [_]struct { ?limits.Nvcc, limits.Nvcc, u32, u32 }{
        .{ null, .{ .major = 13, .minor = 0 }, 121, 45 },
        .{ .{ .major = 12, .minor = 8 }, .{ .major = 13, .minor = 0 }, 121, 45 },
        .{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 0 }, 121, 45 },
        .{ .{ .major = 13, .minor = 1 }, .{ .major = 13, .minor = 0 }, 121, 45 },
        .{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 2 }, 121, 43 },
        .{ .{ .major = 13, .minor = 3 }, .{ .major = 13, .minor = 3 }, 122, 43 },
        .{ .{ .major = 13, .minor = 4 }, .{ .major = 13, .minor = 4 }, 122, 43 },
        .{ .{ .major = 14, .minor = 0 }, .{ .major = 13, .minor = 4 }, 122, 43 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c[1], setFor(c[0]));
        try std.testing.expectEqual(c[2], usageFor(topk, c[0])[3].regs);
        try std.testing.expectEqual(c[3], usageFor(rms, c[0])[4].regs);
        try std.testing.expectEqual(topk.usage[0], usageFor(topk, c[0])[0]);
    }
}

test "mHC receipts: nvcc 13.0 has its own records and 13.1 through 13.3 remain pending" {
    const route = named("qwen35_mhc route").?;
    const old = usageFor(route, .{ .major = 13, .minor = 0 });
    try std.testing.expectEqual(@as(u32, 39), old[3].regs);
    try std.testing.expectEqual(@as(u32, 1068), old[3].static);
    for (1..4) |minor| {
        try std.testing.expectEqual(registry.none, usageFor(route, .{ .major = 13, .minor = @intCast(minor) }));
    }
    const usage = usageFor(route, .{ .major = 13, .minor = 4 });
    try std.testing.expectEqual(@as(u32, 39), usage[3].regs);
    try std.testing.expectEqual(@as(u32, 1068), usage[3].static);
    try std.testing.expectEqual(usage, usageFor(route, .{ .major = 14, .minor = 0 }));
}

test {
    _ = @import("pending_132_test.zig");
    _ = @import("exl3_grouped_records.zig");
    _ = @import("ernie5_5_records.zig");
    _ = @import("proposal_sampling_records.zig");
    _ = @import("pocket_records.zig");
    _ = @import("experts_bf16_records.zig");
    _ = @import("experts_mxfp4_records.zig");
    _ = @import("nemo_audio_cuda_records.zig");
}
