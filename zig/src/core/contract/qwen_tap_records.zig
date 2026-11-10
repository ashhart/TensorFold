//! Compact tap receipts: rebuilt nvcc 13.5 writer and unchanged nvcc 13.0 gathers, only measured SM profiles.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");
const empty: limits.Usage = .{ .regs = 0, .static = 0 };

pub const launches = [_]limits.CudaLaunch{
    .{ .name = "qwen35 tf_q35_tap", .symbol = "tf_q35_tap", .image = "qwen35", .threads = 256 },
    .{ .name = "qwen35_draft tf_q35d_gather_taps", .symbol = "tf_q35d_gather_taps", .image = "qwen35_draft", .threads = 256 },
    .{ .name = "qwen35_draft tf_q35d_gather_rows", .symbol = "tf_q35d_gather_rows", .image = "qwen35_draft", .threads = 256 },
};

// Core cubin parser, work/memory-fatbins; the writer's ptxas log also confirms no stack or spills.
const writer: limits.Usage = .{ .regs = 25, .static = 0, .bound = 256 };
const gather: limits.Usage = .{ .regs = 20, .static = 0 };
pub const measured_135: [launches.len][limits.sms.len]limits.Usage = .{
    .{ empty, empty, empty, writer, writer },
    @splat(empty),
    @splat(empty),
};
pub const measured_130: [launches.len][limits.sms.len]limits.Usage = .{
    @splat(empty),
    .{ empty, empty, empty, gather, empty },
    .{ empty, empty, empty, gather, empty },
};
// The tap writer and selected gather already belong to cuda_lane_pending; only the contiguous gather is new.
pub const records = [_]r.CudaRecord{.{ .launch = launches[1], .usage = measured_130[1] }};

pub fn index(image: []const u8, symbol: []const u8) ?usize {
    for (launches, 0..) |launch, i|
        if (std.mem.eql(u8, image, launch.image) and std.mem.eql(u8, symbol, launch.symbol)) return i;
    return null;
}

/// Changed source needs an explicit receipt even when an older compiler set knows the same symbol.
pub fn lookup(record: r.CudaRecord, release: ?limits.Nvcc) ?[limits.sms.len]limits.Usage {
    const i = index(record.launch.image, record.launch.symbol) orelse return null;
    const version = release orelse return r.none;
    if (version.major == 13 and version.minor == 5) return measured_135[i];
    if (version.major == 13 and version.minor == 0) return measured_130[i];
    return r.none;
}

test "compact target tap launches register once and unmeasured compilers remain unqualified" {
    for (launches, 0..) |launch, i| {
        var seen: usize = 0;
        for (r.all_cuda) |record| {
            if (index(record.launch.image, record.launch.symbol) != i) continue;
            seen += 1;
            try std.testing.expectEqual(@as(u32, 256), record.launch.threads);
            try std.testing.expectEqual(@as(u32, 0), record.launch.dynamic);
            try std.testing.expectEqual(r.none, r.usageFor(record, null));
            try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 3 }));
            try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 6 }));
            try std.testing.expectEqual(measured_135[i], r.usageFor(record, .{ .major = 13, .minor = 5 }));
            try std.testing.expectEqual(measured_130[i], r.usageFor(record, .{ .major = 13, .minor = 0 }));
        }
        _ = launch;
        try std.testing.expectEqual(@as(usize, 1), seen);
    }
}

test "compact tap receipts retain the exact measured compiler and SM scope" {
    try std.testing.expectEqual(@as(u32, 25), measured_135[0][3].regs);
    try std.testing.expectEqual(measured_135[0][3], measured_135[0][4]);
    for (measured_135[0][0..3]) |usage| try std.testing.expectEqual(empty, usage);
    for ([_]usize{ 1, 2 }) |i| {
        try std.testing.expectEqual(@as(u32, 20), measured_130[i][3].regs);
        try std.testing.expectEqual(@as(u32, 0), measured_130[i][4].regs);
        try std.testing.expectEqual(r.none, measured_135[i]);
    }
}
