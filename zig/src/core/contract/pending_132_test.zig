//! Explicit compiler selection keeps newly recorded 13.0 resources separate from later receipts.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

test "compiler receipts: new 13.0 records preserve independent 13.2 selection" {
    const r = for (registry.all_cuda) |row| {
        if (std.mem.eql(u8, row.launch.name, "tf_topk_slices")) break row;
    } else return error.MissingPendingRecord;
    const want: [5]limits.Usage = .{
        .{ .regs = 24, .static = 16448, .bound = 256 },
        .{ .regs = 20, .static = 16448, .bound = 256 },
        .{ .regs = 20, .static = 16448, .bound = 256 },
        .{ .regs = 28, .static = 16448, .bound = 256 },
        .{ .regs = 28, .static = 16448, .bound = 256 },
    };
    for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 } }) |version|
        try std.testing.expectEqualDeep(want, registry.usageFor(r, version));
    try std.testing.expectEqualDeep(want, registry.usageFor(r, .{ .major = 13, .minor = 2 }));
    try std.testing.expectEqualDeep(want, registry.usageFor(r, .{ .major = 13, .minor = 3 }));
    var unseen = r;
    unseen.launch.name = "an unmeasured future launch";
    try std.testing.expectEqualDeep(registry.none, registry.usageFor(unseen, .{ .major = 13, .minor = 2 }));
}

test "prompt int8: changed kernels stay unmeasured and decode receipts belong to exactly 13.3" {
    for ([_][]const u8{ "w4a8_prefill tf_w4a8_quantize", "w4a8_prefill tf_w4a8_prefill" }) |name| {
        const row = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.name, name)) break r;
        } else return error.MissingInt8Record;
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(row, version));
    }
    for (@import("measurements/qwen_lanes_133.zig").records) |receipt| {
        const row = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.name, receipt.name)) break r;
        } else return error.MissingDecodeRecord;
        try std.testing.expectEqualDeep(row.usage, registry.usageFor(row, .{ .major = 13, .minor = 0 }));
        for ([_]?limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 4 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(row, version));
        try std.testing.expectEqualDeep(receipt.usage, registry.usageFor(row, .{ .major = 13, .minor = 3 }));
    }
}

test "compiler receipts: 13.0 additions cannot replace a different later measurement" {
    const row = for (registry.all_cuda) |r| {
        if (std.mem.eql(u8, r.launch.name, "qwen35 tf_q35_gdn_slots_1")) break r;
    } else return error.MissingLaneRecord;
    const old = registry.usageFor(row, .{ .major = 13, .minor = 0 });
    const later = registry.usageFor(row, .{ .major = 13, .minor = 2 });
    try std.testing.expectEqual(@as(u32, 0), old[4].static);
    try std.testing.expectEqual(@as(u32, 48), old[4].regs);
    try std.testing.expectEqual(@as(u32, 5120), later[4].static);
    try std.testing.expectEqual(@as(u32, 40), later[4].regs);
    var absent = row;
    absent.launch.name = "a launch with only a 13.0 receipt";
    try std.testing.expectEqualDeep(old, registry.usageFor(absent, .{ .major = 13, .minor = 1 }));
    for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
        try std.testing.expectEqualDeep(registry.none, registry.usageFor(absent, version));
}

test "compiler receipts: Qwen BF16 rows keep measured 13.0 usage and pending later releases" {
    const names = [_][]const u8{ "tf_q35_embed_bf16", "tf_q35_dense_bf16" };
    const regs = [_][5]u32{ .{ 16, 16, 16, 40, 40 }, .{ 34, 40, 40, 42, 42 } };
    for (names, regs, 0..) |symbol, counts, n| {
        const row = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, r.launch.symbol, symbol)) break r;
        } else return error.MissingBf16Record;
        try std.testing.expectEqualStrings("qwen35_bf16_rows", row.launch.image);
        try std.testing.expectEqual(@as(u32, 256), row.launch.threads);
        var want: [5]limits.Usage = undefined;
        for (&want, counts) |*u, count| u.* = .{ .regs = count, .static = if (n == 0) 0 else 32, .bound = 256 };
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 } }) |version|
            try std.testing.expectEqualDeep(want, registry.usageFor(row, version));
        for (limits.sms, want) |sm, usage| {
            var problems: std.ArrayList(u8) = .empty;
            defer problems.deinit(std.testing.allocator);
            try std.testing.expectEqual(@as(usize, 0), try limits.cuda(row.launch, usage, sm, &problems, std.testing.allocator));
        }
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(row, version));
    }
}
