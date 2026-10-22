//! Explicit compiler selection leaves earlier and unknown measurements pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

test "compiler receipts: measured 13.2 pending rows never promote 13.0 or unseen names" {
    const r = for (registry.all_cuda) |row| {
        if (std.mem.eql(u8, row.launch.name, "tf_topk_slices")) break row;
    } else return error.MissingPendingRecord;
    for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 1 } }) |version|
        try std.testing.expectEqualDeep(registry.none, registry.usageFor(r, version));
    const want: [5]limits.Usage = .{
        .{ .regs = 24, .static = 16448, .bound = 256 },
        .{ .regs = 20, .static = 16448, .bound = 256 },
        .{ .regs = 20, .static = 16448, .bound = 256 },
        .{ .regs = 28, .static = 16448, .bound = 256 },
        .{ .regs = 28, .static = 16448, .bound = 256 },
    };
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
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 4 }, .{ .major = 14, .minor = 0 } }) |version|
            try std.testing.expectEqualDeep(registry.none, registry.usageFor(row, version));
        try std.testing.expectEqualDeep(receipt.usage, registry.usageFor(row, .{ .major = 13, .minor = 3 }));
    }
}
