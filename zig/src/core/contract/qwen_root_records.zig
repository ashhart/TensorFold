//! Null-before snapshot guards, nvcc 13.5 SM120a/121a; build-r11-kernels.log records zero stack and spills.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");
const empty: limits.Usage = .{ .regs = 0, .static = 0 };
pub const symbols = [_][]const u8{
    "tf_q35_gdn_slots_1", "tf_q35_gdn_slots_2",  "tf_q35_gdn_slots_4",
    "tf_q35_gdn_slots_8", "tf_q35_gdn_slots_16", "tf_q35_conv_commit",
};
const native = [_]limits.Usage{
    .{ .regs = 64, .static = 4096, .bound = 128 },
    .{ .regs = 64, .static = 8192, .bound = 128 },
    .{ .regs = 80, .static = 16384, .bound = 128 },
    .{ .regs = 87, .static = 16384, .bound = 128 },
    .{ .regs = 94, .static = 32768, .bound = 128 },
    .{ .regs = 24, .static = 0, .bound = 0 },
};
pub const measured = blk: {
    var out: [symbols.len][limits.sms.len]limits.Usage = undefined;
    for (&out, native) |*row, value| row.* = .{ empty, empty, empty, value, value };
    break :blk out;
};

pub fn index(image: []const u8, symbol: []const u8) ?usize {
    if (!std.mem.eql(u8, image, "qwen35")) return null;
    for (symbols, 0..) |name, i| if (std.mem.eql(u8, name, symbol)) return i;
    return null;
}

pub fn lookup(record: r.CudaRecord, release: ?limits.Nvcc) ?[limits.sms.len]limits.Usage {
    const i = index(record.launch.image, record.launch.symbol) orelse return null;
    const version = release orelse return r.none;
    if (version.major == 13 and version.minor == 5) return measured[i];
    return r.none;
}

test "committed root receipts: six existing launches use only measured compiler and SM profiles" {
    for (symbols, 0..) |symbol, i| {
        var count: usize = 0;
        for (r.all_cuda) |record| {
            if (index(record.launch.image, record.launch.symbol) != i) continue;
            count += 1;
            try std.testing.expectEqual(r.none, r.usageFor(record, null));
            try std.testing.expectEqual(measured[i], r.usageFor(record, .{ .major = 13, .minor = 5 }));
            for ([_]limits.Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 3 }, .{ .major = 13, .minor = 6 } }) |version|
                try std.testing.expectEqual(r.none, r.usageFor(record, version));
        }
        _ = symbol;
        try std.testing.expectEqual(@as(usize, 1), count);
    }
}

test "committed root receipts: declared usage retains exact registers shared bytes and zero local memory" {
    for (measured, native) |row, value| {
        for (row[0..3]) |usage| try std.testing.expectEqual(empty, usage);
        try std.testing.expectEqual(value, row[3]);
        try std.testing.expectEqual(value, row[4]);
        try std.testing.expectEqual(@as(u32, 0), value.local);
    }
}
