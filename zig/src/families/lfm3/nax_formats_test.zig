//! Every tensor-unit prompt format has its launch-limit record, with the launch the route dispatches.
const std = @import("std");
const nax = @import("nax_formats.zig");
const records = @import("../../core/contract/lfm3_records.zig");

fn recordFor(bits: u8, group: u16) ?@TypeOf(records.metal[0]) {
    var buf: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "LFM3 NAX prompt q{d}g{d} tf_qmmw_t_nax_win", .{ bits, group }) catch unreachable;
    for (records.metal) |d| if (std.mem.eql(u8, d.name, name)) return d;
    return null;
}

test "LFM3 NAX prompt: each offered format has one record for the launch it dispatches, tensor-op only" {
    for (nax.formats, 0..) |f, i| {
        for (nax.formats[0..i]) |earlier| try std.testing.expect(earlier.bits != f.bits or earlier.group != f.group);
        const d = recordFor(f.bits, f.group) orelse {
            std.debug.print("no launch record for q{d}g{d}\n", .{ f.bits, f.group });
            return error.MissingNaxLaunchRecord;
        };
        try std.testing.expectEqual(@as(u32, nax.threads), d.threads);
        try std.testing.expectEqual(@as(u32, nax.threadgroup_bytes), d.tg_memory);
        try std.testing.expectEqual(@as(u32, nax.threads), d.reserved);
        try std.testing.expectEqual(@as(u32, 0), d.dynamic);
        try std.testing.expect(!d.offline); // Xcode's translator cannot build tensor-op kernels for M1/M2
    }
}

test "LFM3 NAX prompt: only the offered formats have records, and only they are supported" {
    var recorded: usize = 0;
    for (records.metal) |d| {
        if (!std.mem.startsWith(u8, d.name, "LFM3 NAX prompt ")) continue;
        recorded += 1;
        const tag = d.name["LFM3 NAX prompt q".len..std.mem.lastIndexOfScalar(u8, d.name, ' ').?];
        const split = std.mem.indexOfScalar(u8, tag, 'g').?;
        try std.testing.expect(nax.supported(try std.fmt.parseInt(u8, tag[0..split], 10), try std.fmt.parseInt(u16, tag[split + 1 ..], 10)));
    }
    try std.testing.expectEqual(nax.formats.len, recorded);
    for ([_]u8{ 2, 3, 5, 7, 16 }) |bits| for ([_]u16{ 32, 64 }) |group| try std.testing.expect(!nax.supported(bits, group));
    for ([_]u16{ 0, 16, 48, 96, 128 }) |group| for ([_]u8{ 4, 6, 8 }) |bits| try std.testing.expect(!nax.supported(bits, group));
    try std.testing.expect(nax.supported(6, 32) and nax.supported(8, 32));
}
