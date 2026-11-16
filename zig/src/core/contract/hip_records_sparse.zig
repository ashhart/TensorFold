//! Gate 5's records of Flash Next's indexer block scores and sparse attention on HIP.
const std = @import("std");
const hip_records = @import("hip_records.zig");
const Record = hip_records.Record;

/// Flash Next's lanes sparse attention, 12 query heads a kv head, one cache kind (`early` VGPRs: gfx1030, gfx1100).
fn sparseLanes(comptime kind: []const u8, comptime early: u32, comptime late: u32) Record {
    const symbol = "tf_fn_sparse_lanes12_" ++ kind;
    return .{
        .launch = .{ .name = "flashnext " ++ symbol, .symbol = symbol, .image = "flashnext", .threads = 256 },
        .usage = .{
            .{ .vgprs = early, .sgprs = 55, .lds = 31264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = early, .sgprs = 55, .lds = 31264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = late, .sgprs = 57, .lds = 31264, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = late, .sgprs = 59, .lds = 31264, .private = 0, .bound = 256, .wave = 32 },
        },
    };
}

pub const records = [_]Record{
    .{
        .launch = .{
            .name = "flashnext tf_fn_idx_scores",
            .symbol = "tf_fn_idx_scores",
            .image = "flashnext",
            .threads = 256,
        },
        .usage = .{
            .{ .vgprs = 24, .sgprs = 39, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 39, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 24, .sgprs = 39, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 23, .sgprs = 40, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    sparseLanes("f16", 74, 72),
    sparseLanes("bf16", 84, 83),
};

test "tf_fn_idx_scores carries its LDS-staged Gate 5 record and fits every served RDNA chip" {
    var found: ?Record = null;
    for (hip_records.records) |r| if (std.mem.eql(u8, r.launch.symbol, "tf_fn_idx_scores")) {
        found = r;
    };
    const rec = found orelse return error.MissingRecord;
    try std.testing.expectEqual(@as(u32, 16384), rec.usage[0].lds);
    try std.testing.expectEqual(@as(usize, 0), try hip_records.check(std.testing.allocator, &.{rec}, true));
}
