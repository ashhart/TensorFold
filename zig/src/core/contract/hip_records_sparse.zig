//! Gate 5's records of Flash Next's lanes sparse attention (tf_fn_sparse_lanes12_*).
const Record = @import("hip_records.zig").Record;

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
    sparseLanes("f16", 74, 72),
    sparseLanes("bf16", 84, 83),
};
