//! Gate 5's records of plan_rows.hip's kernels: a multi-row launch's rows that score alone, and its slots' chunks.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// 128 threads, the queries of `rows` rows of at most kMaxGroup heads of 256 in dynamic LDS.
fn rowsScore(comptime symbol: []const u8, comptime rows: u32, usage: [limits.rdna.len]U) Record {
    const name = "planned walk " ++ symbol;
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "plan_rows", .threads = 128, .dynamic = rows * 16 * 256 * 4 }, .usage = usage };
}

pub const records = [4]Record{
    rowsScore("tf_plan_lone_score256_f16", 1, .{
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    rowsScore("tf_plan_lone_score256_bf16", 1, .{
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 108, .sgprs = 77, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    rowsScore("tf_plan_rows_score256_f16", 2, .{
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
    rowsScore("tf_plan_rows_score256_bf16", 2, .{
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
        .{ .vgprs = 115, .sgprs = 95, .lds = 0, .private = 0, .bound = 128, .wave = 32 },
    }),
};
