//! Gate 5's records of plan_rows.hip's values kernels: a multi-row launch's rows applied alone, and its slots' chunks.
const limits = @import("limits.zig");
const Record = @import("hip_records.zig").Record;

const U = limits.HipUsage;

/// 256 threads, the weights in static LDS: kMaxGroup heads of one row alone, kApplyRows rows of kApplyHeads in a chunk.
fn rowsApply(comptime symbol: []const u8, usage: [limits.rdna.len]U) Record {
    const name = "planned walk " ++ symbol;
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "plan_rows", .threads = 256 }, .usage = usage };
}

pub const records = [4]Record{
    rowsApply("tf_plan_lone_apply_f16", .{
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
    }),
    rowsApply("tf_plan_lone_apply_bf16", .{
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 42, .sgprs = 44, .lds = 8192, .private = 0, .bound = 256, .wave = 32 },
    }),
    rowsApply("tf_plan_rows_apply_f16", .{
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 60, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
    }),
    rowsApply("tf_plan_rows_apply_bf16", .{
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 61, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
        .{ .vgprs = 81, .sgprs = 60, .lds = 16384, .private = 0, .bound = 256, .wave = 32 },
    }),
};
